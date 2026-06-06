#!/usr/bin/env python3
"""
control_loop.py — Boucle de contrôle unique (FF + PID longueur)
================================================================
Tick à TARGET_HZ : heartbeat + ESTOP, lectures IMU/ADC, filtrage,
cinématique directe, FF (Sarosi + équilibre statique), PID longueur
(domaine pression OU force/Gattringer), safety + envoi vanne, calibration,
logging, callbacks de séquence automatique.

API : start/stop, start/reset_calibration, start/abort_auto_sequence,
auto_seq_running, zero_all_valves, reset_pid_state.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

# ── Hardware ──────────────────────────────────────────────────────────────────
from imu import read_imu
from dac import set_muscle_pressure
from position_adc import read_positions, apply_position_correction
from pressure_adc import read_pressures

# ── Domaine ───────────────────────────────────────────────────────────────────
from geometry import VALVE_OF_MUSCLE
from stewart_forward_kinematics import stewart_forward_kinematics
from stewart_feedforward import (
    compute_feedforward_pressures,
    pam_pressure_sarosi_bar,
    geom_to_muscle_lengths,
    P_MIN_BAR, P_MAX_BAR,
)
from calibration import compute_position_offsets, compute_imu_offsets
from platform_pose import Vec3, IMUSnap
from position_filters import PositionFilterBank

# ── État partagé ──────────────────────────────────────────────────────────────
from robot_state import RobotState

# ── Auto-séquence ─────────────────────────────────────────────────────────────
from auto_sequence import (
    SEQUENCE as AUTO_SEQUENCE,
    HOME as AUTO_HOME,
)

# ═════════════════════════════════════════════════════════════════════════════
# Constantes de cadence
# ═════════════════════════════════════════════════════════════════════════════
# 13 Hz : cadence max ADC. Dynamique plateforme ~3 Hz → fs/fc ≈ 4.
TARGET_HZ         = 13.0
LOOP_PERIOD_S     = 1.0 / TARGET_HZ

CALIB_DURATION_S  = 3.0
GRAVITY           = 9.81        # m/s²
P_MIN             = P_MIN_BAR   # bar
P_MAX             = P_MAX_BAR   # bar


# ═════════════════════════════════════════════════════════════════════════════
# ControlLoop
# ═════════════════════════════════════════════════════════════════════════════
class ControlLoop:
    """Boucle de contrôle (TARGET_HZ), indépendante de l'UI.

    Threads internes : control loop + auto-sequence worker.
    Synchronisation via RobotState.lock, ControlConfig, _bus_lock et
    callbacks vers le main thread par la GUI.
    """

    def __init__(
        self,
        state: RobotState,
        hw,                        # HardwareBuses
        bus_lock: threading.Lock,
        logger,                    # LogWriter
        pid_ctrl,                  # MuscleLengthPID
        rest_lengths: list,
        on_calibration_done: Callable,
        on_status_update: Callable,
        on_seq_finished: Callable,
        on_pose_request: Callable,
        force_pid_ctrl=None,       # MuscleLengthForcePID (feedback_domain='force')
    ):
        self.state       = state
        self.hw          = hw
        self.bus_imu     = hw.bus_imu
        self.bus_adc     = hw.bus_adc
        self._pressure_adc_ok = hw.pressure_adc_ok
        self._bus_lock   = bus_lock
        self._logger     = logger
        self.pid_ctrl    = pid_ctrl
        # PID force (Gattringer). Créé si non fourni pour que
        # feedback_domain='force' fonctionne sans dépendre du câblage GUI.
        if force_pid_ctrl is None:
            from pid_controller import MuscleLengthForcePID
            force_pid_ctrl = MuscleLengthForcePID()
        self.force_pid_ctrl = force_pid_ctrl
        self.rest_lengths = list(rest_lengths)

        # Callbacks → GUI
        self._on_calibration_done = on_calibration_done
        self._on_status           = on_status_update
        self._on_seq_finished     = on_seq_finished
        self._on_pose_request     = on_pose_request

        # ── État interne (non partagé) ────────────────────────────────────────
        self._running     = False
        self._loop_thread: Optional[threading.Thread] = None

        # Calibration
        self._calib_buf_imu: list = []
        self._calib_buf_pos: list = []
        self._calib_start = 0.0

        # FK warm-start
        self._fk_translation_m = (0.0, 0.0, 0.0)
        self._fk_rotation_rad  = (0.0, 0.0, 0.0)
        self._fk_info = {
            'converged': False, 'iterations': 0,
            'residual_norm': 0.0, 'max_residual': 0.0,
            'valid': False, 'reason': 'not calibrated',
        }

        # dt tracking (utilisé par le filtre de position)
        self._last_loop_time = 0.0

        # Auto-séquence
        self._auto_seq_running = False
        self._auto_seq_thread: Optional[threading.Thread] = None
        self._auto_seq_abort   = threading.Event()

        # Cache local pour le log (rempli et lu dans la boucle)
        self._ff_pressures_bar  = [0.0] * 6
        self._ff_tensions_N     = [0.0] * 6
        self._ff_kappa          = [0.0] * 6
        self._ff_saturated      = [False] * 6
        self._ff_neg_tension    = [False] * 6
        self._pid_outputs_bar   = [0.0] * 6
        self._pid_gate_open     = [False] * 6
        self._pid_length_error  = [0.0] * 6
        self._feedback_domain   = 'pressure'
        self._pid_force_outputs_N = [0.0] * 6
        self._cmd_total_force_N   = [0.0] * 6

        # Filtre position
        self._pos_filter = PositionFilterBank(fc_hz=0.0, n_muscles=6)
        self._pos_filter_active = False
        self._pos_filter_last_fc = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Cycle de vie
    # ─────────────────────────────────────────────────────────────────────────
    def start(self):
        """Démarre le thread de contrôle (idempotent)."""
        if self._running:
            return
        self._running = True
        self._loop_thread = threading.Thread(
            target=self._control_loop, name="ControlLoop", daemon=True)
        self._loop_thread.start()

    def stop(self):
        """Arrête proprement le thread de contrôle."""
        self._running = False
        self._auto_seq_abort.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Boucle de contrôle — TARGET_HZ
    # ─────────────────────────────────────────────────────────────────────────
    def _control_loop(self):
        """Tourne à TARGET_HZ. Cadence verrouillée par timer perf_counter."""
        next_tick = time.perf_counter()
        while self._running:
            t0 = time.perf_counter()

            # Heartbeat unique pour le watchdog (ce thread commande les vannes).
            with self.state.lock:
                self.state.heartbeat_main_loop = t0
                estop_active = self.state.estop_requested
                calibrating  = self.state.is_calibrating

            if estop_active:
                next_tick = self._sleep_until_next(next_tick, LOOP_PERIOD_S)
                continue

            # ── Lecture IMU ────────────────────────────────────────────────────
            try:
                raw = read_imu(self.bus_imu)
                raw_imu = IMUSnap(
                    accel=Vec3(raw.accel.x, raw.accel.y, raw.accel.z),
                    gyro =Vec3(raw.gyro.x,  raw.gyro.y,  raw.gyro.z),
                )
            except Exception as e:
                print(f"[ERROR] IMU : {e}")
                raw_imu = IMUSnap(Vec3(), Vec3())

            # ── Lecture positions ──────────────────────────────────────────────
            try:
                with self._bus_lock:
                    raw_pos = read_positions(self.bus_adc)
            except Exception as e:
                print(f"[ERROR] POS : {e}")
                raw_pos = [0.0] * 6

            # ── Application offsets ────────────────────────────────────────────
            with self.state.lock:
                off_i = self.state.imu_offset[:]
                off_p = self.state.pos_offset[:]

            corrected_imu = IMUSnap(
                accel=Vec3(
                    raw_imu.accel.x - off_i[0],
                    raw_imu.accel.y - off_i[1],
                    raw_imu.accel.z - off_i[2],
                ),
                gyro=Vec3(
                    raw_imu.gyro.x - off_i[3],
                    raw_imu.gyro.y - off_i[4],
                    raw_imu.gyro.z - off_i[5],
                ),
            )
            corrected_pos = apply_position_correction(raw_pos, off_p)

            # ── Filtrage optionnel des positions ───────────────────────────────
            with self.state.lock:
                pf_en = self.state.config.position_filter_enabled
                pf_fc = float(self.state.config.position_filter_fc_hz)
            if pf_en and not calibrating:
                if (not self._pos_filter_active
                        or pf_fc != self._pos_filter_last_fc):
                    self._pos_filter.set_params(
                        fc_hz=pf_fc, seed_values=corrected_pos,
                    )
                    self._pos_filter_active = True
                    self._pos_filter_last_fc = pf_fc
                dt_loop = (LOOP_PERIOD_S if self._last_loop_time <= 0.0
                           else max(1e-3, t0 - self._last_loop_time))
                corrected_pos = self._pos_filter.update(corrected_pos, dt_loop)
            else:
                if self._pos_filter_active:
                    self._pos_filter_active = False
            self._last_loop_time = t0

            # Mise à jour de l'état partagé (capteurs)
            with self.state.lock:
                self.state.imu           = corrected_imu
                self.state.positions_m   = list(corrected_pos)
                calibrating = self.state.is_calibrating

            # ── Cinématique directe ────────────────────────────────────────────
            self._update_fk(corrected_pos, calibrating)

            # ── Lecture pressions mesurées (monitoring) ───────────────────────
            self._read_pressures()

            # ── Contrôle FF + PID longueur → consigne pression ────────────────
            with self.state.lock:
                ff_on  = self.state.config.ff_enabled
                pid_on = self.state.config.pid_enabled

            if (ff_on or pid_on) and not calibrating:
                p_setpoint = self._compute_setpoint(corrected_pos, t0)
            else:
                # Pas d'activité de contrôle → setpoint à zéro.
                p_setpoint = [0.0] * 6

            # ── Snapshot atomique des paramètres pour l'envoi vanne ───────────
            with self.state.lock:
                cfg              = self.state.config
                min_length       = max(0.5, min(1.0, cfg.min_length_m))
                active_mask      = cfg.muscle_active[:]
                manual_mode      = self.state.manual_valve_mode
                manual_setpoint  = list(self.state.manual_pressure_setpoints_bar)

            outer_active = ff_on or pid_on

            # ── Sommation finale + safety + envoi vannes ──────────────────────
            pressure_applied = [0.0] * 6
            for m in range(6):
                # ── Consigne de base selon le mode actif ──────────────────
                if outer_active:
                    p_total = float(p_setpoint[m])
                elif manual_mode:
                    # Aucun contrôleur, mais mode vannes manuelles : on tient
                    # la consigne réglée à la main au lieu de forcer 0.
                    p_total = float(manual_setpoint[m])
                else:
                    # Aucun contrôleur, pas de mode manuel : vanne au repos.
                    p_total = 0.0

                # ── Clamps de sécurité communs (contrôleur ET manuel) ─────
                # Un 0 strict reste 0 (sécurité / vanne désactivée).
                if p_total != 0.0:
                    p_total = max(P_MIN, min(P_MAX, p_total))
                    # Safety : longueur sous le seuil de sécurité → 0 bar
                    L_meas = (corrected_pos[m] if m < len(corrected_pos)
                              else 1.0)
                    if L_meas < min_length:
                        p_total = 0.0
                    if not active_mask[m]:
                        p_total = 0.0

                hw_valve = VALVE_OF_MUSCLE[m]
                try:
                    with self._bus_lock:
                        set_muscle_pressure(self.bus_adc, hw_valve, p_total)
                except Exception as e:
                    print(f"[ERROR] CTRL M{m} (V{hw_valve}) : {e}")
                pressure_applied[m] = p_total

            # ── Mise à jour de l'état partagé ─────────────────────────────────
            with self.state.lock:
                self.state.pressures_commanded_bar = list(pressure_applied)

            # API legacy : passe la pression appliquée au PID (no-op fonctionnel).
            if pid_on:
                try:
                    self.pid_ctrl.record_applied_pressures(pressure_applied)
                except Exception:
                    pass

            # ── Accumulation calibration ───────────────────────────────────────
            if calibrating:
                self._calib_buf_imu.append([
                    raw_imu.accel.x, raw_imu.accel.y, raw_imu.accel.z,
                    raw_imu.gyro.x,  raw_imu.gyro.y,  raw_imu.gyro.z,
                ])
                self._calib_buf_pos.append(list(raw_pos))
                if (time.perf_counter() - self._calib_start) >= CALIB_DURATION_S:
                    self._finish_calibration()

            # ── Logging CSV ────────────────────────────────────────────────────
            with self.state.lock:
                do_log = self.state.logging_active
            if do_log:
                self._write_log_row(corrected_imu, corrected_pos)

            # ── Cadence TARGET_HZ stricte ─────────────────────────────────────
            next_tick = self._sleep_until_next(next_tick, LOOP_PERIOD_S)

    def _read_pressures(self):
        """Lit le bus pression et met à jour state.pressures_measured_bar.
        Retourne la liste 6-éléments en bar (zéros si capteur indisponible)."""
        if not self._pressure_adc_ok:
            p_meas = [0.0] * 6
        else:
            try:
                with self._bus_lock:
                    p_meas = read_pressures(self.bus_adc)
            except Exception as e:
                print(f"[ERROR] P_MES : {e}")
                p_meas = [0.0] * 6
        with self.state.lock:
            self.state.pressures_measured_bar = list(p_meas)
        return p_meas

    @staticmethod
    def _sleep_until_next(next_tick: float, period_s: float) -> float:
        """Endort jusqu'au prochain tick (cadence stricte avec rattrapage).
        Retourne le timestamp du prochain tick à viser."""
        next_tick += period_s
        now = time.perf_counter()
        sleep_t = next_tick - now
        if sleep_t > 0:
            time.sleep(sleep_t)
        else:
            # On a pris du retard : on recale next_tick sur maintenant pour
            # éviter une cascade d'overruns rattrapés sans sleep.
            next_tick = now
        return next_tick

    # ─────────────────────────────────────────────────────────────────────────
    # Cinématique directe
    # ─────────────────────────────────────────────────────────────────────────
    def _update_fk(self, corrected_pos: list, calibrating: bool):
        with self.state.lock:
            calib_done = self.state.calibration_done

        if calib_done and not calibrating:
            try:
                measured_L = [self.rest_lengths[i] + (corrected_pos[i] - 1.0)
                              for i in range(6)]
                t_est, r_est, fk_info = stewart_forward_kinematics(
                    measured_L,
                    initial_translation_m=self._fk_translation_m,
                    initial_rotation_rad=self._fk_rotation_rad,
                )
                fk_valid = (fk_info['converged']
                            and fk_info['residual_norm'] < 5e-3)
                fk_info['valid']  = fk_valid
                fk_info['reason'] = 'ok' if fk_valid else (
                    'no convergence' if not fk_info['converged']
                    else f"residual {fk_info['residual_norm']*1000:.1f} mm > 5 mm")

                if fk_valid:
                    self._fk_translation_m = tuple(t_est)
                    self._fk_rotation_rad  = tuple(r_est)
                self._fk_info = fk_info

            except Exception as e:
                print(f"[ERROR] FK : {e}")
                self._fk_info = {**self._fk_info,
                                 'valid': False, 'reason': f'exception: {e}'}
        else:
            self._fk_translation_m = (0.0, 0.0, 0.0)
            self._fk_rotation_rad  = (0.0, 0.0, 0.0)
            self._fk_info = {
                'converged': False, 'iterations': 0,
                'residual_norm': 0.0, 'max_residual': 0.0,
                'valid': False,
                'reason': 'calibrating' if calibrating else 'not calibrated',
            }

        with self.state.lock:
            self.state.fk_translation_m = self._fk_translation_m
            self.state.fk_rotation_rad  = self._fk_rotation_rad
            self.state.fk_info          = self._fk_info

    # ─────────────────────────────────────────────────────────────────────────
    # FF + PID longueur → consigne de pression (renvoyée à l'appelant)
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_setpoint(self, measured_positions: list, t_now: float) -> list:
        """Calcule p_setpoint = clamp(p_ff + p_pid_len) et le retourne.

        La consigne est aussi déposée dans state pour le monitoring/log.
        """
        with self.state.lock:
            cfg          = self.state.config
            ff_on        = cfg.ff_enabled
            pid_on       = cfg.pid_enabled
            feedback_dom = getattr(cfg, 'feedback_domain', 'pressure')
            mass         = cfg.mass_kg
            com_offset   = cfg.com_offset_m
            active_mask  = cfg.muscle_active[:]
            target_trans = self.state.target_translation_m
            target_rot   = self.state.target_rotation_rad
            target_L_ref = self.state.target_lengths_m[:]
            p_meas_state = self.state.pressures_measured_bar[:]

        # ── Calcul Feedforward ───────────────────────────────────────────────
        if ff_on:
            try:
                cmd = compute_feedforward_pressures(
                    target_translation_m=target_trans,
                    target_rotation_rad=target_rot,
                    mass_kg=mass,
                    com_offset_m=com_offset,
                    gravity=GRAVITY,
                )
            except Exception as e:
                print(f"[ERROR] FF compute : {e}")
                return [0.0] * 6
            ff_pressures = cmd.pressures_bar.tolist()
            target_L     = cmd.target_lengths_m.tolist()
            ff_tensions  = cmd.tensions_N.tolist()
            ff_kappa     = cmd.kappa.tolist()
            ff_sat       = cmd.saturated_mask.tolist()
            ff_neg       = cmd.negative_tension.tolist()
        else:
            ff_pressures = [0.0] * 6
            target_L     = list(target_L_ref)
            ff_tensions  = [0.0] * 6
            ff_kappa     = [0.0] * 6
            ff_sat       = [False] * 6
            ff_neg       = [False] * 6

        # PID longueur — 'pressure' (historique) ou 'force' (Gattringer).
        pid_out          = [0.0] * 6   # bar (domaine pression)
        pid_force_out_N  = [0.0] * 6   # N   (domaine force)
        cmd_total_F_N    = list(ff_tensions)
        gate_open        = [False] * 6
        length_err       = [0.0] * 6

        if pid_on:
            measured_L_geom = [self.rest_lengths[i]
                               + (measured_positions[i] - 1.0)
                               for i in range(6)]
            length_err = [measured_L_geom[i] - target_L[i] for i in range(6)]

            if feedback_dom == 'force':
                try:
                    dF = self.force_pid_ctrl.step(
                        target_lengths_m=target_L,
                        measured_lengths_m=measured_L_geom,
                        ff_tensions_N=ff_tensions,
                        t_now=t_now,
                        active_mask=active_mask,
                    )
                    pid_force_out_N = dF.tolist()
                except Exception as e:
                    print(f"[ERROR] PID force step : {e}")
                    pid_force_out_N = [0.0] * 6
                # Force totale = tension FF + correction, bornée ≥ 0
                # (les PAM ne tirent que vers le haut).
                cmd_total_F_N = [max(0.0, float(ff_tensions[m]) + pid_force_out_N[m])
                                 for m in range(6)]
                gate_open = [bool(active_mask[m]) for m in range(6)]
            else:
                try:
                    pid_out, gate_open = self.pid_ctrl.step(
                        target_lengths_m=target_L,
                        measured_lengths_m=measured_L_geom,
                        p_measured_bar=p_meas_state,
                        ff_pressures_bar=ff_pressures,
                        t_now=t_now,
                        active_mask=active_mask,
                    )
                    pid_out    = pid_out.tolist()
                    gate_open  = gate_open.tolist()
                except Exception as e:
                    print(f"[ERROR] PID longueur step : {e}")
                    pid_out = [0.0] * 6
                    gate_open = [False] * 6

        # Consigne pression — clampée [P_MIN, P_MAX], envoyée directement.
        p_setpoint = [0.0] * 6
        if pid_on and feedback_dom == 'force':
            # Sarosi inverse à la longueur CIBLE (Gattringer fig.4, h_d).
            target_L_musc = geom_to_muscle_lengths(target_L).tolist()
            for m in range(6):
                try:
                    ps, _info = pam_pressure_sarosi_bar(
                        cmd_total_F_N[m], target_L_musc[m])
                except Exception as e:
                    print(f"[ERROR] Sarosi inverse (muscle {m}) : {e}")
                    ps = 0.0
                ps = max(P_MIN, min(P_MAX, ps))
                if not active_mask[m]:
                    ps = 0.0
                p_setpoint[m] = ps
        else:
            for m in range(6):
                ps = float(ff_pressures[m]) + float(pid_out[m])
                ps = max(P_MIN, min(P_MAX, ps))
                if not active_mask[m]:
                    ps = 0.0
                p_setpoint[m] = ps

        # ── Mise à jour de l'état partagé ───────────────────────────────────
        with self.state.lock:
            self.state.ff_pressures_bar  = list(ff_pressures)
            self.state.ff_tensions_N     = list(ff_tensions)
            self.state.ff_kappa          = list(ff_kappa)
            self.state.ff_saturated      = list(ff_sat)
            self.state.ff_neg_tension    = list(ff_neg)
            self.state.pid_enabled       = pid_on
            self.state.pid_outputs_bar   = list(pid_out)
            self.state.pid_gate_open     = list(gate_open)
            self.state.pid_length_error_m = list(length_err)
            self.state.pid_force_outputs_N = list(pid_force_out_N)
            self.state.cmd_total_force_N   = list(cmd_total_F_N)
            if pid_on:
                if feedback_dom == 'force':
                    self.state.pid_integral_m_s = \
                        self.force_pid_ctrl.integral_m_s.tolist()
                else:
                    self.state.pid_integral_m_s = self.pid_ctrl.integral_m_s.tolist()

            # Erreur de pose cible ↔ mesurée (monitoring)
            fk_t = self.state.fk_translation_m
            fk_r = self.state.fk_rotation_rad
            self.state.pose_error_t_m = tuple(
                target_trans[i] - fk_t[i] for i in range(3))
            self.state.pose_error_r_rad = tuple(
                target_rot[i] - fk_r[i] for i in range(3))

        # Cache local pour le log
        self._ff_pressures_bar  = list(ff_pressures)
        self._ff_tensions_N     = list(ff_tensions)
        self._ff_kappa          = list(ff_kappa)
        self._ff_saturated      = list(ff_sat)
        self._ff_neg_tension    = list(ff_neg)
        self._pid_outputs_bar   = list(pid_out)
        self._pid_gate_open     = list(gate_open)
        self._pid_length_error  = list(length_err)
        self._feedback_domain   = feedback_dom
        self._pid_force_outputs_N = list(pid_force_out_N)
        self._cmd_total_force_N   = list(cmd_total_F_N)

        return p_setpoint

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration
    # ─────────────────────────────────────────────────────────────────────────
    def start_calibration(self):
        """Démarre l'accumulation des échantillons de calibration."""
        with self.state.lock:
            if self.state.is_calibrating:
                return
            self._calib_buf_imu = []
            self._calib_buf_pos = []
            self._calib_start   = time.perf_counter()
            self.state.is_calibrating = True

    def _finish_calibration(self):
        new_imu = compute_imu_offsets(self._calib_buf_imu)
        new_pos = compute_position_offsets(self._calib_buf_pos)

        with self.state.lock:
            self.state.imu_offset     = list(new_imu)
            self.state.pos_offset     = list(new_pos)
            self.state.is_calibrating = False
            self.state.calibration_done = True
            self._fk_translation_m = (0.0, 0.0, 0.0)
            self._fk_rotation_rad  = (0.0, 0.0, 0.0)

        self._pos_filter_active = False
        self._on_calibration_done()

    def reset_calibration(self):
        with self.state.lock:
            self.state.imu_offset     = [0.0] * 6
            self.state.pos_offset     = [0.0] * 6
            self.state.calibration_done = False
            self.state.is_calibrating   = False
            self._fk_translation_m = (0.0, 0.0, 0.0)
            self._fk_rotation_rad  = (0.0, 0.0, 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Séquence automatique
    # ─────────────────────────────────────────────────────────────────────────
    def start_auto_sequence(self):
        if self._auto_seq_running:
            return
        self._auto_seq_abort.clear()
        self._auto_seq_running = True
        self._auto_seq_thread = threading.Thread(
            target=self._auto_sequence_worker,
            name="AutoSequence", daemon=True,
        )
        self._auto_seq_thread.start()

    def abort_auto_sequence(self):
        self._auto_seq_abort.set()

    @property
    def auto_seq_running(self) -> bool:
        return self._auto_seq_running

    def _auto_sequence_worker(self):
        try:
            n = len(AUTO_SEQUENCE)
            self._on_status("▶  Séquence : aller à HOME initiale…")
            self._request_pose(AUTO_HOME)
            if not self._dwell_with_abort(AUTO_HOME.dwell_s):
                return

            for i, wp in enumerate(AUTO_SEQUENCE, start=1):
                if self._auto_seq_abort.is_set():
                    break
                self._on_status(
                    f"▶  Séquence {i}/{n} : « {wp.name} »  "
                    f"({wp.tx_m:+.3f}, {wp.ty_m:+.3f}, {wp.tz_m:+.3f}) m  "
                    f"({wp.rx_deg:+.1f}, {wp.ry_deg:+.1f}, {wp.rz_deg:+.1f})°  "
                    f"hold {wp.dwell_s:.0f} s")
                self._request_pose(wp)
                if not self._dwell_with_abort(wp.dwell_s):
                    break

                if i < n and not getattr(wp, 'skip_home_return', False):
                    self._on_status(
                        f"↩  Séquence {i}/{n} : retour HOME ({AUTO_HOME.dwell_s:.0f} s)")
                    self._request_pose(AUTO_HOME)
                    if not self._dwell_with_abort(AUTO_HOME.dwell_s):
                        break

        except Exception as e:
            print(f"[ERROR] Auto-sequence worker : {e}")
            self._on_status(f"✘  Séquence interrompue (erreur) : {e}")
        finally:
            try:
                self._request_pose(AUTO_HOME)
            except Exception:
                pass
            self._auto_seq_running = False
            self._on_seq_finished()

    def _request_pose(self, wp):
        self._on_pose_request(
            wp.tx_m, wp.ty_m, wp.tz_m,
            wp.rx_deg, wp.ry_deg, wp.rz_deg,
        )
        # Laisse un tick complet pour propagation IK → state
        time.sleep(LOOP_PERIOD_S * 1.5)

    def _dwell_with_abort(self, duration_s: float) -> bool:
        deadline = time.perf_counter() + duration_s
        while time.perf_counter() < deadline:
            if self._auto_seq_abort.is_set():
                return False
            remaining = deadline - time.perf_counter()
            time.sleep(min(0.1, max(0.0, remaining)))
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────
    def _write_log_row(self, imu: IMUSnap, corrected_pos: list):
        """Écrit une ligne CSV à la cadence TARGET_HZ."""
        if not self._logger.is_active:
            return
        with self.state.lock:
            target_t       = self.state.target_translation_m
            target_r       = self.state.target_rotation_rad
            target_lengths = list(self.state.target_lengths_m)
            fk_t           = self.state.fk_translation_m
            fk_r           = self.state.fk_rotation_rad
            fk_valid       = bool(self.state.fk_info.get('valid', False))
            p_meas         = self.state.pressures_measured_bar[:]
            p_applied      = self.state.pressures_commanded_bar[:]
            ff_on          = self.state.config.ff_enabled
            pid_on         = self.state.config.pid_enabled
            calib_in_prog  = self.state.is_calibrating

        measured_lengths = [
            self.rest_lengths[i] + (corrected_pos[i] - 1.0) for i in range(6)]

        _pid_terms_ok = pid_on and self._feedback_domain != 'force'
        pid_p = self.pid_ctrl.p_terms_bar.tolist() if _pid_terms_ok else [0.0]*6
        pid_i = self.pid_ctrl.i_terms_bar.tolist() if _pid_terms_ok else [0.0]*6
        pid_d = self.pid_ctrl.d_terms_bar.tolist() if _pid_terms_ok else [0.0]*6

        self._logger.write_row(
            target_translation_m=target_t,
            target_rotation_rad=target_r,
            fk_translation_m=fk_t,
            fk_rotation_rad=fk_r,
            fk_valid=fk_valid,
            target_lengths_m=target_lengths,
            measured_lengths_m=measured_lengths,
            target_pressures_bar=p_applied,
            measured_pressures_bar=p_meas,
            target_tensions_N=self._ff_tensions_N,
            ff_pressures_raw_bar=self._ff_pressures_bar,
            ff_kappa=self._ff_kappa,
            ff_saturated=self._ff_saturated,
            ff_neg_tension=self._ff_neg_tension,
            ff_enabled=ff_on,
            calibrating=calib_in_prog,
            imu_accel=(imu.accel.x, imu.accel.y, imu.accel.z),
            imu_gyro =(imu.gyro.x,  imu.gyro.y,  imu.gyro.z),
            pid_enabled=pid_on,
            pid_length_error_m=self._pid_length_error,
            pid_p_terms_bar=pid_p,
            pid_i_terms_bar=pid_i,
            pid_d_terms_bar=pid_d,
            pid_output_bar=self._pid_outputs_bar,
            # ── Domaine de retour + PID force (Gattringer) ───────────────
            feedback_domain=self._feedback_domain,
            pid_force_output_N=self._pid_force_outputs_N,
            cmd_total_force_N=self._cmd_total_force_N,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Utilitaires publics (appelés depuis le main thread)
    # ─────────────────────────────────────────────────────────────────────────
    def zero_all_valves(self):
        """Met toutes les vannes à 0 bar et coupe le mode manuel."""
        with self.state.lock:
            self.state.pressures_commanded_bar = [0.0] * 6
            self.state.manual_valve_mode = False
            self.state.manual_pressure_setpoints_bar = [0.0] * 6
        for m in range(6):
            try:
                with self._bus_lock:
                    set_muscle_pressure(self.bus_adc, VALVE_OF_MUSCLE[m], 0.0)
            except Exception:
                pass

    def reset_pid_state(self):
        """Reset PID longueur (intégrateur, compteurs), domaines pression ET force."""
        now = time.perf_counter()
        self.pid_ctrl.reset(t_now=now)
        if self.force_pid_ctrl is not None:
            self.force_pid_ctrl.reset(t_now=now)
        self._pid_outputs_bar  = [0.0] * 6
        self._pid_gate_open    = [False] * 6
        self._pid_length_error = [0.0] * 6
        self._pid_force_outputs_N = [0.0] * 6
        self._cmd_total_force_N   = [0.0] * 6
        with self.state.lock:
            self.state.pid_outputs_bar    = [0.0] * 6
            self.state.pid_integral_m_s   = [0.0] * 6
            self.state.pid_gate_open      = [False] * 6
            self.state.pid_length_error_m = [0.0] * 6
            self.state.pid_force_outputs_N = [0.0] * 6
            self.state.cmd_total_force_N   = [0.0] * 6
