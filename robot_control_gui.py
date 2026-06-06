#!/usr/bin/env python3
"""
robot_control_gui.py — Orchestration GUI + callbacks utilisateur
================================================================
Init état/hardware/ControlLoop, callbacks (FF/PID toggle, calibration,
logging, séquence auto, vannes manuelles, estop), fermeture propre.

Logique de contrôle : control_loop.py. Panneaux Tkinter : gui_panels.py.
État partagé : robot_state.py.
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import os
from datetime import datetime

# ── Hardware ──────────────────────────────────────────────────────────────────
from dac import set_muscle_pressure, clear_all
from hardware import HardwareBuses

# ── Domaine ───────────────────────────────────────────────────────────────────
from geometry import MUSCLE_LABELS, VALVE_OF_MUSCLE, stewart_inverse_kinematics
from stewart_feedforward import P_MIN_BAR, P_MAX_BAR, MUSCLE_NOMINAL_LENGTH_M
from pid_controller import MuscleLengthPID, PIDConfig
from workspace import find_workspace_center_z
from data_logger import LogWriter
from auto_sequence import (
    SEQUENCE as AUTO_SEQUENCE,
    total_duration_s as auto_seq_total_duration_s,
)

# ── État partagé + boucle de contrôle ────────────────────────────────────────
from robot_state import RobotState, SafetyEvent
from safety_watchdog import SafetyWatchdog
from control_loop import ControlLoop, TARGET_HZ as CONTROL_LOOP_HZ

# ── Mixin panneaux GUI ────────────────────────────────────────────────────────
from gui_panels import GUIPanelsMixin

# ── Thème UI ──────────────────────────────────────────────────────────────────
from ui_theme import (
    BG, GREEN, RED, ORANGE, PURPLE, YELLOW,
)

# ── Constantes ────────────────────────────────────────────────────────────────
TARGET_HZ     = CONTROL_LOOP_HZ
LOOP_PERIOD_S = 1.0 / TARGET_HZ
P_MAX         = P_MAX_BAR
P_MIN         = P_MIN_BAR
GRAVITY       = 9.81


# ═════════════════════════════════════════════════════════════════════════════
# Application
# ═════════════════════════════════════════════════════════════════════════════
class RobotGUI(GUIPanelsMixin):
    """Fenêtre principale. Hérite de GUIPanelsMixin pour tous les _build_*/_update_*."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Robot Pneumatique — Contrôle & Monitoring")
        self.root.configure(bg=BG)
        self.root.geometry("1280x1024")
        self.root.minsize(1100, 800)
        # Maximize au démarrage pour que tout soit visible sans scroll
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        # ── État partagé ──────────────────────────────────────────────────────
        self.state  = RobotState()
        self._lock  = self.state.lock   # alias de compatibilité

        # ── Variables Tkinter (GUI uniquement — jamais lues depuis un autre thread) ──
        self.pressure_vars    = [tk.StringVar(value="0.00") for _ in range(6)]
        self.pressure_applied = [0.0] * 6   # mode manuel
        self.muscle_active    = [tk.BooleanVar(value=True) for _ in range(6)]

        self.ff_enabled       = tk.BooleanVar(value=False)
        self.pid_enabled      = tk.BooleanVar(value=False)
        self.mass_var         = tk.StringVar(value="29.14909")   # kg
        self.min_length_var   = tk.StringVar(value="0.750")

        self.com_offset_x_var = tk.StringVar(value="0.000")
        self.com_offset_y_var = tk.StringVar(value="0.000")
        self.com_offset_z_var = tk.StringVar(value="0.000")

        # Gains PID pression (bar/m, bar/(m·s), bar·s/m).
        self.pid_kp_var = tk.StringVar(value="2")
        self.pid_ki_var = tk.StringVar(value="2")
        self.pid_kd_var = tk.StringVar(value="0.2")

        # Domaine de retour : 'pressure' (PID→bar+FF) ou 'force' (Gattringer).
        self.feedback_domain_var  = tk.StringVar(value="pressure")
        # Gains PID force (N/m, N/(m·s), N·s/m).
        self.pid_force_kp_var     = tk.StringVar(value="700.0")
        self.pid_force_ki_var     = tk.StringVar(value="500.0")
        self.pid_force_kd_var     = tk.StringVar(value="30.0")

        # Deadband (mm) commun aux deux PID. Sous ce seuil → consigne figée.
        self.pid_deadband_var     = tk.StringVar(value="7.0")

        # Cible IK (saisie GUI)
        self.target_vars = {
            "x": tk.StringVar(value="0.000"), "y": tk.StringVar(value="0.000"),
            "z": tk.StringVar(value="0.000"), "alpha": tk.StringVar(value="0.00"),
            "beta": tk.StringVar(value="0.00"), "gamma": tk.StringVar(value="0.00"),
        }

        # Longueurs cibles (résultat IK, partagées avec le mixin pour les marqueurs)
        self.rest_lengths   = stewart_inverse_kinematics((0.,0.,0.), (0.,0.,0.)).tolist()
        self.target_lengths = list(self.rest_lengths)

        # Centre du workspace (géométrique, calculé une fois)
        self._workspace_center_z = find_workspace_center_z()

        # Synchronisation des StringVar vers state.config à chaque changement
        self.mass_var.trace_add('write', self._sync_mass)
        self.min_length_var.trace_add('write', self._sync_min_length)
        self.com_offset_x_var.trace_add('write', self._sync_com)
        self.com_offset_y_var.trace_add('write', self._sync_com)
        self.com_offset_z_var.trace_add('write', self._sync_com)

        # ── Bus I2C ───────────────────────────────────────────────────────────
        try:
            self.hw = HardwareBuses()
            self.bus_imu = self.hw.bus_imu
            self.bus_adc = self.hw.bus_adc
            self._pressure_adc_ok = self.hw.pressure_adc_ok
        except Exception as e:
            messagebox.showerror("Erreur I2C",
                                 f"Impossible d'ouvrir les bus I2C :\n{e}")
            raise SystemExit(1)

        self._bus_lock = threading.Lock()

        # ── Logger CSV ────────────────────────────────────────────────────────
        self._logger = LogWriter()

        # Contrôleur PID pression. fc D = 6 Hz : aligné avec l'IIR position
        # amont, cascade homogène (~2e ordre Butterworth).
        self.pid_ctrl = MuscleLengthPID(PIDConfig(
            p_total_max_bar=P_MAX,
            p_total_min_bar=P_MIN,
            d_filter_cutoff_hz=6.0,
        ))

        # ── ControlLoop (thread unique) ─────────────────────────────────────
        self.ctrl = ControlLoop(
            state             = self.state,
            hw                = self.hw,
            bus_lock          = self._bus_lock,
            logger            = self._logger,
            pid_ctrl          = self.pid_ctrl,
            rest_lengths      = self.rest_lengths,
            on_calibration_done = lambda: self.root.after(0, self._on_calibration_done),
            on_status_update    = lambda msg: self.root.after(0, lambda m=msg: self.status_var.set(m)),
            on_seq_finished     = lambda: self.root.after(0, self._on_auto_sequence_finished),
            on_pose_request     = self._on_pose_request,
        )

        # ── Watchdog de sécurité ──────────────────────────────────────────────
        self.watchdog = SafetyWatchdog(
            self.state,
            bus_adc=self.bus_adc,
            bus_lock=self._bus_lock,
            on_event=self._on_safety_event,
        )
        self.watchdog.start()

        # ── Construction de l'UI (via GUIPanelsMixin._build_ui) ───────────────
        self._build_ui()

        # Synchronise le contrôleur force avec les gains/domaine par défaut.
        self._apply_force_pid_settings_from_gui()

        # ── Démarrage de la boucle de contrôle ───────────────────────────────
        self._running = True
        self.ctrl.start()

        # ── Rafraîchissement GUI 10 Hz (via GUIPanelsMixin._gui_update) ───────
        self._gui_update()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────────────────────
    # Synchronisation StringVar → state.config (trace callbacks, main thread)
    # ─────────────────────────────────────────────────────────────────────────
    def _sync_mass(self, *_):
        try:
            v = float(self.mass_var.get())
            with self.state.lock:
                self.state.config.mass_kg = v
        except ValueError:
            pass

    def _sync_min_length(self, *_):
        try:
            v = max(0.5, min(1.0, float(self.min_length_var.get())))
            with self.state.lock:
                self.state.config.min_length_m = v
        except ValueError:
            pass

    def _sync_com(self, *_):
        try:
            cx = float(self.com_offset_x_var.get())
            cy = float(self.com_offset_y_var.get())
            cz = float(self.com_offset_z_var.get())
            cx = max(-0.5, min(0.5, cx))
            cy = max(-0.5, min(0.5, cy))
            cz = max(-0.5, min(0.5, cz))
            with self.state.lock:
                self.state.config.com_offset_m = (cx, cy, cz)
        except ValueError:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # COM helpers (affichage GUI)
    # ─────────────────────────────────────────────────────────────────────────
    def _get_com_offset_m(self) -> tuple:
        try:
            cx = float(self.com_offset_x_var.get())
            cy = float(self.com_offset_y_var.get())
            cz = float(self.com_offset_z_var.get())
        except ValueError:
            return (0.0, 0.0, 0.0)
        return (max(-0.5, min(0.5, cx)),
                max(-0.5, min(0.5, cy)),
                max(-0.5, min(0.5, cz)))

    def _refresh_com_display(self) -> None:
        cx, cy, cz = self._get_com_offset_m()
        self.com_offset_x_var.set(f"{cx:.3f}")
        self.com_offset_y_var.set(f"{cy:.3f}")
        self.com_offset_z_var.set(f"{cz:.3f}")
        self.status_var.set(f"COM offset → ({cx:+.3f}, {cy:+.3f}, {cz:+.3f}) m")

    def _reset_com_offset(self) -> None:
        self.com_offset_x_var.set("0.000")
        self.com_offset_y_var.set("0.000")
        self.com_offset_z_var.set("0.000")
        self.status_var.set("COM offset remis à (0, 0, 0)")

    # ─────────────────────────────────────────────────────────────────────────
    # Toggle Feedforward / PID
    # ─────────────────────────────────────────────────────────────────────────
    def _toggle_feedforward(self):
        ff_on = self.ff_enabled.get()
        with self.state.lock:
            self.state.config.ff_enabled = ff_on

        if ff_on:
            self.ff_btn.config(text="⏸  Désactiver Feedforward", fg=RED)
            self.status_var.set("✅  Feedforward activé — pressions calculées Sarosi+Mehdi")
        else:
            # FF off → PID longueur off également
            if self.pid_enabled.get():
                self.pid_enabled.set(False)
                self._toggle_pid()
            self.ctrl.zero_all_valves()
            for m in range(6):
                self.pressure_vars[m].set("0.00")
                self.pressure_applied[m] = 0.0
                self._update_valve_display(m, 0.0)
            self.ff_btn.config(text="▶  Activer Feedforward", fg=GREEN)
            self.status_var.set("⏸  Feedforward désactivé — vannes remises à 0")

    def _toggle_pid(self):
        pid_on = self.pid_enabled.get()
        with self.state.lock:
            self.state.config.pid_enabled = pid_on

        if pid_on:
            # Cascade : PID on → FF on si pas déjà actif
            if not self.ff_enabled.get():
                self.ff_enabled.set(True)
                self._toggle_feedforward()
            self._apply_pid_settings_from_gui()
            self.ctrl.reset_pid_state()
            try:
                self.pid_btn.config(text="⏸  Désactiver PID", fg=RED)
            except Exception:
                pass
            self.status_var.set("✅  PID activé — correction ajoutée au feedforward")
        else:
            self.ctrl.reset_pid_state()
            with self.state.lock:
                self.state.config.pid_enabled = False
            try:
                self.pid_btn.config(text="▶  Activer PID", fg=GREEN)
            except Exception:
                pass
            self.status_var.set("⏸  PID désactivé — feedforward seul actif")

    def _apply_pid_settings_from_gui(self):
        """Pousse les gains PID PRESSION vers le contrôleur et state.config.
        Le PID force a son propre bouton (_apply_force_pid_settings_from_gui)."""
        try:
            kp = float(self.pid_kp_var.get())
            ki = float(self.pid_ki_var.get())
            kd = float(self.pid_kd_var.get())
        except ValueError:
            self.status_var.set("⚠️  Gain PID invalide — saisie ignorée")
            return
        self.pid_ctrl.set_gains(kp, ki, kd, reset_integral=False)
        with self.state.lock:
            self.state.config.pid_kp = kp
            self.state.config.pid_ki = ki
            self.state.config.pid_kd = kd
        self._apply_deadband_from_gui()

    def _apply_force_pid_settings_from_gui(self):
        """Pousse domaine + gains PID force vers state.config et le contrôleur."""
        domain = self.feedback_domain_var.get()
        if domain not in ('pressure', 'force'):
            domain = 'pressure'
        try:
            fkp = float(self.pid_force_kp_var.get())
            fki = float(self.pid_force_ki_var.get())
            fkd = float(self.pid_force_kd_var.get())
        except ValueError:
            self.status_var.set("⚠️  Gain PID force invalide — saisie ignorée")
            fkp = fki = fkd = None
        with self.state.lock:
            self.state.config.feedback_domain = domain
            if fkp is not None:
                self.state.config.pid_force_kp = fkp
                self.state.config.pid_force_ki = fki
                self.state.config.pid_force_kd = fkd
        if fkp is not None and getattr(self.ctrl, 'force_pid_ctrl', None) is not None:
            self.ctrl.force_pid_ctrl.set_gains(fkp, fki, fkd, reset_integral=False)
        self._apply_deadband_from_gui()

    def _apply_deadband_from_gui(self):
        """Pousse le deadband (mm) vers les DEUX PID longueur."""
        try:
            db_mm = float(self.pid_deadband_var.get())
        except ValueError:
            self.status_var.set("⚠️  Deadband invalide — saisie ignorée")
            return
        db_m = max(0.0, db_mm / 1000.0)
        if getattr(self.ctrl, 'force_pid_ctrl', None) is not None:
            self.ctrl.force_pid_ctrl.set_deadband(db_m)
        if getattr(self, 'pid_ctrl', None) is not None:
            self.pid_ctrl.set_deadband(db_m)

    def _toggle_feedback_domain(self):
        """Bascule pression ↔ force. Reset l'intégrateur pour éviter un saut."""
        self._apply_force_pid_settings_from_gui()
        self.ctrl.reset_pid_state()
        domain = self.feedback_domain_var.get()
        if domain == 'force':
            self.status_var.set(
                "🔁  Retour FORCE — PID→ΔF→Sarosi⁻¹ (gain de boucle linéarisé)")
        else:
            self.status_var.set(
                "🔁  Retour PRESSION — PID→bar ajouté au feedforward (historique)")

    def _reset_pid_state(self):
        self.ctrl.reset_pid_state()
        self.status_var.set("PID : intégrateur et compteurs remis à zéro")

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration
    # ─────────────────────────────────────────────────────────────────────────
    def _start_calibration(self):
        self.ctrl.start_calibration()
        self.calib_btn.config(text="⏳  Calibration…", state="disabled", fg=ORANGE)
        self.status_var.set("Calibration en cours — robot immobile…")

    def _on_calibration_done(self):
        """Appelée sur le main thread par le callback ControlLoop."""
        self.calib_btn.config(text="🎯  Calibrer", state="normal", fg=YELLOW)
        n = int(3.0 * TARGET_HZ)
        self.status_var.set(f"✅  Calibration terminée ({n} échantillons)")
        self._refresh_offset_display()

    def _reset_calibration(self):
        self.ctrl.reset_calibration()
        self._refresh_offset_display()
        self.status_var.set("Calibration réinitialisée (offsets à 0)")

    def _refresh_offset_display(self):
        with self.state.lock:
            io = self.state.imu_offset[:]
            po = self.state.pos_offset[:]
        for i, lbl in enumerate(["ax", "ay", "az", "gx", "gy", "gz"]):
            self.imu_off_labels[i].config(text=f"{lbl}: {io[i]:+.4f}")
        for i in range(6):
            self.pos_off_labels[i].config(text=f"off: {po[i]:+.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Vannes manuelles
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_pressure(self, muscle: int):
        try:
            val = float(self.pressure_vars[muscle].get())
        except ValueError:
            self.status_var.set(f"⚠️  Valeur invalide pour le muscle {MUSCLE_LABELS[muscle]}")
            return
        val = max(0.0, min(P_MAX, val))
        if not self.muscle_active[muscle].get():
            val = 0.0
            self.pressure_vars[muscle].set("0.00")
            self.status_var.set(
                f"⏸  Muscle {MUSCLE_LABELS[muscle]} désactivé — vanne forcée à 0 bar")
        else:
            self.pressure_vars[muscle].set(f"{val:.2f}")
        self.pressure_applied[muscle] = val
        hw_valve = VALVE_OF_MUSCLE[muscle]
        try:
            with self._bus_lock:
                set_muscle_pressure(self.bus_adc, hw_valve, val)
        except Exception as e:
            self.status_var.set(f"[ERROR] Muscle {MUSCLE_LABELS[muscle]} (V{hw_valve}) : {e}")
            return
        with self._lock:
            self.state.pressures_commanded_bar[muscle] = val
            # Mode manuel : mémorise + arme le drapeau pour que la boucle
            # maintienne la consigne (sinon écrasée à 0 sans contrôleur).
            self.state.manual_pressure_setpoints_bar[muscle] = val
            self.state.manual_valve_mode = True
        self._update_valve_display(muscle, val)
        if self.muscle_active[muscle].get():
            self.status_var.set(
                f"Muscle {MUSCLE_LABELS[muscle]} (V{hw_valve}) → {val:.2f} bar")

    def _apply_all_pressures(self):
        for i in range(6):
            self._apply_pressure(i)

    def _on_muscle_active_toggled(self, muscle: int):
        active = self.muscle_active[muscle].get()
        with self.state.lock:
            self.state.config.muscle_active[muscle] = active
        if not active:
            hw_valve = VALVE_OF_MUSCLE[muscle]
            try:
                with self._bus_lock:
                    set_muscle_pressure(self.bus_adc, hw_valve, 0.0)
            except Exception as e:
                self.status_var.set(
                    f"[ERROR] Désactivation M{muscle} (V{hw_valve}) : {e}")
                return
            self.pressure_vars[muscle].set("0.00")
            self.pressure_applied[muscle] = 0.0
            with self._lock:
                self.state.pressures_commanded_bar[muscle] = 0.0
                self.state.manual_pressure_setpoints_bar[muscle] = 0.0
            self._update_valve_display(muscle, 0.0)
            self.status_var.set(
                f"⏸  Muscle {MUSCLE_LABELS[muscle]} désactivé — vanne à 0 bar")
        else:
            self.status_var.set(
                f"▶  Muscle {MUSCLE_LABELS[muscle]} réactivé")

    # ─────────────────────────────────────────────────────────────────────────
    # Arrêt d'urgence
    # ─────────────────────────────────────────────────────────────────────────
    def _emergency_zero(self):
        self.state.request_estop(source='gui_button')

        # Coupe FF + PID immédiatement dans la GUI
        if self.ff_enabled.get():
            self.ff_enabled.set(False)
            with self.state.lock:
                self.state.config.ff_enabled = False
            try:
                self.ff_btn.config(text="▶  Activer Feedforward", fg=GREEN)
            except Exception:
                pass
        if self.pid_enabled.get():
            self.pid_enabled.set(False)
            with self.state.lock:
                self.state.config.pid_enabled = False
            try:
                self.pid_btn.config(text="▶  Activer PID", fg=GREEN)
            except Exception:
                pass
        self.pid_ctrl.reset()

        errors = []
        for m in range(6):
            hw_valve = VALVE_OF_MUSCLE[m]
            sent_ok = False
            for _ in range(2):
                try:
                    with self._bus_lock:
                        set_muscle_pressure(self.bus_adc, hw_valve, 0.0)
                    sent_ok = True
                    break
                except Exception as e:
                    last_err = e
            if not sent_ok:
                errors.append(f"{MUSCLE_LABELS[m]}(V{hw_valve}): {last_err}")
            self.pressure_vars[m].set("0.00")
            self.pressure_applied[m] = 0.0
            self._update_valve_display(m, 0.0)
        with self._lock:
            self.state.pressures_commanded_bar = [0.0] * 6
        try:
            with self._bus_lock:
                clear_all(self.bus_adc)
        except Exception:
            pass

        if errors:
            self.status_var.set(f"⚠️  Urgence partielle : {', '.join(errors)}")
        else:
            self.status_var.set(
                "🛑  ARRÊT D'URGENCE armé — appuyer ACQUITTER pour réarmer")

    def _acknowledge_estop(self):
        with self.state.lock:
            if not self.state.estop_requested:
                self.status_var.set("Aucun arrêt d'urgence actif")
                return
        self.state.clear_estop()
        self.status_var.set("✅  Arrêt d'urgence acquitté — système réarmé")

    def _on_safety_event(self, event: SafetyEvent):
        """Appelé par le watchdog (thread watchdog) → marshallé vers main thread."""
        msg = f"⚠️  {event.message}"
        try:
            self.root.after(0, lambda m=msg: self.status_var.set(m))
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Logging CSV
    # ─────────────────────────────────────────────────────────────────────────
    @property
    def logging_active(self) -> bool:
        return self._logger.is_active

    def _toggle_logging(self):
        if self.logging_active:
            self._stop_logging()
        else:
            self._start_logging()

    def _start_logging(self):
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"robot_log_{timestamp}.csv"
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_name,
            initialdir=os.path.expanduser("~"),
            title="Choisir le fichier de log",
        )
        if not path:
            return
        self._start_logging_at(path)

    def _start_logging_at(self, path: str) -> bool:
        try:
            mass = float(self.mass_var.get())
        except ValueError:
            mass = 0.0
        com = self._get_com_offset_m()
        with self.state.lock:
            pid_kp  = self.state.config.pid_kp
            pid_ki  = self.state.config.pid_ki
            pid_kd  = self.state.config.pid_kd
            pid_tol = self.state.config.pid_settle_tol_bar
            fb_dom  = getattr(self.state.config, 'feedback_domain', 'pressure')
            pf_kp   = getattr(self.state.config, 'pid_force_kp', 0.0)
            pf_ki   = getattr(self.state.config, 'pid_force_ki', 0.0)
            pf_kd   = getattr(self.state.config, 'pid_force_kd', 0.0)
        try:
            self._logger.start(
                path,
                mass_kg=mass,
                com_offset_m=com,
                l0_m=MUSCLE_NOMINAL_LENGTH_M,
                rest_lengths_m=self.rest_lengths,
                loop_hz=TARGET_HZ,
                pid_kp=pid_kp,
                pid_ki=pid_ki,
                pid_kd=pid_kd,
                pid_settle_tol_bar=pid_tol,
                feedback_domain=fb_dom,
                pid_force_kp=pf_kp,
                pid_force_ki=pf_ki,
                pid_force_kd=pf_kd,
            )
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de créer le fichier :\n{e}")
            return False
        # Synchronise le flag logging vers state pour que ControlLoop écrive
        with self.state.lock:
            self.state.logging_active = True
        self.log_btn.config(text="⏹  Arrêter log", fg=RED)
        self.status_var.set(f"📝  Logging → {os.path.basename(path)}")
        return True

    def _stop_logging(self):
        path = self._logger.path
        self._logger.stop()
        with self.state.lock:
            self.state.logging_active = False
        self.log_btn.config(text="📝  Démarrer log", fg=PURPLE)
        self.status_var.set(f"Log sauvegardé → {os.path.basename(path)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Séquence automatique
    # ─────────────────────────────────────────────────────────────────────────
    def _toggle_auto_sequence(self):
        if self.ctrl.auto_seq_running:
            self._abort_auto_sequence()
        else:
            self._start_auto_sequence()

    def _start_auto_sequence(self):
        with self.state.lock:
            calib_done = self.state.calibration_done
        if not calib_done:
            messagebox.showwarning("Calibration requise",
                                   "Lancez la calibration avant la séquence automatique.")
            return
        if self.logging_active:
            messagebox.showwarning("Log déjà actif",
                                   "Arrêtez le log manuel avant de lancer la séquence.")
            return
        n   = len(AUTO_SEQUENCE)
        dur = auto_seq_total_duration_s()
        ff_msg = ("Le feedforward est activé."
                  if self.ff_enabled.get()
                  else "⚠ Le feedforward sera ACTIVÉ automatiquement.")
        pid_msg = ("Le feedback PID est ACTIF (FF + PID)."
                   if self.pid_enabled.get()
                   else "Le feedback PID est inactif (feedforward pur).\n"
                        "    → Activez le PID avant de lancer pour un mode FF+PID.")
        ok = messagebox.askyesno(
            "Séquence automatique",
            f"Lancer la séquence de caractérisation ?\n\n"
            f"  • {n} points + retours HOME\n"
            f"  • Durée estimée : {dur:.0f} s ({dur/60:.1f} min)\n"
            f"  • Log CSV automatique (FF + PID loggés)\n"
            f"  • {ff_msg}\n"
            f"  • {pid_msg}\n\n"
            f"Appuyez à nouveau sur le bouton pour interrompre.")
        if not ok:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag  = "ff_pid" if self.pid_enabled.get() else "ff_only"
        log_path  = os.path.join(os.path.expanduser("~"),
                                 f"robot_log_seq_{mode_tag}_{timestamp}.csv")
        if not self._start_logging_at(log_path):
            return
        if not self.ff_enabled.get():
            self.ff_enabled.set(True)
            self._toggle_feedforward()
        # PID conservé tel quel pendant la séquence.
        self.auto_seq_btn.config(text="⏹  Arrêter séquence", fg=RED)
        self.ctrl.start_auto_sequence()

    def _abort_auto_sequence(self):
        self.status_var.set("⏸  Interruption séquence en cours…")
        self.ctrl.abort_auto_sequence()

    def _on_pose_request(self, tx, ty, tz, rx_deg, ry_deg, rz_deg):
        """Callback appelé depuis le worker AutoSequence (autre thread).
        Marshalle sur le main thread via root.after."""
        def _apply():
            try:
                self.target_vars["x"].set(f"{tx:.3f}")
                self.target_vars["y"].set(f"{ty:.3f}")
                self.target_vars["z"].set(f"{tz:.3f}")
                self.target_vars["alpha"].set(f"{rx_deg:.2f}")
                self.target_vars["beta"].set(f"{ry_deg:.2f}")
                self.target_vars["gamma"].set(f"{rz_deg:.2f}")
                self._compute_inverse_kinematics()
            except Exception as e:
                print(f"[ERROR] _on_pose_request : {e}")
        self.root.after(0, _apply)

    def _on_auto_sequence_finished(self):
        """Appelée sur le main thread (via callback ControlLoop) à la fin."""
        if self.logging_active:
            try:
                self._stop_logging()
            except Exception as e:
                print(f"[ERROR] stop_logging in seq finally : {e}")
        self.auto_seq_btn.config(text="▶  Démarrer séquence auto", fg=GREEN)
        self.status_var.set("✔  Séquence terminée — log sauvegardé")

    # ─────────────────────────────────────────────────────────────────────────
    # Fermeture propre
    # ─────────────────────────────────────────────────────────────────────────
    def _on_close(self):
        self._running = False
        if hasattr(self, 'watchdog'):
            try:
                self.watchdog.stop()
                self.watchdog.join(timeout=1.0)
            except Exception:
                pass
        self.ctrl.abort_auto_sequence()
        self.ctrl.stop()
        if self.logging_active:
            self._stop_logging()
        for m in range(6):
            try:
                with self._bus_lock:
                    set_muscle_pressure(self.bus_adc, VALVE_OF_MUSCLE[m], 0.0)
            except Exception:
                pass
        self.hw.close()
        self.root.destroy()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = RobotGUI(root)
    root.mainloop()
