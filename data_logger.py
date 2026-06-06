#!/usr/bin/env python3
"""
data_logger.py — Logging CSV non-bloquant
==========================================
L'écriture est délocalisée dans un thread dédié alimenté par une queue,
pour ne pas bloquer la control loop lors des stalls SD.
`write_row` retourne en quelques µs ; si la queue sature, la ligne est
droppée (`dropped_count` exposé).

Format : bloc métadonnées en tête (lignes `# ...`, sauté par pandas via
`comment='#'`), en-tête CSV, puis lignes de données.

API : start, stop, write_row, is_active, path.
"""
from __future__ import annotations

import csv
import math
import queue
import threading
import time
from datetime import datetime
from typing import Optional, Sequence

from geometry import MUSCLE_LABELS, VALVE_OF_MUSCLE, ADC_CH_OF_MUSCLE


_STOP = object()                # sentinelle d'arrêt du thread
_QUEUE_MAXSIZE = 200            # ~20 s de buffer à 10 Hz
_FLUSH_INTERVAL_S = 1.0         # flush périodique (limite la perte sur crash)


class LogWriter:
    """Wrapper CSV non-bloquant (écriture déléguée à un thread dédié)."""

    def __init__(self):
        self._file = None
        self._writer = None
        self._path: str = ""
        self._t0: float = 0.0

        # Threading / queue
        self._queue: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt: Optional[threading.Event] = None

        # Stats / diagnostic
        self._dropped_count: int = 0

    # ─── Lifecycle ──────────────────────────────────────────────────────────
    @property
    def is_active(self) -> bool:
        return self._writer is not None

    @property
    def path(self) -> str:
        return self._path

    def start(
        self,
        path: str,
        *,
        mass_kg: float,
        com_offset_m: Sequence[float],
        l0_m: float,
        rest_lengths_m: Sequence[float],
        loop_hz: float,
        pid_kp: float = 0.0,
        pid_ki: float = 0.0,
        pid_kd: float = 0.0,
        pid_settle_tol_bar: float = 0.0,
        feedback_domain: str = 'pressure',
        pid_force_kp: float = 0.0,
        pid_force_ki: float = 0.0,
        pid_force_kd: float = 0.0,
    ) -> None:
        """Ouvre le fichier, écrit métadonnées + en-tête, démarre le thread."""
        if self.is_active:
            raise RuntimeError("LogWriter already active — call stop() first")

        self._file = open(path, "w", newline="")

        # Métadonnées (lignes commentées, écrites une fois).
        meta = self._build_metadata(
            mass_kg=mass_kg,
            com_offset_m=com_offset_m,
            l0_m=l0_m,
            rest_lengths_m=rest_lengths_m,
            loop_hz=loop_hz,
            pid_kp=pid_kp,
            pid_ki=pid_ki,
            pid_kd=pid_kd,
            pid_settle_tol_bar=pid_settle_tol_bar,
            feedback_domain=feedback_domain,
            pid_force_kp=pid_force_kp,
            pid_force_ki=pid_force_ki,
            pid_force_kd=pid_force_kd,
        )
        self._file.write(meta)
        self._file.flush()

        self._writer = csv.writer(self._file)
        self._writer.writerow(self._build_header())
        self._path = path
        self._t0 = time.perf_counter()
        self._dropped_count = 0

        self._queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(
            target=self._writer_loop, name="LogWriter", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Vide la queue puis ferme proprement."""
        if not self.is_active and self._thread is None:
            return

        if self._stop_evt is not None:
            self._stop_evt.set()

        if self._queue is not None:
            try:
                self._queue.put(_STOP, timeout=2.0)
            except queue.Full:
                pass  # thread bloqué sur SD ; stop_evt suffira

        if self._thread is not None:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                print("[WARN] LogWriter thread did not terminate cleanly")
            self._thread = None

        self._queue = None
        self._stop_evt = None

        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
        self._file = None
        self._writer = None

        if self._dropped_count > 0:
            print(f"[INFO] LogWriter: {self._dropped_count} ligne(s) "
                  f"droppée(s) (queue saturée — SD trop lente)")

    # ─── Écriture (non bloquante) ──────────────────────────────────────────
    def write_row(
        self,
        # — Pose —
        target_translation_m: Sequence[float],   # (tx, ty, tz)         m
        target_rotation_rad: Sequence[float],    # (rx, ry, rz)         rad
        fk_translation_m: Sequence[float],       # FK depuis longueurs  m
        fk_rotation_rad: Sequence[float],        # FK depuis longueurs  rad
        fk_valid: bool,
        # — Longueurs muscles —
        target_lengths_m: Sequence[float],       # 6 longueurs IK       m
        measured_lengths_m: Sequence[float],     # 6 longueurs capteur  m
        # — Pressions —
        target_pressures_bar: Sequence[float],   # 6 cmd vanne (post safety) bar
        measured_pressures_bar: Sequence[float], # 6 ADC pression       bar
        # — Forces axiales cibles —
        target_tensions_N: Sequence[float],      # 6 tensions requises  N
        # — Auxiliaires —
        ff_pressures_raw_bar: Sequence[float],   # FF avant safety      bar
        ff_kappa: Sequence[float],               # contraction utilisée
        ff_saturated: Sequence[bool],            # cmd vanne saturée [0,6]
        ff_neg_tension: Sequence[bool],          # solver demande compression
        ff_enabled: bool,
        calibrating: bool,
        imu_accel: Sequence[float],              # (ax, ay, az)         m/s²
        imu_gyro: Sequence[float],               # (gx, gy, gz)         dps
        # — PID longueur (optionnels, zéro si PID off) —
        pid_enabled: bool = False,
        pid_length_error_m: Sequence[float] = (0.0,)*6,  # L_meas − L_cible m
        pid_p_terms_bar: Sequence[float] = (0.0,)*6,     # contribution Kp·err
        pid_i_terms_bar: Sequence[float] = (0.0,)*6,     # contribution Ki·∫err
        pid_d_terms_bar: Sequence[float] = (0.0,)*6,     # contribution Kd·d(err)
        pid_output_bar: Sequence[float] = (0.0,)*6,      # sortie PID totale
        # — Domaine de retour + PID force  ────────────────────
        feedback_domain: str = 'pressure',               # 'pressure' | 'force'
        pid_force_output_N: Sequence[float] = (0.0,)*6,  # correction ΔF (N)
        cmd_total_force_N: Sequence[float] = (0.0,)*6,   # T_ff + ΔF (N)
    ) -> None:
        """Format + push dans la queue. Non bloquant : ligne droppée si
        queue saturée (`dropped_count` incrémenté)."""
        if self._writer is None or self._queue is None:
            return
        try:
            t = time.perf_counter() - self._t0

            row = [
                f"{t:.4f}",
                # Pose cible
                f"{target_translation_m[0]:+.5f}",
                f"{target_translation_m[1]:+.5f}",
                f"{target_translation_m[2]:+.5f}",
                f"{math.degrees(target_rotation_rad[0]):+.4f}",
                f"{math.degrees(target_rotation_rad[1]):+.4f}",
                f"{math.degrees(target_rotation_rad[2]):+.4f}",
                # Pose mesurée (FK)
                f"{fk_translation_m[0]:+.5f}",
                f"{fk_translation_m[1]:+.5f}",
                f"{fk_translation_m[2]:+.5f}",
                f"{math.degrees(fk_rotation_rad[0]):+.4f}",
                f"{math.degrees(fk_rotation_rad[1]):+.4f}",
                f"{math.degrees(fk_rotation_rad[2]):+.4f}",
                "1" if fk_valid else "0",
            ]
            # Longueurs : cibles puis mesurées (M0..M5)
            row.extend(f"{v:.5f}" for v in target_lengths_m)
            row.extend(f"{v:.5f}" for v in measured_lengths_m)
            # Pressions : cibles puis mesurées (M0..M5)
            row.extend(f"{v:.3f}" for v in target_pressures_bar)
            row.extend(f"{v:.3f}" for v in measured_pressures_bar)
            # Forces axiales cibles (M0..M5)
            row.extend(f"{v:+.3f}" for v in target_tensions_N)
            # Auxiliaires
            row.extend(f"{v:.3f}" for v in ff_pressures_raw_bar)
            row.extend(f"{v:.5f}" for v in ff_kappa)
            row.extend(str(int(bool(s))) for s in ff_saturated)
            row.extend(str(int(bool(s))) for s in ff_neg_tension)
            row.append("1" if ff_enabled else "0")
            row.append("1" if calibrating else "0")
            row.extend(f"{v:.5f}" for v in imu_accel)
            row.extend(f"{v:.5f}" for v in imu_gyro)
            # PID longueur
            row.append("1" if pid_enabled else "0")
            row.extend(f"{v*1000:+.3f}" for v in pid_length_error_m)  # → mm
            row.extend(f"{v:+.4f}" for v in pid_p_terms_bar)
            row.extend(f"{v:+.4f}" for v in pid_i_terms_bar)
            row.extend(f"{v:+.4f}" for v in pid_d_terms_bar)
            row.extend(f"{v:+.4f}" for v in pid_output_bar)
            # Domaine de retour + PID force (Gattringer)
            row.append("force" if feedback_domain == 'force' else "pressure")
            row.extend(f"{v:+.3f}" for v in pid_force_output_N)
            row.extend(f"{v:+.3f}" for v in cmd_total_force_N)

            try:
                self._queue.put_nowait(row)
            except queue.Full:
                self._dropped_count += 1
        except Exception as e:
            # JAMAIS propager : la control loop ne doit pas crasher pour le log.
            print(f"[ERROR] LogWriter.write_row : {e}")

    def _writer_loop(self) -> None:
        """Thread dédié : drain queue, écrit, flush. Absorbe les stalls SD."""
        last_flush = time.perf_counter()

        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stop_evt is not None and self._stop_evt.is_set():
                    break
                if (self._file is not None
                        and time.perf_counter() - last_flush > _FLUSH_INTERVAL_S):
                    try:
                        self._file.flush()
                    except Exception:
                        pass
                    last_flush = time.perf_counter()
                continue

            if item is _STOP:
                break

            try:
                self._writer.writerow(item)
            except Exception as e:
                print(f"[ERROR] LogWriter writer thread : {e}")

            if time.perf_counter() - last_flush > _FLUSH_INTERVAL_S:
                try:
                    self._file.flush()
                except Exception:
                    pass
                last_flush = time.perf_counter()

        # Vidange finale : draine ce qui reste.
        if self._queue is not None:
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is _STOP:
                    continue
                try:
                    self._writer.writerow(item)
                except Exception:
                    pass

        # Flush final avant close().
        if self._file is not None:
            try:
                self._file.flush()
            except Exception:
                pass

    # ─── Métadonnées (écrites une fois au début, inchangé) ─────────────────
    @staticmethod
    def _build_metadata(
        *,
        mass_kg: float,
        com_offset_m: Sequence[float],
        l0_m: float,
        rest_lengths_m: Sequence[float],
        loop_hz: float,
        pid_kp: float = 0.0,
        pid_ki: float = 0.0,
        pid_kd: float = 0.0,
        pid_settle_tol_bar: float = 0.0,
        feedback_domain: str = 'pressure',
        pid_force_kp: float = 0.0,
        pid_force_ki: float = 0.0,
        pid_force_kd: float = 0.0,
    ) -> str:
        """Bloc de lignes commentées résumant les paramètres de session."""
        lines = []
        lines.append("# Stewart Platform Log Control")
        lines.append(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"# Mass_kg: {mass_kg:.5f}")
        lines.append(
            f"# COM_offset_m: x={com_offset_m[0]:+.5f} "
            f"y={com_offset_m[1]:+.5f} z={com_offset_m[2]:+.5f}"
        )
        lines.append(f"# Muscle_nominal_length_L0_m: {l0_m:.5f}")
        rest_str = ", ".join(
            f"{MUSCLE_LABELS[i]}={rest_lengths_m[i]:.5f}"
            for i in range(6)
        )
        lines.append(f"# Rest_geometric_lengths_m: {rest_str}")
        lines.append(f"# Loop_frequency_Hz: {loop_hz:.2f}")
        mapping = ", ".join(
            f"M{i}={MUSCLE_LABELS[i]}(ADC{ADC_CH_OF_MUSCLE[i]},V{VALVE_OF_MUSCLE[i]})"
            for i in range(6)
        )
        lines.append(f"# Muscle_mapping: {mapping}")
        lines.append(
            f"# PID_length_gains: Kp={pid_kp:.6f} bar/m  "
            f"Ki={pid_ki:.6f} bar/(m·s)  "
            f"Kd={pid_kd:.6f} bar·s/m  "
            f"settle_tol={pid_settle_tol_bar:.4f} bar"
        )
        lines.append(
            f"# Feedback_domain: {feedback_domain}  "
            f"(pressure=PID→bar | force=PID→ΔF→Sarosi⁻¹, approche Gattringer)"
        )
        lines.append(
            f"# PID_force_gains: Kp={pid_force_kp:.4f} N/m  "
            f"Ki={pid_force_ki:.4f} N/(m·s)  "
            f"Kd={pid_force_kd:.4f} N·s/m"
        )
        lines.append("# Architecture: FF + PID_length → valve (boucle unique)")
        lines.append(
            "# Conventions: angles=deg, longueurs/positions=m, "
            "pressions=bar, forces=N (positif=traction)"
        )
        return "\n".join(lines) + "\n"

    # ─── En-tête CSV  ────────────────────────────────────────────
    @staticmethod
    def _build_header() -> list:
        # 1. Pose cible
        pose_target = [
            "target_tx_m", "target_ty_m", "target_tz_m",
            "target_rx_deg", "target_ry_deg", "target_rz_deg",
        ]
        # 2. Pose mesurée (FK)
        pose_meas = [
            "fk_tx_m", "fk_ty_m", "fk_tz_m",
            "fk_rx_deg", "fk_ry_deg", "fk_rz_deg",
            "fk_valid",
        ]
        # 3. Longueurs muscles
        target_L = [f"target_L_M{m}_{MUSCLE_LABELS[m]}_m" for m in range(6)]
        meas_L   = [f"meas_L_M{m}_{MUSCLE_LABELS[m]}_m"   for m in range(6)]
        # 4. Pressions
        target_P = [f"target_P_M{m}_{MUSCLE_LABELS[m]}_bar" for m in range(6)]
        meas_P   = [f"meas_P_M{m}_{MUSCLE_LABELS[m]}_bar"   for m in range(6)]
        # 5. Forces axiales cibles
        target_F = [f"target_F_M{m}_{MUSCLE_LABELS[m]}_N" for m in range(6)]
        # 6. Auxiliaires
        ff_raw  = [f"ff_raw_P_M{m}_{MUSCLE_LABELS[m]}_bar" for m in range(6)]
        ff_kap  = [f"ff_kappa_M{m}_{MUSCLE_LABELS[m]}"     for m in range(6)]
        ff_sat  = [f"ff_saturated_M{m}_{MUSCLE_LABELS[m]}" for m in range(6)]
        ff_neg  = [f"ff_neg_tension_M{m}_{MUSCLE_LABELS[m]}" for m in range(6)]
        flags   = ["ff_enabled", "calibrating"]
        imu     = [
            "accel_x_ms2", "accel_y_ms2", "accel_z_ms2",
            "gyro_x_dps",  "gyro_y_dps",  "gyro_z_dps",
        ]
        return [
            "timestamp_s",
            *pose_target, *pose_meas,
            *target_L, *meas_L,
            *target_P, *meas_P,
            *target_F,
            *ff_raw, *ff_kap, *ff_sat, *ff_neg,
            *flags,
            *imu,
            # PID longueur (outer)
            "pid_enabled",
            *[f"pid_err_M{m}_{MUSCLE_LABELS[m]}_mm"  for m in range(6)],
            *[f"pid_P_M{m}_{MUSCLE_LABELS[m]}_bar"   for m in range(6)],
            *[f"pid_I_M{m}_{MUSCLE_LABELS[m]}_bar"   for m in range(6)],
            *[f"pid_D_M{m}_{MUSCLE_LABELS[m]}_bar"   for m in range(6)],
            *[f"pid_out_M{m}_{MUSCLE_LABELS[m]}_bar" for m in range(6)],
            # Domaine de retour + PID force (Gattringer)
            "feedback_domain",
            *[f"pid_dF_M{m}_{MUSCLE_LABELS[m]}_N"   for m in range(6)],
            *[f"cmd_F_M{m}_{MUSCLE_LABELS[m]}_N"    for m in range(6)],
        ]
