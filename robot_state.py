#!/usr/bin/env python3
"""
robot_state.py — État partagé thread-safe entre GUI, control loop et watchdog.
=============================================================================
Toutes les données partagées sont regroupées dans `RobotState` ; accès
explicite via `state.lock`. Le module est pur Python (testable sans Tk ni I2C).

Conventions : longueurs en m, pressions en bar, angles en rad, temps en s
(perf_counter).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from platform_pose import Vec3, IMUSnap


# ═════════════════════════════════════════════════════════════════════════════
# Évènements de sécurité (utilisés par le watchdog)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class SafetyEvent:
    """Évènement détecté par le watchdog. Conservé dans un buffer
      pour diagnostic et affichage. """
    timestamp: float        # perf_counter au moment de la détection
    kind: str               # 'estop' | 'loop_blocked' | 'over_contraction' |
                            # 'over_pressure_cmd' | 'over_pressure_meas'
    muscle: Optional[int]   # 0..5 si muscle-spécifique, None sinon
    value: float            # valeur incriminée
    threshold: float        # seuil dépassé
    message: str            # description courte


# ═════════════════════════════════════════════════════════════════════════════
# ControlConfig — paramètres de contrôle (écrits par la GUI, lus par ControlLoop)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class ControlConfig:
    """Paramètres modifiables depuis la GUI. Lus par la
    control loop sous state.lock (jamais de tk.*Var.get() en thread fond)."""
    ff_enabled:  bool = False
    pid_enabled: bool = False

    # Paramètres feedforward. Les coefficients Sarosi sont redimensionnés
    # pour la masse mécanique réelle (29.14909 kg).
    mass_kg:      float = 29.14909
    min_length_m: float = 0.76                  # seuil sécurité contraction
    com_offset_m: tuple = (0.0, 0.0, 0.0)      # CoM dans le repère plateforme

    # Mask muscle (False → vanne forcée à 0).
    muscle_active: list = field(default_factory=lambda: [True] * 6)

    # Gains PID pression (bar/m, bar/(m·s), bar·s/m).
    pid_kp: float = 2.0
    pid_ki: float = 2.0
    pid_kd: float = 0.2
    pid_settle_tol_bar: float = 0.10   # hérité (gating retiré)

    # Domaine de la boucle de retour :
    #   'pressure' : PID → correction bar ajoutée au FF (gain de boucle variable).
    #   'force'    : PID → ΔF (N), F_total → Sarosi⁻¹ (gain linéarisé,
    #                Gattringer 2009).
    feedback_domain: str = 'pressure'

    # Gains PID force (N/m, N/(m·s), N·s/m).
    pid_force_kp: float = 700.0
    pid_force_ki: float = 500.0
    pid_force_kd: float = 30.0

    # Filtrage IIR sur les longueurs mesurées. fc=6 Hz aligné sur le filtre
    # D du PID (cascade homogène).
    position_filter_enabled: bool  = True
    position_filter_fc_hz:   float = 6.0   # 0 = IIR off


# ═════════════════════════════════════════════════════════════════════════════
# RobotState — état partagé thread-safe
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class RobotState:
    """État complet partagé entre threads. Toutes les lectures/écritures
    sous `state.lock"""

    config: ControlConfig = field(default_factory=ControlConfig)
    logging_active: bool = False

    # RLock pour autoriser les appels imbriqués (coût négligeable à 10 Hz).
    lock: threading.RLock = field(default_factory=threading.RLock)

    # ── Capteurs (valeurs corrigées par les offsets de calibration) ───────
    imu: IMUSnap = field(default_factory=lambda: IMUSnap(Vec3(), Vec3()))
    positions_m: list = field(default_factory=lambda: [0.0] * 6)
    pressures_measured_bar: list = field(default_factory=lambda: [0.0] * 6)

    # ── Calibration ────────────────────────────────────────────────────────
    imu_offset: list = field(default_factory=lambda: [0.0] * 6)   # ax,ay,az,gx,gy,gz
    pos_offset: list = field(default_factory=lambda: [0.0] * 6)
    is_calibrating: bool = False
    calibration_done: bool = False

    # ── Cible utilisateur (entrée) ─────────────────────────────────────────
    target_translation_m: tuple = (0.0, 0.0, 0.0)
    target_rotation_rad: tuple = (0.0, 0.0, 0.0)
    target_lengths_m: list = field(default_factory=lambda: [1.0] * 6)

    # ── Pose estimée par cinématique directe ──────────────────────────────
    fk_translation_m: tuple = (0.0, 0.0, 0.0)
    fk_rotation_rad: tuple = (0.0, 0.0, 0.0)
    fk_info: dict = field(default_factory=lambda: {
        'converged': False, 'iterations': 0,
        'residual_norm': 0.0, 'max_residual': 0.0,
        'valid': False, 'reason': 'not calibrated',
    })

    # ── Erreur de pose (cible − mesurée), pour MONITORING uniquement ──────
    pose_error_t_m: tuple = (0.0, 0.0, 0.0)
    pose_error_r_rad: tuple = (0.0, 0.0, 0.0)

    # ── Commandes (sortie du contrôleur, envoyées aux vannes) ─────────────
    pressures_commanded_bar: list = field(default_factory=lambda: [0.0] * 6)

    # ── Sortie intermédiaire du feedforward (pour affichage/log) ──────────
    ff_pressures_bar: list = field(default_factory=lambda: [0.0] * 6)
    ff_tensions_N: list = field(default_factory=lambda: [0.0] * 6)
    ff_kappa: list = field(default_factory=lambda: [0.0] * 6)
    ff_saturated: list = field(default_factory=lambda: [False] * 6)
    ff_neg_tension: list = field(default_factory=lambda: [False] * 6)

    # PID longueur — domaine pression (s'ajoute au FF avant envoi vanne).
    pid_enabled: bool = False
    pid_outputs_bar: list = field(default_factory=lambda: [0.0] * 6)
    pid_integral_m_s: list = field(default_factory=lambda: [0.0] * 6)
    pid_gate_open: list = field(default_factory=lambda: [False] * 6)
    pid_length_error_m: list = field(default_factory=lambda: [0.0] * 6)

    # PID longueur — domaine force (renseignés seulement si feedback_domain='force').
    pid_force_outputs_N: list = field(default_factory=lambda: [0.0] * 6)
    cmd_total_force_N:   list = field(default_factory=lambda: [0.0] * 6)

    # Mode vannes manuelles : sans contrôleur actif, la boucle remet
    # normalement les vannes à 0 à chaque tick. Si manual_valve_mode est
    # True, elle applique manual_pressure_setpoints_bar (mêmes clamps de
    # sécurité). Armé par GUI._apply_pressure, désarmé par
    # zero_all_valves / estop / désactivation FF.
    manual_valve_mode: bool = False
    manual_pressure_setpoints_bar: list = field(default_factory=lambda: [0.0] * 6)

    # Sécurité.
    estop_requested: bool = False
    heartbeat_main_loop: float = 0.0   # mis à jour par la control loop
    heartbeat_gui: float = 0.0         # mis à jour par le refresh Tk

    # Ring-buffer borné des évènements de sécurité (diagnostic + affichage).
    safety_events: list = field(default_factory=list)
    _max_safety_events: int = 100

    def record_safety_event(self, event: SafetyEvent) -> None:
        """Pousse dans le ring-buffer (À APPELER SOUS LOCK)."""
        self.safety_events.append(event)
        if len(self.safety_events) > self._max_safety_events:
            self.safety_events.pop(0)

    def snapshot(self) -> dict:
        """Snapshot atomique des champs utilisés par le watchdog."""
        with self.lock:
            return {
                'positions_m'           : self.positions_m[:],
                'pressures_measured_bar': self.pressures_measured_bar[:],
                'pressures_commanded_bar': self.pressures_commanded_bar[:],
                'estop_requested'       : self.estop_requested,
                'heartbeat_main_loop'   : self.heartbeat_main_loop,
                'heartbeat_gui'         : self.heartbeat_gui,
                'calibration_done'      : self.calibration_done,
                'is_calibrating'        : self.is_calibrating,
            }

    def request_estop(self, source: str = 'unknown') -> None:
        """Arme l'estop. source loggé pour diagnostic."""
        with self.lock:
            if not self.estop_requested:
                self.estop_requested = True
                self.record_safety_event(SafetyEvent(
                    timestamp=time.perf_counter(),
                    kind='estop',
                    muscle=None,
                    value=0.0,
                    threshold=0.0,
                    message=f'estop armé depuis : {source}',
                ))

    def clear_estop(self) -> None:
        with self.lock:
            self.estop_requested = False

