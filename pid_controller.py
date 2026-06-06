#!/usr/bin/env python3
"""
pid_controller.py — PID longueur (par muscle) pour la plateforme Stewart
========================================================================
Deux contrôleurs :
  - MuscleLengthPID       : sortie = pression bar (ajoutée au FF)
  - MuscleLengthForcePID  : sortie = force ΔF (N), force totale inversée
                            par Sarosi (approche Gattringer 2009)

Triplet (Kp, Ki, Kd) commun aux 6 muscles, anti-windup conditionnel,
dérivée filtrée sur la mesure (passe-bas 1er ordre, fc ≈ 6 Hz par défaut)
pour amortir le bruit fil-pot et éviter le derivative kick.

Convention de signe : error = L_measured − L_target.
  - muscle trop long (e>0)  → output > 0  → +pression/force → contraction
  - muscle trop court (e<0) → output < 0  → relâchement

Cadence cible : 13 Hz (bornée par l'ADC position). Pas thread-safe.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration et gains
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PIDGains:
    """Triplet de gains commun aux 6 muscles."""
    kp: float = 0.2   # bar / m
    ki: float = 0.2   # bar / (m·s)
    kd: float = 0.2   # bar·s / m


@dataclass
class PIDConfig:
    """Bornes du PID pression. Surchargeables depuis la GUI."""
    # Saturation sortie PID (bar). Borne la correction additive au FF.
    output_min_bar: float = -2.0
    output_max_bar: float = +2.0

    # Clamp intégral (bar). Limite la dérive vers le windup.
    integral_clamp_bar: float = 0.5

    # Champs hérités (gating supprimé), conservés pour data_logger / GUI.
    settle_tol_bar: float = 0.15
    settle_min_ticks: int = 3

    # Plafond dt (s). Évite un saut au premier tick après reset/freeze.
    dt_max_s: float = 0.1

    # Filtre passe-bas 1er ordre sur le terme D (Hz). fc=6 Hz forme une
    # cascade homogène avec l'IIR position.
    # 0 = filtre désactivé.
    d_filter_cutoff_hz: float = 6.0

    # Pression max vanne (bar) — anti-windup conditionnel sur p_ff+p_pid.
    p_total_max_bar: float = 4.0
    p_total_min_bar: float = 0.0

    # Deadband erreur de longueur (m). Sous ce seuil, intégrateur gelé +
    # sortie latchée → consigne pression figée (cf. justif. ForcePIDConfig).
    deadband_m: float = 0.007   # 7 mm


# ─────────────────────────────────────────────────────────────────────────────
#  Contrôleur
# ─────────────────────────────────────────────────────────────────────────────
class MuscleLengthPID:
    """PID pression, identique sur les 6 muscles. Stateful, pas thread-safe."""

    N = 6

    def __init__(self, config: PIDConfig | None = None):
        self.config = config or PIDConfig()
        self.gains  = PIDGains()

        # État interne par muscle
        self.integral_m_s        = np.zeros(self.N)
        self.prev_meas_length_m  = np.zeros(self.N)
        self.outputs_bar         = np.zeros(self.N)
        self.d_filtered          = np.zeros(self.N)

        # État hérité (gating supprimé), conservé pour data_logger / GUI.
        self.last_p_cmd_bar      = np.zeros(self.N)
        self.settle_count        = np.zeros(self.N, dtype=int)
        self.stabilized          = np.ones(self.N, dtype=bool)

        # Termes P/I/D séparés (lecture seule, pour affichage/log).
        self.p_terms_bar         = np.zeros(self.N)
        self.i_terms_bar         = np.zeros(self.N)
        self.d_terms_bar         = np.zeros(self.N)

        self._has_prev_meas      = False
        self._last_update_time   = None

    def set_gains(self, kp: float, ki: float, kd: float,
                  reset_integral: bool = True) -> None:
        """Met à jour les gains. Par défaut reset_integral=True : un saut
        de Ki sur une intégrale en m·s peut produire un saut violent en
        sortie. Passer False pour conserver explicitement l'état."""
        self.gains = PIDGains(kp=float(kp), ki=float(ki), kd=float(kd))
        if reset_integral:
            self.integral_m_s.fill(0.0)

    def set_d_filter_cutoff(self, fc_hz: float) -> None:
        """fc_hz = 0 → filtre désactivé (D direct, sensible au bruit)."""
        self.config.d_filter_cutoff_hz = max(0.0, float(fc_hz))

    def set_deadband(self, deadband_m: float) -> None:
        """0 = désactivé."""
        self.config.deadband_m = max(0.0, float(deadband_m))

    def reset(self, t_now: float | None = None) -> None:
        """Reset complet (à appeler sur (dés)activation, E-stop, calibration)."""
        self.integral_m_s.fill(0.0)
        self.prev_meas_length_m.fill(0.0)
        self.outputs_bar.fill(0.0)
        self.d_filtered.fill(0.0)
        self.last_p_cmd_bar.fill(0.0)
        self.settle_count.fill(0)
        self.stabilized.fill(True)
        self.p_terms_bar.fill(0.0)
        self.i_terms_bar.fill(0.0)
        self.d_terms_bar.fill(0.0)
        self._has_prev_meas    = False
        self._last_update_time = t_now

    def _update_settling(self, p_measured_bar: np.ndarray) -> None:
        """Hérité (gating retiré). Maintient last_p_cmd_bar/settle_count
        pour data_logger ; n'affecte plus le PID."""
        diff = np.abs(p_measured_bar - self.last_p_cmd_bar)
        within = diff < self.config.settle_tol_bar
        self.settle_count = np.where(
            within,
            np.minimum(self.settle_count + 1, self.config.settle_min_ticks * 4),
            0,
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Step principal
    # ─────────────────────────────────────────────────────────────────────
    def step(
        self,
        target_lengths_m:   Sequence[float],
        measured_lengths_m: Sequence[float],
        p_measured_bar:     Sequence[float],
        ff_pressures_bar:   Sequence[float],
        t_now: float,
        active_mask: Sequence[bool] | None = None,
    ):
        """Un pas de PID pour les 6 muscles.

        ff_pressures_bar sert à l'anti-windup conditionnel (saturation
        de la pression totale ff+pid). active_mask[m]=False → sortie 0
        (l'intégrateur n'est PAS reset, faire reset() explicite si besoin).

        Retourne (outputs_bar, gate_open). gate_open est hérité (toujours
        True), conservé pour la GUI.
        """
        L_target = np.asarray(target_lengths_m,  dtype=float)
        L_meas   = np.asarray(measured_lengths_m, dtype=float)
        p_meas   = np.asarray(p_measured_bar,    dtype=float)
        p_ff     = np.asarray(ff_pressures_bar,  dtype=float)

        # Maintien des compteurs de stabilisation (no-op fonctionnel).
        self._update_settling(p_meas)

        # dt clampé à dt_max_s pour éviter un saut après reset/freeze.
        if self._last_update_time is None:
            dt = 0.0
        else:
            dt = max(0.0, min(self.config.dt_max_s,
                              t_now - self._last_update_time))

        error = L_meas - L_target

        # Coefficient filtre dérivé : α = dt/(RC+dt), RC = 1/(2π·fc).
        fc = self.config.d_filter_cutoff_hz
        if fc > 0.0 and dt > 1e-6:
            rc = 1.0 / (2.0 * np.pi * fc)
            d_alpha = dt / (rc + dt)
        else:
            d_alpha = 1.0

        gate_open = self.stabilized.copy()
        ki = self.gains.ki
        i_clamp = self.config.integral_clamp_bar
        deadband = self.config.deadband_m

        for m in range(self.N):
            if active_mask is not None and not bool(active_mask[m]):
                # Muscle désactivé : sortie 0, intégrateur gelé.
                self.outputs_bar[m] = 0.0
                self.p_terms_bar[m] = 0.0
                self.i_terms_bar[m] = 0.0
                self.d_terms_bar[m] = 0.0
                continue

            e = float(error[m])

            # Deadband : intégrateur gelé + sortie latchée → consigne figée.
            if deadband > 0.0 and abs(e) < deadband:
                self.prev_meas_length_m[m] = float(L_meas[m])
                continue

            p_term = self.gains.kp * e

            # Dérivée sur la mesure (anti derivative-kick), filtrée.
            if self._has_prev_meas and dt > 1e-6:
                dL_dt_raw = (float(L_meas[m]) - self.prev_meas_length_m[m]) / dt
            else:
                dL_dt_raw = 0.0
            self.d_filtered[m] += d_alpha * (dL_dt_raw - self.d_filtered[m])
            d_term = self.gains.kd * self.d_filtered[m]

            # Intégrale provisoire avec clamp.
            integral_provisional = self.integral_m_s[m] + e * dt
            i_term_provisional = ki * integral_provisional
            if i_term_provisional > i_clamp:
                i_term_provisional = i_clamp
                if abs(ki) > 1e-9:
                    integral_provisional = i_clamp / ki
            elif i_term_provisional < -i_clamp:
                i_term_provisional = -i_clamp
                if abs(ki) > 1e-9:
                    integral_provisional = -i_clamp / ki

            out_provisional = p_term + i_term_provisional + d_term
            out_provisional = float(np.clip(
                out_provisional,
                self.config.output_min_bar,
                self.config.output_max_bar,
            ))

            # Anti-windup conditionnel sur la pression totale (ff+pid).
            p_total = float(p_ff[m]) + out_provisional
            saturating_high = (p_total >= self.config.p_total_max_bar - 1e-6
                               and e > 0.0)
            saturating_low  = (p_total <= self.config.p_total_min_bar + 1e-6
                               and e < 0.0)
            if not (saturating_high or saturating_low):
                self.integral_m_s[m] = integral_provisional

            i_term_final = ki * self.integral_m_s[m]
            i_term_final = float(np.clip(i_term_final, -i_clamp, +i_clamp))
            out_final = p_term + i_term_final + d_term
            out_final = float(np.clip(
                out_final,
                self.config.output_min_bar,
                self.config.output_max_bar,
            ))

            self.outputs_bar[m]        = out_final
            self.prev_meas_length_m[m] = float(L_meas[m])
            self.p_terms_bar[m] = p_term
            self.i_terms_bar[m] = i_term_final
            self.d_terms_bar[m] = d_term

        self._last_update_time = t_now
        self._has_prev_meas    = True

        return self.outputs_bar.copy(), gate_open

    def record_applied_pressures(self,
                                 p_total_applied_bar: Sequence[float]) -> None:
        """No-op fonctionnel (le PID n'utilise plus la pression appliquée).
        Conservée pour ne pas casser les appelants."""
        self.last_p_cmd_bar = np.asarray(p_total_applied_bar, dtype=float).copy()


# ─────────────────────────────────────────────────────────────────────────────
#  PID en domaine FORCE (approche Gattringer et al. 2009, figure 4)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ForcePIDConfig:
    """Bornes du PID force. La sortie ΔF (N) s'ajoute à T_ff avant
    inversion Sarosi → pression."""
    output_min_N: float = -400.0
    output_max_N: float = +400.0
    integral_clamp_N: float = 200.0
    dt_max_s: float = 0.1
    # Filtre passe-bas D (Hz). Aligné sur le length PID.
    d_filter_cutoff_hz: float = 6.0
    # Bornes force TOTALE (T_ff + ΔF) pour l'anti-windup. PAM ne pousse
    # pas → ≥ 0. Borne haute ≈ force max DMSP-40 à P_MAX.
    f_total_min_N: float = 0.0
    f_total_max_N: float = 1500.0

    # Deadband (m). Sous ce seuil → intégrateur gelé + sortie latchée.
    deadband_m: float = 0.007   # 7 mm


@dataclass
class ForcePIDGains:
    kp: float = 700.0    # N / m
    ki: float = 500.0    # N / (m·s)
    kd: float = 30.0     # N·s / m


class MuscleLengthForcePID:
    """PID erreur de longueur → correction de FORCE ΔF (N).

        F_total[m] = T_ff[m] + ΔF[m]
        p_valve[m] = Sarosi_inverse(F_total[m], L_cible_muscle[m])

    Place l'inversion PAM dans la boucle (g⁻¹∘g ≈ 1) → gain de boucle
    linéarisé, gains robustes sur toute la plage (Gattringer 2009).
    Structure identique à MuscleLengthPID. Pas thread-safe.
    """

    N = 6

    def __init__(self, config: ForcePIDConfig | None = None):
        self.config = config or ForcePIDConfig()
        self.gains  = ForcePIDGains()

        self.integral_m_s       = np.zeros(self.N)   # ∫ error dt  (m·s)
        self.prev_meas_length_m = np.zeros(self.N)
        self.outputs_N          = np.zeros(self.N)   # dernière correction ΔF
        self.d_filtered         = np.zeros(self.N)   # dL/dt filtré (m/s)

        self.p_terms_N = np.zeros(self.N)
        self.i_terms_N = np.zeros(self.N)
        self.d_terms_N = np.zeros(self.N)

        self._has_prev_meas    = False
        self._last_update_time = None

    # ── Configuration ────────────────────────────────────────────────────
    def set_gains(self, kp: float, ki: float, kd: float,
                  reset_integral: bool = True) -> None:
        self.gains = ForcePIDGains(kp=float(kp), ki=float(ki), kd=float(kd))
        if reset_integral:
            self.integral_m_s.fill(0.0)

    def set_d_filter_cutoff(self, fc_hz: float) -> None:
        self.config.d_filter_cutoff_hz = max(0.0, float(fc_hz))

    def set_deadband(self, deadband_m: float) -> None:
        self.config.deadband_m = max(0.0, float(deadband_m))

    def reset(self, t_now: float | None = None) -> None:
        self.integral_m_s.fill(0.0)
        self.prev_meas_length_m.fill(0.0)
        self.outputs_N.fill(0.0)
        self.d_filtered.fill(0.0)
        self.p_terms_N.fill(0.0)
        self.i_terms_N.fill(0.0)
        self.d_terms_N.fill(0.0)
        self._has_prev_meas    = False
        self._last_update_time = t_now

    def step(
        self,
        target_lengths_m:   Sequence[float],
        measured_lengths_m: Sequence[float],
        ff_tensions_N:      Sequence[float],
        t_now: float,
        active_mask: Sequence[bool] | None = None,
    ):
        """Un pas de PID force. ff_tensions_N pour l'anti-windup sur la
        force totale (T_ff + ΔF). active_mask[m]=False → ΔF=0.
        Retourne outputs_N (6,)."""
        L_target = np.asarray(target_lengths_m,  dtype=float)
        L_meas   = np.asarray(measured_lengths_m, dtype=float)
        T_ff     = np.asarray(ff_tensions_N,      dtype=float)

        if self._last_update_time is None:
            dt = 0.0
        else:
            dt = max(0.0, min(self.config.dt_max_s,
                              t_now - self._last_update_time))

        error = L_meas - L_target

        fc = self.config.d_filter_cutoff_hz
        if fc > 0.0 and dt > 1e-6:
            rc = 1.0 / (2.0 * np.pi * fc)
            d_alpha = dt / (rc + dt)
        else:
            d_alpha = 1.0

        ki = self.gains.ki
        i_clamp = self.config.integral_clamp_N
        deadband = self.config.deadband_m

        for m in range(self.N):
            if active_mask is not None and not bool(active_mask[m]):
                self.outputs_N[m] = 0.0
                self.p_terms_N[m] = 0.0
                self.i_terms_N[m] = 0.0
                self.d_terms_N[m] = 0.0
                continue

            e = float(error[m])

            # Deadband : intégrateur gelé + sortie ΔF latchée.
            # prev_meas mis à jour pour éviter un kick dérivé à la sortie.
            if deadband > 0.0 and abs(e) < deadband:
                self.prev_meas_length_m[m] = float(L_meas[m])
                continue

            p_term = self.gains.kp * e

            if self._has_prev_meas and dt > 1e-6:
                dL_dt_raw = (float(L_meas[m]) - self.prev_meas_length_m[m]) / dt
            else:
                dL_dt_raw = 0.0
            self.d_filtered[m] += d_alpha * (dL_dt_raw - self.d_filtered[m])
            d_term = self.gains.kd * self.d_filtered[m]

            integral_provisional = self.integral_m_s[m] + e * dt
            i_term_provisional = ki * integral_provisional
            if i_term_provisional > i_clamp:
                i_term_provisional = i_clamp
                if abs(ki) > 1e-9:
                    integral_provisional = i_clamp / ki
            elif i_term_provisional < -i_clamp:
                i_term_provisional = -i_clamp
                if abs(ki) > 1e-9:
                    integral_provisional = -i_clamp / ki

            out_provisional = p_term + i_term_provisional + d_term
            out_provisional = float(np.clip(
                out_provisional,
                self.config.output_min_N,
                self.config.output_max_N,
            ))

            # Anti-windup conditionnel sur la force TOTALE (T_ff + ΔF).
            f_total = float(T_ff[m]) + out_provisional
            saturating_high = (f_total >= self.config.f_total_max_N - 1e-6
                               and e > 0.0)
            saturating_low  = (f_total <= self.config.f_total_min_N + 1e-6
                               and e < 0.0)
            if not (saturating_high or saturating_low):
                self.integral_m_s[m] = integral_provisional

            i_term_final = ki * self.integral_m_s[m]
            i_term_final = float(np.clip(i_term_final, -i_clamp, +i_clamp))
            out_final = p_term + i_term_final + d_term
            out_final = float(np.clip(
                out_final,
                self.config.output_min_N,
                self.config.output_max_N,
            ))

            self.outputs_N[m]          = out_final
            self.prev_meas_length_m[m] = float(L_meas[m])
            self.p_terms_N[m] = p_term
            self.i_terms_N[m] = i_term_final
            self.d_terms_N[m] = d_term

        self._last_update_time = t_now
        self._has_prev_meas    = True

        return self.outputs_N.copy()

