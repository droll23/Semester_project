#!/usr/bin/env python3
"""
stewart_feedforward.py — Pipeline feedforward de la plateforme Stewart
======================================================================
Pose cible → IK → 6 longueurs cibles → équilibre statique J^T·T = w →
6 tensions T_i → Sarosi inverse → 6 pressions P_i → vannes.

Modèle Sarosi (exponentiel, coefficients Mehdi) :
    F(p, κ) = (a₁·p + a₂)·exp(a₃·κ) + a₄·κ·p + a₅·p + a₆
    avec p en Pa, κ = (l₀ − l)/l₀ ∈ [0, 0.25], l₀ = 1.0 m

À κ fixé, F est affine en p, donc l'inverse est analytique :
    p = [F − a₂·exp(a₃·κ) − a₆] / [a₁·exp(a₃·κ) + a₄·κ + a₅]

Conventions : longueurs en m, angles en rad (Euler ZYX), pressions en bar
(clampées à [P_MIN, P_MAX]), forces en N (positives = traction).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

# On réutilise IK et FK.
from stewart_forward_kinematics import (
    STEWART_BASE_POINTS_M,
    STEWART_PLATFORM_POINTS_M,
    _euler_rotation_matrix,
    stewart_inverse_kinematics,
    stewart_forward_kinematics,
)


# ═════════════════════════════════════════════════════════════════════════════
# Modèle Sarosi — coefficients identifiés par Mehdi sur banc DMSP-40
# ═════════════════════════════════════════════════════════════════════════════
# Les coefficients originaux Mehdi (a1=0.0002, a2=369.27, a3=-37.51, a4=-0.0304,
# a5=0.0093, a6=-546.02) reproduisent le comportement plateforme à une masse
# effective de 47 kg. La masse mécanique réelle est 29.14909 kg.
#
# L'inverse Sarosi est affine en force à κ fixé : P = (F − β(κ)) / α(κ). Pour
# obtenir P(F_29) = P_Mehdi(F_47) avec F_47 = r·F_29 (r = 47/29.14909 ≈ 1.612),
# on divise a1, a2, a4, a5, a6 par r (a3 inchangé car argument exponentielle).
SAROSI_A1 =  1.2404e-4    # Pa⁻¹·N
SAROSI_A2 =  229.0208     # N
SAROSI_A3 =  -37.5125     # —
SAROSI_A4 =  -1.8854e-2   # Pa⁻¹·N
SAROSI_A5 =   5.7678e-3   # Pa⁻¹·N
SAROSI_A6 = -338.6382     # N

# Longueur nominale du muscle utilisée par Mehdi pour définir κ.
# Sur la plateforme réelle, le potentiomètre à fil renvoie ~1.0 m au repos
# (calibration statique), donc on conserve cette convention.
MUSCLE_NOMINAL_LENGTH_M = 1.0

# Plage de validité du modèle : pression ∈ [0, 6] bar, contraction ∈ [0, 25 %].
P_MIN_BAR = 0.0
P_MAX_BAR = 6.0
KAPPA_MAX = 0.25

# Conversion bar ↔ Pa
BAR_TO_PA = 1.0e5
PA_TO_BAR = 1.0e-5

# Constantes physiques
GRAVITY_M_S2 = 9.81


# ═════════════════════════════════════════════════════════════════════════════
# Conventions de longueur
# ═════════════════════════════════════════════════════════════════════════════
# Deux systèmes coexistent :
#   (a) L_geom = ‖base_i − plat_world_i‖, ~1.27 m au repos (utilisé par IK/FK).
#   (b) L_musc = lecture potentiomètre à fil, 1.0 m au repos par construction
#       (utilisée par Sarosi, convention l₀ = 1.0 m).
# Conversion : L_musc = L_geom − Δᵢ avec Δᵢ = L_repos_geom_i − 1.0 (constante).
def _compute_rest_geom_lengths(base_points: np.ndarray,
                               platform_points: np.ndarray) -> np.ndarray:
    """Longueurs géométriques au repos (m) pour chaque muscle."""
    return np.linalg.norm(base_points - platform_points, axis=1)


REST_GEOM_LENGTHS_M = _compute_rest_geom_lengths(
    STEWART_BASE_POINTS_M, STEWART_PLATFORM_POINTS_M)
LENGTH_OFFSETS_M = REST_GEOM_LENGTHS_M - MUSCLE_NOMINAL_LENGTH_M  # Δᵢ


def geom_to_muscle_lengths(L_geom: Iterable[float],
                            offsets: np.ndarray = LENGTH_OFFSETS_M) -> np.ndarray:
    """Conversion longueur géométrique → longueur muscle (Sarosi/capteur)."""
    return np.asarray(L_geom, dtype=float) - offsets


def muscle_to_geom_lengths(L_musc: Iterable[float],
                            offsets: np.ndarray = LENGTH_OFFSETS_M) -> np.ndarray:
    """Conversion longueur muscle (capteur) → longueur géométrique (IK/FK)."""
    return np.asarray(L_musc, dtype=float) + offsets


# ═════════════════════════════════════════════════════════════════════════════
# 1) Modèle Sarosi — forward (pour vérification / sanity-check)
# ═════════════════════════════════════════════════════════════════════════════
def pam_force_sarosi_N(pressure_pa: float, kappa: float) -> float:
    """Force de traction du muscle (N) selon le modèle de Sarosi.

    Paramètres
    ----------
    pressure_pa : pression interne (Pa). Plage modèle : [0, 6e5].
    kappa       : contraction relative (sans unité), κ = (l₀ − l)/l₀.
                  Plage modèle : [0, 0.25].

    Retour
    ------
    Force axiale produite par le muscle (N), positive en traction.
    """
    p = float(pressure_pa)
    k = float(kappa)
    return ((SAROSI_A1 * p + SAROSI_A2) * math.exp(SAROSI_A3 * k)
            + SAROSI_A4 * k * p
            + SAROSI_A5 * p
            + SAROSI_A6)


# ═════════════════════════════════════════════════════════════════════════════
# 2) Modèle Sarosi — inverse (closed-form, c'est l'opération clé du feedforward)
# ═════════════════════════════════════════════════════════════════════════════
def pam_pressure_sarosi_bar(
        force_N: float,
        length_m: float,
        l0_m: float = MUSCLE_NOMINAL_LENGTH_M,
        ) -> Tuple[float, dict]:
    """Inverse analytique Sarosi : pression (bar) à commander pour produire
    `force_N` quand le muscle mesure `length_m`.

    À κ fixé, F est affine en p : F = α(κ)·p + β(κ), d'où p = (F − β)/α.
    α(κ) > 0 sur [0, 0.25] avec les coefficients Mehdi.

    Retourne (p_bar, info) ; p clampée à [P_MIN, P_MAX]. info contient
    kappa, p_unclamped_bar, force_min/max_N (à 0/6 bar), saturated.
    """
    # 1) Contraction relative, clampée dans la plage du modèle
    kappa_raw = (l0_m - length_m) / l0_m
    kappa = max(0.0, min(KAPPA_MAX, kappa_raw))
    kappa_clamped = (kappa != kappa_raw)

    # 2) Pré-calcul des deux termes
    exp_term = math.exp(SAROSI_A3 * kappa)
    alpha = SAROSI_A1 * exp_term + SAROSI_A4 * kappa + SAROSI_A5
    beta  = SAROSI_A2 * exp_term + SAROSI_A6

    # 3) Pression brute (Pa) puis conversion en bar
    if abs(alpha) < 1e-30:
        p_pa_unclamped = 0.0
    else:
        p_pa_unclamped = (force_N - beta) / alpha

    p_bar_unclamped = p_pa_unclamped * PA_TO_BAR

    # 4) Saturation aux bornes physiques de la vanne / plage du modèle
    p_bar = max(P_MIN_BAR, min(P_MAX_BAR, p_bar_unclamped))
    saturated = (p_bar != p_bar_unclamped)

    # 5) Forces atteignables aux deux bornes (utile pour diagnostiquer
    #    pourquoi on sature) — calculées en Pa puis renvoyées en N.
    force_min_N = pam_force_sarosi_N(P_MIN_BAR * BAR_TO_PA, kappa)
    force_max_N = pam_force_sarosi_N(P_MAX_BAR * BAR_TO_PA, kappa)

    return p_bar, {
        'kappa'           : kappa,
        'kappa_clamped'   : kappa_clamped,
        'p_unclamped_bar' : p_bar_unclamped,
        'force_min_N'     : force_min_N,
        'force_max_N'     : force_max_N,
        'saturated'       : saturated,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3) Équilibre statique 6×6 → tensions axiales requises par muscle
# ═════════════════════════════════════════════════════════════════════════════
def compute_static_tensions_N(
        translation_m: Iterable[float],
        rotation_rad:  Iterable[float],
        mass_kg: float,
        com_offset_m: Iterable[float] = (0.0, 0.0, 0.0),
        gravity: float = GRAVITY_M_S2,
        base_points: np.ndarray = STEWART_BASE_POINTS_M,
        platform_points: np.ndarray = STEWART_PLATFORM_POINTS_M,
        ) -> Tuple[np.ndarray, dict]:
    """Tensions axiales (N) qui équilibrent statiquement la plateforme.

    Résout le système 6×6 équilibre forces + moments sous gravité, avec
    uᵢ = (base_i − plat_world_i)/‖·‖ (la base est AU-DESSUS, uᵢ tire vers
    le haut). Pose singulière (cond > 1e10) → fallback uniforme.

    Retourne (T (6,), info). T peut contenir des valeurs négatives si la
    pose est instable — l'appelant décide.
    """
    R = _euler_rotation_matrix(rotation_rad)
    t = np.asarray(translation_m, dtype=float)
    com_local = np.asarray(com_offset_m, dtype=float)
    com_world = R @ com_local + t

    JT = np.zeros((6, 6), dtype=float)
    for i in range(6):
        p_world = R @ platform_points[i] + t
        vec = base_points[i] - p_world
        L = float(np.linalg.norm(vec))
        if L < 1e-9:
            # Plateforme qui touche la base : cas pathologique
            return (np.full(6, mass_kg * gravity / 6.0),
                    {'singular': True, 'condition': float('inf'),
                     'used_fallback': True})
        u = vec / L
        r = p_world - com_world
        JT[0:3, i] = u
        JT[3:6, i] = np.cross(r, u)

    w = np.array([0.0, 0.0, mass_kg * gravity, 0.0, 0.0, 0.0])
    cond = float(np.linalg.cond(JT))
    singular = (not math.isfinite(cond)) or cond > 1.0e10

    if singular:
        u_z_mean = float(np.mean(JT[2, :]))
        if abs(u_z_mean) < 1e-6:
            T = np.zeros(6)
        else:
            T = np.full(6, mass_kg * gravity / (6.0 * u_z_mean))
        return T, {'singular': True, 'condition': cond, 'used_fallback': True}

    T = np.linalg.solve(JT, w)
    return T, {'singular': False, 'condition': cond, 'used_fallback': False}


# ═════════════════════════════════════════════════════════════════════════════
# 4) Pipeline complet : pose cible → 6 pressions
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class FeedforwardCommand:
    """Sortie du calcul feedforward pour une pose cible."""
    target_lengths_m  : np.ndarray   # (6,) longueurs cibles (IK)
    tensions_N        : np.ndarray   # (6,) tensions requises (équilibre)
    pressures_bar     : np.ndarray   # (6,) pressions à envoyer aux vannes
    pressures_unclamped_bar: np.ndarray  # (6,) avant saturation [0, 6]
    kappa             : np.ndarray   # (6,) contractions utilisées
    saturated_mask    : np.ndarray   # (6,) bool, vannes en saturation
    negative_tension  : np.ndarray   # (6,) bool, muscles que le solver veut "pousser"
    static_info       : dict         # diagnostic du solveur d'équilibre

    def summary(self) -> str:
        lines = [
            "Tensions (N)   : " + ", ".join(f"{x:7.1f}" for x in self.tensions_N),
            "Pressions (bar): " + ", ".join(f"{x:5.2f}"  for x in self.pressures_bar),
            "κ (%)          : " + ", ".join(f"{x*100:5.2f}" for x in self.kappa),
        ]
        if self.negative_tension.any():
            idx = np.where(self.negative_tension)[0]
            lines.append(f"⚠ Tensions négatives sur muscles {idx.tolist()} "
                         f"(pose statiquement instable, clampées à P_MIN)")
        if self.saturated_mask.any():
            idx = np.where(self.saturated_mask)[0]
            lines.append(f"⚠ Saturation pression sur muscles {idx.tolist()}")
        if self.static_info.get('singular'):
            lines.append(f"⚠ Système d'équilibre singulier (cond={self.static_info['condition']:.2e})")
        return "\n".join(lines)


def compute_feedforward_pressures(
        target_translation_m: Iterable[float],
        target_rotation_rad:  Iterable[float],
        mass_kg: float,
        com_offset_m: Iterable[float] = (0.0, 0.0, 0.0),
        l0_muscle_m: float = MUSCLE_NOMINAL_LENGTH_M,
        gravity: float = GRAVITY_M_S2,
        base_points: np.ndarray = STEWART_BASE_POINTS_M,
        platform_points: np.ndarray = STEWART_PLATFORM_POINTS_M,
        ) -> FeedforwardCommand:
    """Pipeline feedforward complet pour une pose cible.

    IK → 6 L_target ; équilibre statique → 6 tensions T ; Sarosi inverse
    (T_i, L_target_i) → 6 pressions. Aucune mesure n'intervient. Tensions
    négatives clampées à 0 N (un muscle ne peut pas pousser).

    Retourne FeedforwardCommand.
    """
    # ─ Étape 1 : IK ──────────────────────────────────────────────────────────
    L_target = stewart_inverse_kinematics(
        target_translation_m, target_rotation_rad,
        base_points=base_points, platform_points=platform_points,
    )

    # ─ Étape 2 : équilibre statique ──────────────────────────────────────────
    T, static_info = compute_static_tensions_N(
        target_translation_m, target_rotation_rad,
        mass_kg=mass_kg, com_offset_m=com_offset_m,
        gravity=gravity,
        base_points=base_points, platform_points=platform_points,
    )

    # Un muscle ne peut pas pousser → tensions négatives clampées à 0.
    negative = (T < 0.0)
    T_clamped = np.where(negative, 0.0, T)

    # ─ Étape 3 : Sarosi inverse muscle par muscle ────────────────────────────
    pressures_bar           = np.empty(6, dtype=float)
    pressures_unclamped_bar = np.empty(6, dtype=float)
    kappa_arr               = np.empty(6, dtype=float)
    saturated               = np.zeros(6, dtype=bool)

    # Longueurs CIBLES géométriques → muscle (Sarosi valide en repère muscle).
    # On utilise la cible, pas la mesure : feedforward strict.
    L_target_muscle = geom_to_muscle_lengths(L_target)

    for i in range(6):
        p_bar_i, info_i = pam_pressure_sarosi_bar(
            force_N=float(T_clamped[i]),
            length_m=float(L_target_muscle[i]),
            l0_m=l0_muscle_m,
        )
        pressures_bar[i]           = p_bar_i
        pressures_unclamped_bar[i] = info_i['p_unclamped_bar']
        kappa_arr[i]               = info_i['kappa']
        saturated[i]               = info_i['saturated']

    return FeedforwardCommand(
        target_lengths_m        = L_target,
        tensions_N              = T,                          # non clampées (info brute)
        pressures_bar           = pressures_bar,
        pressures_unclamped_bar = pressures_unclamped_bar,
        kappa                   = kappa_arr,
        saturated_mask          = saturated,
        negative_tension        = negative,
        static_info             = static_info,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 5) Erreur de pose à partir des longueurs mesurées (forward kinematics)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class PoseError:
    """Erreur 6-DOF entre pose cible et pose mesurée."""
    measured_translation_m : np.ndarray  # (3,) pose courante par FK
    measured_rotation_rad  : np.ndarray  # (3,)
    target_translation_m   : np.ndarray  # (3,)
    target_rotation_rad    : np.ndarray  # (3,)
    error_translation_m    : np.ndarray  # (3,) target − measured
    error_rotation_rad     : np.ndarray  # (3,) target − measured
    fk_info                : dict        # diagnostic FK (convergence, résidu, ...)

    @property
    def position_error_norm_m(self) -> float:
        """‖erreur translation‖₂ en m."""
        return float(np.linalg.norm(self.error_translation_m))

    @property
    def rotation_error_norm_deg(self) -> float:
        """‖erreur rotation‖₂ en degrés."""
        return float(math.degrees(np.linalg.norm(self.error_rotation_rad)))

    def summary(self) -> str:
        et = self.error_translation_m * 1000.0  # → mm
        er = np.degrees(self.error_rotation_rad)  # → °
        lines = [
            f"FK : {'OK' if self.fk_info['converged'] else 'NON-CONVERGÉE'} "
            f"({self.fk_info['iterations']} it, résidu max = "
            f"{self.fk_info['max_residual']*1e6:.1f} µm)",
            f"Pose mesurée   : t=({self.measured_translation_m[0]*1000:+6.2f}, "
            f"{self.measured_translation_m[1]*1000:+6.2f}, "
            f"{self.measured_translation_m[2]*1000:+6.2f}) mm   "
            f"r=({math.degrees(self.measured_rotation_rad[0]):+5.2f}, "
            f"{math.degrees(self.measured_rotation_rad[1]):+5.2f}, "
            f"{math.degrees(self.measured_rotation_rad[2]):+5.2f})°",
            f"Pose cible     : t=({self.target_translation_m[0]*1000:+6.2f}, "
            f"{self.target_translation_m[1]*1000:+6.2f}, "
            f"{self.target_translation_m[2]*1000:+6.2f}) mm   "
            f"r=({math.degrees(self.target_rotation_rad[0]):+5.2f}, "
            f"{math.degrees(self.target_rotation_rad[1]):+5.2f}, "
            f"{math.degrees(self.target_rotation_rad[2]):+5.2f})°",
            f"Erreur t (mm)  : ({et[0]:+6.2f}, {et[1]:+6.2f}, {et[2]:+6.2f})  "
            f"‖·‖ = {self.position_error_norm_m*1000:5.2f} mm",
            f"Erreur r (°)   : ({er[0]:+5.2f}, {er[1]:+5.2f}, {er[2]:+5.2f})  "
            f"‖·‖ = {self.rotation_error_norm_deg:5.2f}°",
        ]
        return "\n".join(lines)


def compute_pose_error(
        measured_muscle_lengths_m: Iterable[float],
        target_translation_m: Iterable[float],
        target_rotation_rad:  Iterable[float],
        initial_guess_translation_m: Iterable[float] = (0.0, 0.0, 0.0),
        initial_guess_rotation_rad:  Iterable[float] = (0.0, 0.0, 0.0),
        lengths_are_geometric: bool = False,
        base_points: np.ndarray = STEWART_BASE_POINTS_M,
        platform_points: np.ndarray = STEWART_PLATFORM_POINTS_M,
        ) -> PoseError:
    """Erreur 6-DOF entre pose cible et pose reconstruite par FK depuis
    les longueurs de muscle mesurées.

    Conversion L_musc → L_geom puis FK warm-startée. Passer la pose mesurée
    précédente comme initial_guess pour converger en 1-3 itérations.
    """
    L_in = np.asarray(measured_muscle_lengths_m, dtype=float)
    if lengths_are_geometric:
        L_geom = L_in
    else:
        L_geom = muscle_to_geom_lengths(L_in)

    t_meas, r_meas, fk_info = stewart_forward_kinematics(
        L_geom,
        initial_translation_m=initial_guess_translation_m,
        initial_rotation_rad=initial_guess_rotation_rad,
        base_points=base_points,
        platform_points=platform_points,
    )

    t_target = np.asarray(target_translation_m, dtype=float)
    r_target = np.asarray(target_rotation_rad,  dtype=float)

    return PoseError(
        measured_translation_m = t_meas,
        measured_rotation_rad  = r_meas,
        target_translation_m   = t_target,
        target_rotation_rad    = r_target,
        error_translation_m    = t_target - t_meas,
        error_rotation_rad     = r_target - r_meas,
        fk_info                = fk_info,
    )

