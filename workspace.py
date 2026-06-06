#!/usr/bin/env python3
"""
workspace.py
============
Vérification d'appartenance au workspace géométrique de la plateforme et
projection d'une pose cible vers la pose la plus proche atteignable.

Workspace géométrique = ensemble des poses (t, θ) telles que les 6 longueurs
de muscle (capteur) restent dans [L_MUSC_MIN, L_MUSC_MAX].

Note : on ignore volontairement la faisabilité en pression (équilibre statique).
Seule la course mécanique des muscles est testée.
"""
from __future__ import annotations

import numpy as np

from geometry import (
    STEWART_BASE_POINTS_M,
    STEWART_PLATFORM_POINTS_M,
    stewart_inverse_kinematics,
)

# ─────────────────────────────────────────────────────────────────────────────
# Bornes de course muscle (avec marge de sécurité par rapport au constructeur)
# ─────────────────────────────────────────────────────────────────────────────
# DMSP-40-400N : κ ∈ [-0.5 %, 25 %] ⇒ L_musc ∈ [0.75, 1.005] m théoriques.
# On garde une marge raisonnable : 0.78 → 1.00 m.
L_MUSC_MIN = 0.78
L_MUSC_MAX = 1.00

# Tolérance numérique sur les bornes (la pose de repos donne L = 1.0 + ε flottant
# qui sortirait de [0.78, 1.00] par 1e-16 ; on accepte ces erreurs d'arrondi).
_EPS = 1e-9

# Offsets longueur géométrique → longueur muscle (capteur), calculés une fois.
# L_musc_i = L_geom_i − Δ_i,  Δ_i = ‖base_i − plat_i‖_repos − 1.0
_REST_GEOM_LENGTHS_M = np.linalg.norm(
    STEWART_BASE_POINTS_M - STEWART_PLATFORM_POINTS_M, axis=1
)
LENGTH_OFFSETS_M = _REST_GEOM_LENGTHS_M - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def muscle_lengths_for_pose(translation_m, rotation_rad) -> np.ndarray:
    """6 longueurs muscle (capteur, m) pour une pose donnée."""
    L_geom = stewart_inverse_kinematics(translation_m, rotation_rad)
    return L_geom - LENGTH_OFFSETS_M


def is_pose_in_workspace(translation_m, rotation_rad,
                         l_min: float = L_MUSC_MIN,
                         l_max: float = L_MUSC_MAX) -> bool:
    """True ssi les 6 muscles tiennent dans [l_min, l_max] pour cette pose."""
    L = muscle_lengths_for_pose(translation_m, rotation_rad)
    return bool(np.all((L >= l_min - _EPS) & (L <= l_max + _EPS)))


# ─────────────────────────────────────────────────────────────────────────────
# Projection : pose cible → pose atteignable la plus proche
# ─────────────────────────────────────────────────────────────────────────────
def clamp_pose_to_workspace(translation_m, rotation_rad,
                            l_min: float = L_MUSC_MIN,
                            l_max: float = L_MUSC_MAX,
                            reference_translation_m=(0.0, 0.0, 0.0),
                            reference_rotation_rad=(0.0, 0.0, 0.0),
                            tol: float = 1e-4,
                            max_iter: int = 50):
    """
    Projette une pose cible (t, θ) sur le workspace géométrique.

    Si la pose cible est faisable, renvoyée telle quelle.
    Sinon, on cherche λ ∈ [0, 1] maximal tel que la pose interpolée
        t_proj = ref_t + λ · (t − ref_t)
        r_proj = ref_r + λ · (r − ref_r)
    soit faisable, par bissection.

    Paramètres
    ----------
    translation_m, rotation_rad : pose cible (m, rad)
    l_min, l_max                : bornes de course muscle (m)
    reference_translation_m, reference_rotation_rad :
        pose d'ancrage (origine du clamp). Par défaut = repos (0, 0, 0, 0, 0, 0).
  
    Renvoie
    -------
    translation_clamped : np.ndarray (3,)  m
    rotation_clamped    : np.ndarray (3,)  rad
    info : dict
        in_workspace   (bool)        — la pose d'origine était-elle faisable ?
        scale          (float)       — λ ∈ [0, 1] appliqué (1.0 = pas de clamp)
        muscle_lengths (np.ndarray)  — 6 longueurs muscle à la pose renvoyée
        violated       (np.ndarray)  — masque des muscles qui dépassaient avant clamp

    Notes
    -----
    On scale **linéairement** la pose entière (translation et rotation). C'est
    la projection la plus naturelle pour du motion-cueing : on conserve la
    direction du mouvement demandé, on en réduit l'amplitude. Pas de pose
    surprenante (pas de tilt parasite injecté pour minimiser une métrique
    SE(3) arbitraire).
    """
    t      = np.asarray(translation_m, dtype=float).copy()
    r      = np.asarray(rotation_rad, dtype=float).copy()
    t_ref  = np.asarray(reference_translation_m, dtype=float)
    r_ref  = np.asarray(reference_rotation_rad,  dtype=float)

    # Longueurs à la cible
    L_target = muscle_lengths_for_pose(t, r)
    violated_target = (L_target < l_min - _EPS) | (L_target > l_max + _EPS)

    # Cible déjà faisable → rien à faire
    if not violated_target.any():
        return t, r, {
            "in_workspace": True,
            "scale": 1.0,
            "muscle_lengths": L_target,
            "violated": violated_target,
        }

    # Vérification : la pose de référence doit être faisable
    if not is_pose_in_workspace(t_ref, r_ref, l_min, l_max):
        raise RuntimeError(
            f"Pose de référence {t_ref}, {r_ref} hors workspace "
            f"[{l_min}, {l_max}] — choisis une autre référence."
        )

    # Bissection sur λ ∈ [0, 1] : pose(λ) = ref + λ·(cible − ref)
    def pose_at(lam):
        return t_ref + lam * (t - t_ref), r_ref + lam * (r - r_ref)

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        t_m, r_m = pose_at(mid)
        if is_pose_in_workspace(t_m, r_m, l_min, l_max):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    scale = lo
    t_clamped, r_clamped = pose_at(scale)
    L_clamped = muscle_lengths_for_pose(t_clamped, r_clamped)

    return t_clamped, r_clamped, {
        "in_workspace": False,
        "scale": scale,
        "muscle_lengths": L_clamped,
        "violated": violated_target,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Recherche du centre du workspace (muscles à mi-course)
# ─────────────────────────────────────────────────────────────────────────────
def find_workspace_center_z(target_kappa: float = 0.125) -> float:
    """
    Hauteur Z (m, > 0) telle que la longueur muscle moyenne corresponde
    à une contraction κ donnée (par défaut 12.5 % = mi-course DMSP).

    Utile pour définir une pose de référence centrée :
        z_c = find_workspace_center_z()
        clamp_pose_to_workspace(t, r, reference_translation_m=(0, 0, z_c))
    """
    target_L = 1.0 * (1.0 - target_kappa)  # 0.875 m pour κ = 12.5 %
    lo, hi = 0.0, 0.4
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        L_mean = muscle_lengths_for_pose([0, 0, mid], [0, 0, 0]).mean()
        if L_mean > target_L:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


