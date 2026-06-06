#!/usr/bin/env python3
"""
geometry.py
===========
Géométrie de la plateforme Stewart pneumatique d'Aeropoly et mappings entre
indexation logique (muscle 0 = A … 5 = F) et matériel (canaux ADC, vannes DAC).

Convention de repère
--------------------
- Origine = centre de la plateforme au repos.
- Plateforme mobile : 6 points d'attache coplanaires (z = 0), cercle de rayon
  ~690 mm, 3 paires rapprochées (A-B, C-D, E-F) — c'est une Stewart 6-6 inversée.
- Base fixe : 6 points sur un cercle plus large (~1210 mm) à z ≈ +951 mm
  (donc « au-dessus » dans le repère mathématique, la plateforme pend dessous).
- Muscle i au repos ≈ 1274 mm, incliné 41,5° / verticale.
"""
from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Points d'attache mesurés (en millimètres)
# ─────────────────────────────────────────────────────────────────────────────
# Indexation : muscle i relie platform_points[i] à base_points[i]
STEWART_PLATFORM_POINTS_MM = np.array([
    [-533.56,  437.96, 0.00],   # A → muscle 0
    [-646.06,  243.11, 0.00],   # B → muscle 1
    [-112.50, -681.05, 0.00],   # C → muscle 2
    [ 112.50, -681.05, 0.00],   # D → muscle 3
    [ 646.06,  243.11, 0.00],   # E → muscle 4
    [ 533.56,  437.96, 0.00],   # F → muscle 5
])

STEWART_BASE_POINTS_MM = np.array([
    [-160.56,  1199.07, 951.00],   # point 0 → muscle 0
    [-1125.62, -457.86, 949.62],   # point 1 → muscle 1
    [-955.07,  -743.91, 953.47],   # point 2 → muscle 2
    [ 953.53,  -737.09, 955.13],   # point 3 → muscle 3
    [ 1121.72, -455.12, 953.51],   # point 4 → muscle 4
    [ 160.60,  1199.41, 950.76],   # point 5 → muscle 5
])

# Conversion mm → m (tout le reste du code travaille en mètres)
STEWART_PLATFORM_POINTS_M = STEWART_PLATFORM_POINTS_MM / 1000.0
STEWART_BASE_POINTS_M     = STEWART_BASE_POINTS_MM     / 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Mapping muscle logique ↔ matériel
# ─────────────────────────────────────────────────────────────────────────────
MUSCLE_LABELS = ["A", "B", "C", "D", "E", "F"]

# Vanne DAC associée à chaque muscle logique (muscle i → vanne VALVE_OF_MUSCLE[i])
VALVE_OF_MUSCLE = [5, 2, 4, 0, 1, 3]

# Canal ADC position associé à chaque muscle (muscle i → canal ADC_CH_OF_MUSCLE[i])
ADC_CH_OF_MUSCLE = [0, 4, 2, 1, 5, 3]

# Vérification d'intégrité : permutations de [0..5]
assert sorted(VALVE_OF_MUSCLE)  == list(range(6)), \
    "VALVE_OF_MUSCLE must be a permutation of 0..5"
assert sorted(ADC_CH_OF_MUSCLE) == list(range(6)), \
    "ADC_CH_OF_MUSCLE must be a permutation of 0..5"


# ─────────────────────────────────────────────────────────────────────────────
# Cinématique inverse (formule de Mehdi)
# ─────────────────────────────────────────────────────────────────────────────
def _euler_rotation_matrix(rotation_rad):
    """R(rx, ry, rz) = Rz(rz) · Ry(ry) · Rx(rx) — convention Mehdi/FK."""
    rx, ry, rz = rotation_rad
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return np.array([
        [cy * cz, -cx * sz + sx * sy * cz,  sx * sz + cx * sy * cz],
        [cy * sz,  cx * cz + sx * sy * sz, -sx * cz + cx * sy * sz],
        [-sy,      sx * cy,                 cx * cy],
    ])


def stewart_inverse_kinematics(translation_m=(0.0, 0.0, 0.0),
                               rotation_rad=(0.0, 0.0, 0.0),
                               base_points=STEWART_BASE_POINTS_M,
                               platform_points=STEWART_PLATFORM_POINTS_M):
    """6 longueurs de muscle (m) pour une pose cible : L_i = ‖R(θ)·p_i + t − b_i‖.

    translation_m en m, rotation_rad en (roll, pitch, yaw).
    """
    R = _euler_rotation_matrix(rotation_rad)
    t = np.asarray(translation_m, dtype=float)
    out = np.empty(6, dtype=float)
    for i in range(6):
        p_world = R @ platform_points[i] + t
        out[i]  = np.linalg.norm(p_world - base_points[i])
    return out
