#!/usr/bin/env python3
"""
calibration.py — Calcul des offsets capteur depuis un buffer
=====================================================================
Pas de dépendance Tkinter (testable seul). L'offset au repos par muscle
est simplement la moyenne des raw capturés ; consommé par
`apply_position_correction` qui ramène alors L_muscle à 1.0 m exact au repos.
"""
from __future__ import annotations

from typing import List, Sequence


# ─────────────────────────────────────────────────────────────────────────────
# Calcul des offsets
# ─────────────────────────────────────────────────────────────────────────────
def compute_position_offsets(buf_pos: Sequence[Sequence[float]]) -> List[float]:
    """Moyenne des raw par muscle sur le buffer (6 floats). [0.0]*6 si vide."""
    if not buf_pos:
        return [0.0] * 6
    n = len(buf_pos)
    return [sum(row[m] for row in buf_pos) / n for m in range(6)]


def compute_imu_offsets(buf_imu: Sequence[Sequence[float]]) -> List[float]:
    """Moyenne (ax, ay, az, gx, gy, gz) sur le buffer. [0.0]*6 si vide."""
    if not buf_imu:
        return [0.0] * 6
    n = len(buf_imu)
    return [sum(row[c] for row in buf_imu) / n for c in range(6)]


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
def _selftest():
    """Vérifie qu'un buffer raw constant → L_muscle = 1.000 m exact."""
    from position_adc import apply_position_correction

    # Valeurs raw plausibles au repos (ordre de grandeur des mesures banc) :
    #   2097 ↔ ch4 (CH0-CH4, capteurs 4-20 mA), 1165 ↔ ch5 (0-5 V).
    # On suppose que les 5 capteurs courant ont ~2097 au repos pour le test.
    raw_steady = [2097.0, 2097.0, 2097.0, 2097.0, 2097.0, 1165.0]
    buf = [list(raw_steady) for _ in range(30)]

    offsets = compute_position_offsets(buf)
    corrected = apply_position_correction(raw_steady, offsets)

    print("Test calibration (longueur muscle axiale après Pythagore)")
    print(f"  raw       = {raw_steady}")
    print(f"  offsets   = {[f'{x:.1f}' for x in offsets]}")
    print(f"  L_muscle  = {[f'{x:.6f}' for x in corrected]} m")
    print("  cible     = 1.000000 m  pour chaque muscle")
    ok = all(abs(c - 1.0) < 1e-9 for c in corrected)
    print("RESULT:", "ALL OK ✅" if ok else "FAIL ❌")
    return ok


if __name__ == "__main__":
    _selftest()
