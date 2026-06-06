#!/usr/bin/env python3
"""
position_filters.py — Filtre temps réel sur les longueurs muscle
==================================================================
Atténue le bruit fil-pot en amont
de la FK et du PID, via un passe-bas IIR 1er ordre appliqué muscle par
muscle (PositionFilterBank).

Filtre en MÈTRES (pas en raw ADC : Pythagore rendrait le filtre non
linéaire en m).

"""
from __future__ import annotations

import math
from typing import Sequence


# ═════════════════════════════════════════════════════════════════════════════
# Passe-bas IIR 1er ordre
# ═════════════════════════════════════════════════════════════════════════════
class FirstOrderIIR:
    """Passe-bas exponentiel scalaire : y[n] = α·x[n] + (1−α)·y[n−1],
    α = dt/(RC+dt), RC = 1/(2π·fc). fc_hz ≤ 0 → désactivé."""

    __slots__ = ("fc_hz", "y", "_initialized")

    def __init__(self, fc_hz: float = 6.0, seed: float = 0.0):
        self.fc_hz = float(fc_hz)
        self.y = float(seed)
        self._initialized = False

    def update(self, x: float, dt: float) -> float:
        if self.fc_hz <= 0.0 or dt <= 1e-9:
            self.y = float(x)
            self._initialized = True
            return self.y

        # Première mesure : init à la valeur courante (évite l'amorçage depuis 0).
        if not self._initialized:
            self.y = float(x)
            self._initialized = True
            return self.y

        rc = 1.0 / (2.0 * math.pi * self.fc_hz)
        alpha = dt / (rc + dt)
        self.y += alpha * (float(x) - self.y)
        return self.y

    def reset(self, seed: float = 0.0) -> None:
        self.y = float(seed)
        self._initialized = False


# ═════════════════════════════════════════════════════════════════════════════
# PositionFilterBank — 6 IIR en parallèle (un par muscle)
# ═════════════════════════════════════════════════════════════════════════════
class PositionFilterBank:
    """6 IIR parallèles. fc_hz modifiable à chaud via set_params (reset propre)."""

    def __init__(self, fc_hz: float = 6.0, n_muscles: int = 6):
        self.n = int(n_muscles)
        self.fc_hz = float(fc_hz)
        self._lp = [FirstOrderIIR(self.fc_hz) for _ in range(self.n)]

    def update(self, L_in: Sequence[float], dt: float) -> list:
        """Filtre 6 longueurs muscle. Retourne la liste filtrée."""
        out = [0.0] * self.n
        for m in range(self.n):
            out[m] = self._lp[m].update(float(L_in[m]), dt)
        return out

    def reset(self, seed_values: Sequence[float] | None = None) -> None:
        """seed_values amorce chaque chaîne pour éviter le transitoire."""
        for m in range(self.n):
            seed_m = float(seed_values[m]) if seed_values is not None else 0.0
            self._lp[m].reset(seed_m)

    def set_params(
        self,
        fc_hz: float | None = None,
        seed_values: Sequence[float] | None = None,
    ) -> None:
        """Modifie fc à chaud + reset propre."""
        if fc_hz is not None:
            self.fc_hz = float(fc_hz)
            for f in self._lp:
                f.fc_hz = self.fc_hz
        self.reset(seed_values)


