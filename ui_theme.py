#!/usr/bin/env python3
"""
ui_theme.py
===========
Constantes de style (palette dark, polices) et helpers Tkinter partagés
par toute la GUI.
"""
from __future__ import annotations

import tkinter as tk


# ─────────────────────────────────────────────────────────────────────────────
# Palette de couleurs — dark theme (inspiration GitHub)
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#0d1117"   # fond principal
BG2     = "#161b22"   # fond secondaire (cartes)
BG3     = "#21262d"   # fond tertiaire (champs)
ACCENT  = "#58a6ff"   # bleu accent
GREEN   = "#3fb950"
RED     = "#f85149"
ORANGE  = "#ffa657"
PURPLE  = "#d2a8ff"
YELLOW  = "#e3b341"
CYAN    = "#79c0ff"
TEXT    = "#e6edf3"   # texte principal
DIMTEXT = "#8b949e"   # texte secondaire
BORDER  = "#30363d"

# ─────────────────────────────────────────────────────────────────────────────
# Polices
# ─────────────────────────────────────────────────────────────────────────────
FMONO  = ("Courier New", 10)
FLABEL = ("Helvetica", 9, "bold")
FTITLE = ("Helvetica", 10, "bold")
FBIG   = ("Helvetica", 13, "bold")

# ─────────────────────────────────────────────────────────────────────────────
# Couleurs sémantiques
# ─────────────────────────────────────────────────────────────────────────────
AXIS_COLORS   = {"X": ACCENT, "Y": GREEN,  "Z": PURPLE}
GYRO_COLORS   = {"X": ORANGE, "Y": RED,    "Z": CYAN}
SENSOR_COLORS = [ACCENT, GREEN, PURPLE, ORANGE, RED, YELLOW]


def card(parent, title: str) -> tk.Frame:
    """Crée une carte avec un titre et retourne le Frame intérieur où placer
    les widgets enfants.

        outer ──┐ (bordure 1 px de couleur BORDER)
                └── wrapper (BG2, padding) ──┐
                                              ├── titre
                                              └── inner  ← retourné
    """
    outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
    outer.pack(fill="x", pady=(0, 8))
    wrapper = tk.Frame(outer, bg=BG2, padx=8, pady=8)
    wrapper.pack(fill="both", expand=True)
    tk.Label(wrapper, text=title, font=FTITLE, bg=BG2, fg=TEXT).pack(
        anchor="w", pady=(0, 6))
    inner = tk.Frame(wrapper, bg=BG2)
    inner.pack(fill="both", expand=True)
    return inner
