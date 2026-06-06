#!/usr/bin/env python3
"""
auto_sequence.py
================
Sequence de caracterisation sur la plateforme Stewart
EPFL / Aeropoly - version "carac controleur ~3 min" orientee sur 3 axes :

    1. BLOCK_BOUNDARIES_EDGE    : 10 points pres des bords du workspace,
                                  1 DOF isole par point.
    2. BLOCK_COMBO_EDGE         : 5 points multi-DOF pres des bords
                                  (test couplage non-lineaire).
    3. BLOCK_HYSTERESIS_RAMP_Z  : rampe Z aller-retour monotone sans
                                  retour HOME entre paliers (hysteresis
                                  directionnelle pure).
    4. BLOCK_REPEATABILITY_MID  : 5 visites identiques de X=+0.10
                                  (plancher de bruit a amplitude moderee).
    5. BLOCK_REPEATABILITY_EDGE : 3 visites identiques de X=+0.13
                                  (repetabilite AU BORD).

Principe HOME <-> cible
-----------------------
Pour chaque point :
    1. Aller a HOME (reference neutre, mi-course)
    2. Maintenir HOME pendant HOME_DWELL_S
    3. Aller au point cible
    4. Maintenir wp.dwell_s
    5. Logger en continu

Exception : rampe d'hysteresis = skip_home_return=True, on enchaine
les paliers SANS repasser par HOME pour mesurer l'hysteresis pure.

HOME = tz=0.185 m (mi-course), 0 partout ailleurs. Toutes les poses ont
ete verifiees dans le workspace geometrique (workspace.py).

Budget temps  : ~183 s = 3 min 03 s.

Usage : importe par robot_control_gui.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


# -- Parametres globaux ------------------------------------------------------
HOME_DWELL_S: float = 4
DEFAULT_TARGET_DWELL_S: float = 6


# -- Dataclass waypoint ------------------------------------------------------
@dataclass(frozen=True)
class Waypoint:
    name: str
    tx_m: float
    ty_m: float
    tz_m: float
    rx_deg: float
    ry_deg: float
    rz_deg: float
    dwell_s: float = DEFAULT_TARGET_DWELL_S
    description: str = ""
    skip_home_return: bool = False


# -- Pose de reference (HOME) ------------------------------------------------
HOME = Waypoint(
    name="HOME",
    tx_m=0.0, ty_m=0.0, tz_m=0.185,
    rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
    dwell_s=HOME_DWELL_S,
    description="Pose de reference - visitee entre chaque point cible.",
)


# ============================================================================
#  BLOC 1 - BOUNDARIES (1 DOF isole, pres du bord)
# ============================================================================
# Test : le FF degenere-t-il aux bords ? Asymetrie X+/X-, Y+/Y-, YAW+/YAW- ?
# Valeurs : +/-0.13 m / +/-12 deg / Z=[0.155, 0.225], in-workspace verifie.
BLOCK_BOUNDARIES_EDGE: List[Waypoint] = [
    Waypoint(name="EDGE Z+ 0.225", tx_m=0.0, ty_m=0.0, tz_m=0.225,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Z haut - forte pression sur les 6 muscles (~3 bar)."),
    Waypoint(name="EDGE Z- 0.155", tx_m=0.0, ty_m=0.0, tz_m=0.155,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Z bas - faible pression (~0.5 bar). Zone non-lineaire Sarosi."),
    Waypoint(name="EDGE X+ 0.13", tx_m=+0.13, ty_m=0.0, tz_m=0.185,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Surge avant - desequilibre 3 muscles avant / 3 arriere."),
    Waypoint(name="EDGE X- -0.13", tx_m=-0.13, ty_m=0.0, tz_m=0.185,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Surge arriere - test symetrie vs X+."),
    Waypoint(name="EDGE Y+ 0.13", tx_m=0.0, ty_m=+0.13, tz_m=0.185,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Sway gauche - axe orthogonal a X, geometrie C3 differente."),
    Waypoint(name="EDGE Y- -0.13", tx_m=0.0, ty_m=-0.13, tz_m=0.185,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Sway droite - test symetrie vs Y+."),
    Waypoint(name="EDGE ROLL+ 12", tx_m=0.0, ty_m=0.0, tz_m=0.185,
             rx_deg=+12.0, ry_deg=0.0, rz_deg=0.0,
             description="Roll bord - forte differentielle entre paires de muscles."),
    Waypoint(name="EDGE PITCH+ 12", tx_m=0.0, ty_m=0.0, tz_m=0.185,
             rx_deg=0.0, ry_deg=+12.0, rz_deg=0.0,
             description="Pitch bord - axe orthogonal a roll, geometrie C3 differente."),
    Waypoint(name="EDGE YAW+ 12", tx_m=0.0, ty_m=0.0, tz_m=0.185,
             rx_deg=0.0, ry_deg=0.0, rz_deg=+12.0,
             description="Yaw bord - rotation antagoniste 3 muscles vs 3."),
    Waypoint(name="EDGE YAW- -12", tx_m=0.0, ty_m=0.0, tz_m=0.185,
             rx_deg=0.0, ry_deg=0.0, rz_deg=-12.0,
             description="Yaw bord oppose - symetrie yaw."),
]


# ============================================================================
#  BLOC 2 - COMBOS (multi-DOF pres du bord)
# ============================================================================

BLOCK_COMBO_EDGE: List[Waypoint] = [
    Waypoint(name="COMBO Z+ & PITCH+", tx_m=0.0, ty_m=0.0, tz_m=0.215,
             rx_deg=0.0, ry_deg=+10.0, rz_deg=0.0,
             description="Heave haut + pitch cabre - decollage simule, haute pression."),
    Waypoint(name="COMBO Z- & PITCH-", tx_m=0.0, ty_m=0.0, tz_m=0.165,
             rx_deg=0.0, ry_deg=-10.0, rz_deg=0.0,
             description="Heave bas + pitch pique - descente, basse pression."),
    Waypoint(name="COMBO X+ & PITCH+", tx_m=+0.10, ty_m=0.0, tz_m=0.185,
             rx_deg=0.0, ry_deg=+8.0, rz_deg=0.0,
             description="Surge + pitch - acceleration longitudinale couplee."),
    Waypoint(name="COMBO Y+ & ROLL+", tx_m=0.0, ty_m=+0.10, tz_m=0.185,
             rx_deg=+8.0, ry_deg=0.0, rz_deg=0.0,
             description="Sway + roll - acceleration laterale couplee."),
    Waypoint(name="COMBO TRIPLE TILT", tx_m=0.0, ty_m=0.0, tz_m=0.185,
             rx_deg=+8.0, ry_deg=+8.0, rz_deg=+8.0,
             description="Roll+pitch+yaw simultanes - bord workspace (L_min=0.724 m)."),
]


# ============================================================================
#  BLOC 3 - HYSTERESIS Z (rampe monotone aller-retour)
# ============================================================================
# Memes Z visites dans les 2 sens -> hysteresis directionnelle pure.
# Requiert que le worker GUI respecte skip_home_return (patche).
BLOCK_HYSTERESIS_RAMP_Z: List[Waypoint] = [
    # Aller (Z montante)
    Waypoint(name="HYST^ Z=0.155", tx_m=0.0, ty_m=0.0, tz_m=0.155,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Rampe ascendante - palier bas (reference aller).",
             skip_home_return=True),
    Waypoint(name="HYST^ Z=0.178", tx_m=0.0, ty_m=0.0, tz_m=0.178,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Rampe ascendante - palier bas-intermediaire.",
             skip_home_return=True),
    Waypoint(name="HYST^ Z=0.202", tx_m=0.0, ty_m=0.0, tz_m=0.202,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Rampe ascendante - palier haut-intermediaire.",
             skip_home_return=True),
    Waypoint(name="HYST^ Z=0.225", tx_m=0.0, ty_m=0.0, tz_m=0.225,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Rampe ascendante - sommet (inversion sens).",
             skip_home_return=True),
    # Retour (Z descendante)
    Waypoint(name="HYSTv Z=0.202", tx_m=0.0, ty_m=0.0, tz_m=0.202,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Rampe descendante - comparer a HYST^ Z=0.202.",
             skip_home_return=True),
    Waypoint(name="HYSTv Z=0.178", tx_m=0.0, ty_m=0.0, tz_m=0.178,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Rampe descendante - comparer a HYST^ Z=0.178.",
             skip_home_return=True),
    # Dernier point : skip_home_return=False -> retour HOME avant repetabilite.
    Waypoint(name="HYSTv Z=0.155", tx_m=0.0, ty_m=0.0, tz_m=0.155,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Rampe descendante - fin de rampe, comparer a HYST^ Z=0.155."),
]


# ============================================================================
#  BLOC 4 - REPETABILITE AMPLITUDE MODEREE (5 visites depuis HOME)
# ============================================================================
# sigma_mid = plancher de bruit deterministe-corrigible en zone Sarosi
# quasi-lineaire. Aucun FB ne descendra en dessous.
BLOCK_REPEATABILITY_MID: List[Waypoint] = [
    Waypoint(name="REP MID 1/5 - X=+0.10", tx_m=+0.10, ty_m=0.0, tz_m=0.185,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Visite 1 amplitude moderee - reference pour sigma et tendance."),
    Waypoint(name="REP MID 2/5 - X=+0.10", tx_m=+0.10, ty_m=0.0, tz_m=0.185,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Visite 2 - depart identique HOME."),
    Waypoint(name="REP MID 3/5 - X=+0.10", tx_m=+0.10, ty_m=0.0, tz_m=0.185,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Visite 3 - depart identique HOME."),
    Waypoint(name="REP MID 4/5 - X=+0.10", tx_m=+0.10, ty_m=0.0, tz_m=0.185,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Visite 4 - depart identique HOME."),
    Waypoint(name="REP MID 5/5 - X=+0.10", tx_m=+0.10, ty_m=0.0, tz_m=0.185,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Visite 5 - fin du bloc mid."),
]


# ============================================================================
#  BLOC 5 - REPETABILITE AU BORD (3 visites depuis HOME)
# ============================================================================
BLOCK_REPEATABILITY_EDGE: List[Waypoint] = [
    Waypoint(name="REP EDGE 1/3 - X=+0.13", tx_m=+0.13, ty_m=0.0, tz_m=0.185,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Visite 1 bord workspace - reference sigma_edge."),
    Waypoint(name="REP EDGE 2/3 - X=+0.13", tx_m=+0.13, ty_m=0.0, tz_m=0.185,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Visite 2 - depart identique HOME."),
    Waypoint(name="REP EDGE 3/3 - X=+0.13", tx_m=+0.13, ty_m=0.0, tz_m=0.185,
             rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
             description="Visite 3 - fin de sequence, retour HOME automatique."),
]


# ============================================================================
#  COMPOSITION DE LA SEQUENCE
# ============================================================================
# Ordre : boundaries -> combos -> hysteresis -> rep mid -> rep edge.
SEQUENCE: List[Waypoint] = (
    BLOCK_BOUNDARIES_EDGE        # 10 pts - ~70 s
    + BLOCK_COMBO_EDGE           #  5 pts - ~35 s
    + BLOCK_HYSTERESIS_RAMP_Z    #  7 pts - ~34 s (6 skip + 1 retour HOME)
    + BLOCK_REPEATABILITY_MID    #  5 pts - ~35 s
    + BLOCK_REPEATABILITY_EDGE   #  3 pts - ~21 s
)
# Total : 30 pts - ~183 s = 3 min 03 s (+ 2.5 s HOME initial)


# -- Helpers d'introspection -------------------------------------------------
def total_duration_s(sequence=None, home_dwell_s=HOME_DWELL_S):
    """Estimation de la duree totale (s)."""
    if sequence is None:
        sequence = SEQUENCE
    n = len(sequence)
    if n == 0:
        return 0.0
    total = home_dwell_s
    for i, wp in enumerate(sequence):
        total += wp.dwell_s
        if i < n - 1 and not wp.skip_home_return:
            total += home_dwell_s
    return total


def summary():
    """Resume textuel - affiche dans le GUI au demarrage."""
    n = len(SEQUENCE)
    total = total_duration_s()
    n_skip = sum(1 for wp in SEQUENCE if wp.skip_home_return)
    skip_info = "  -  {} skip-HOME".format(n_skip) if n_skip > 0 else ""
    return ("Sequence: {} points  -  duree: {:.0f} s ({:.1f} min)  -  "
            "home dwell: {:.1f} s  -  target dwell: {:.1f} s{}").format(
        n, total, total / 60, HOME_DWELL_S, DEFAULT_TARGET_DWELL_S, skip_info)


if __name__ == "__main__":
    print(summary())
    print("\n  {:>2}  {:<24}  {:>7}  {:>7}  {:>7}  {:>6}  {:>6}  {:>6}  {:>5}  {:>4}".format(
        "#", "name", "tx", "ty", "tz", "rx", "ry", "rz", "dwell", "skip"))
    for i, wp in enumerate(SEQUENCE, 1):
        skip = "x" if wp.skip_home_return else ""
        print("  {:>2}  {:<24}  {:>+7.3f}  {:>+7.3f}  {:>+7.3f}  {:>+6.1f}  {:>+6.1f}  {:>+6.1f}  {:>5.1f}  {:>4}".format(
            i, wp.name, wp.tx_m, wp.ty_m, wp.tz_m,
            wp.rx_deg, wp.ry_deg, wp.rz_deg, wp.dwell_s, skip))
    print("\nHOME entre chaque (sauf skip): tz={:.3f} m dwell={:.1f} s".format(
        HOME.tz_m, HOME.dwell_s))
