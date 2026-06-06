#!/usr/bin/env python3
"""
position_adc.py
===============
Lecture de l'ADC128D818 (0x1D) — capteurs fil-pot des 6 muscles.
CH0..CH4 : capteurs 4-20 mA via shunt 241 Ω. CH5 : capteur 0-5 V direct.

Chaîne de conversion : fit linéaire direct raw_ADC → L_capteur (m) mesuré
sur banc, ancré au repos par calibration runtime, puis Pythagore vers
l'axe muscle :

    L_capteur = slope × (raw − raw_rest_offset) + L_REST_SENSOR_M
    L_muscle  = √(L_capteur² − SENSOR_AXIS_OFFSET_M²)

Le potentiomètre à fil est décalé parallèlement à l'axe muscle de 6 cm —
fil, muscle et décalage forment un triangle rectangle.
"""
from __future__ import annotations

import math
import time
from smbus2 import SMBus

from geometry import ADC_CH_OF_MUSCLE


# ─────────────────────────────────────────────────────────────────────────────
# Registres et constantes ADC
# ─────────────────────────────────────────────────────────────────────────────
ADC_ADDR         = 0x1D
ADC_VREF         = 5.0      # V — Vref EXTERNE (cf. init_adc)
ADC_R_SHUNT      = 241.0    # Ω — résistance shunt (capteurs 4-20 mA)
ADC_I_MIN        = 0.004    # A — 4 mA  → 0 m
ADC_I_MAX        = 0.020    # A — 20 mA → 1 m
ADC_REG_READINGS = 0x20     # CH0=0x20 … CH5=0x25

# Registre Advanced Configuration:
#   bit 0 : External Reference Enable (0 = interne 2.56 V, 1 = externe)
#   bits 2-1 : Mode Select (00 = 7 SE + temp, 01 = 8 SE)
ADC_REG_ADV_CONFIG       = 0x0B
ADC_ADV_CONFIG_EXT_VREF  = 0b00000001   # Vref externe ON, Mode 0

# ─────────────────────────────────────────────────────────────────────────────
# Géométrie capteur ↔ muscle
# ─────────────────────────────────────────────────────────────────────────────
# Décalage parallèle entre l'axe du capteur fil-pot et l'axe du muscle (m).
# Le capteur mesure l'hypoténuse, le muscle est le côté axial.
SENSOR_AXIS_OFFSET_M = 0.06   # 6 cm

# ─────────────────────────────────────────────────────────────────────────────
# Calibration banc — fit direct raw_ADC → L_capteur (en mètres)
# ─────────────────────────────────────────────────────────────────────────────
# Régression linéaire L_capteur = slope × raw + b sur 6 points couvrant
# 0.75–1.00 m. Fit avec R² > 0.9998, RMSE banc < 1 mm.
# Hypothèse : tous les capteurs courant CH0-CH4 ont la MÊME pente (transmetteur
# 4-20 mA passif, shunt commun, même Vref). Si un muscle montre une erreur
# statique anormale, refaire le banc sur ce canal et créer une pente dédiée.
RAW_TO_LCAP_SLOPE_CURRENT = 3.07545e-4   # m/raw  — CH0-CH4 (4-20 mA)
RAW_TO_LCAP_SLOPE_TENSION = 3.81710e-4   # m/raw  — CH5     (0-5 V)

# Longueur capteur cible au repos (équivalent L_muscle = 1.0 m après
# Pythagore). Sert d'ancrage pour la calibration : `apply_position_correction`
# ramène la moyenne des raw mesurés à `L_REST_SENSOR_M` — la pente fait le
# reste. Avec SENSOR_AXIS_OFFSET_M = 0.06 m → L_REST_SENSOR_M ≈ 1.00180 m.
L_REST_SENSOR_M = math.sqrt(1.0 + SENSOR_AXIS_OFFSET_M ** 2)


def init_adc(bus: SMBus) -> None:
    """Initialise l'ADC128D818 positions avec Vref externe 5 V.

    Cycle complet ≈ 6 × 12 ms = 72 ms (~14 Hz). Ordre obligatoire
    (datasheet §Quick Start) : stop → config → start.
    """
    bus.write_byte_data(ADC_ADDR, 0x00, 0x00)        # Stop
    # Vref EXTERNE (sans ça l'ADC utilise sa Vref interne 2.56 V et toutes
    # les tensions sont surévaluées d'un facteur 5.0/2.56 ≈ 1.95).
    bus.write_byte_data(ADC_ADDR, ADC_REG_ADV_CONFIG,
                        ADC_ADV_CONFIG_EXT_VREF)
    bus.write_byte_data(ADC_ADDR, 0x03, 0xFF)        # Masquer interruptions
    bus.write_byte_data(ADC_ADDR, 0x07, 0x01)        # Conversion continue
    bus.write_byte_data(ADC_ADDR, 0x08, 0b11000000)  # Désactiver CH6, CH7, temp
    bus.write_byte_data(ADC_ADDR, 0x00, 0x01)        # Start
    time.sleep(0.1)                                  # Premier cycle


def read_positions(bus: SMBus) -> list:
    """Lit CH0..CH5 et retourne 6 valeurs RAW (12 bits, 0..4095) indexées
    par muscle logique (muscle 0 = A … 5 = F). Conversion raw → L_capteur
    déléguée à `apply_position_correction`.
    """
    raw_by_ch = [0] * 6
    for ch in range(6):
        data = bus.read_i2c_block_data(ADC_ADDR, ADC_REG_READINGS + ch, 2)
        raw_by_ch[ch] = ((data[0] << 8) | data[1]) >> 4   # 12 bits

    # Remappage CH hardware → muscle logique
    return [raw_by_ch[ADC_CH_OF_MUSCLE[m]] for m in range(6)]


def apply_position_correction(raw_logical: list, raw_rest_offset: list) -> list:
    """Convertit 6 raw ADC en longueurs muscle axiales (m).

    Par muscle : choix de pente (CH5 → tension, CH0-CH4 → courant),
    fit linéaire ancré au repos, puis Pythagore. Au repos exact
    (raw == raw_rest_offset), retourne L_muscle = 1.0 m. Vaut 0.0 si
    L_capteur < 6 cm (cas pathologique : contraction > 100 %).
    """
    out = []
    offset_sq = SENSOR_AXIS_OFFSET_M * SENSOR_AXIS_OFFSET_M
    for m in range(6):
        hw_ch = ADC_CH_OF_MUSCLE[m]
        if hw_ch == 5:
            slope = RAW_TO_LCAP_SLOPE_TENSION
        else:
            slope = RAW_TO_LCAP_SLOPE_CURRENT

        l_sensor = slope * (raw_logical[m] - raw_rest_offset[m]) + L_REST_SENSOR_M

        # Pythagore
        l_sq = l_sensor * l_sensor - offset_sq
        l_muscle = math.sqrt(l_sq) if l_sq > 0.0 else 0.0
        out.append(l_muscle)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Mode debug : affichage continu des valeurs brutes ADC dans le terminal
# Lancer avec :  python3 position_adc.py
# Ctrl+C pour quitter.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import csv
    from datetime import datetime

    # Inverse de ADC_CH_OF_MUSCLE : pour chaque canal hardware, le muscle logique.
    MUSCLE_OF_ADC_CH = [ADC_CH_OF_MUSCLE.index(ch) for ch in range(6)]

    # Fichier CSV horodaté
    csv_name = f"adc_position_log_{datetime.now():%Y%m%d_%H%M%S}.csv"

    # En-tête : timestamp + (raw, V, pos_norm) pour chaque canal
    header = ["timestamp"]
    for ch in range(6):
        header += [f"ch{ch}_raw", f"ch{ch}_V", f"ch{ch}_pos_norm"]

    print("Initialisation ADC positions (0x1D)...")
    print(f"Log écrit dans : {csv_name}")
    with SMBus(1) as bus, open(csv_name, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(header)
        fcsv.flush()

        init_adc(bus)
        print("OK. Lecture en boucle. Ctrl+C pour quitter.\n")

        try:
            while True:
                # Effacer l'écran + curseur en haut (ANSI)
                print("\033[2J\033[H", end="")
                print("=== ADC128D818 positions (0x1D) — valeurs brutes ===")
                print(f"Log : {csv_name}\n")
                print(f"{'CH':>2} | {'raw':>5} | {'V':>6} | {'mA':>6} | "
                      f"{'pos_norm':>8} | {'muscle':>6} | type")
                print("-" * 60)

                ts  = datetime.now().isoformat(timespec="milliseconds")
                row = [ts]

                for ch in range(6):
                    data    = bus.read_i2c_block_data(ADC_ADDR,
                                                     ADC_REG_READINGS + ch, 2)
                    raw     = ((data[0] << 8) | data[1]) >> 4         # 12 bits
                    voltage = (raw / 4095.0) * ADC_VREF

                    if ch == 5:
                        # Capteur 0-5 V direct
                        current_str = "  --  "
                        pos         = voltage / ADC_VREF
                        type_str    = "0-5 V"
                    else:
                        # Capteur 4-20 mA via shunt 241 Ω
                        current     = voltage / ADC_R_SHUNT
                        current_str = f"{current * 1000:6.2f}"
                        pos         = (current - ADC_I_MIN) / (ADC_I_MAX - ADC_I_MIN)
                        type_str    = "4-20 mA"

                    muscle = MUSCLE_OF_ADC_CH[ch]
                    print(f"{ch:>2} | {raw:>5d} | {voltage:6.3f} | {current_str} | "
                          f"{pos:>8.3f} | {muscle:>6d} | {type_str}")

                    row += [raw, f"{voltage:.6f}", f"{pos:.6f}"]

                print("\n(raw 0..4095 ; pos_norm devrait être ∈ [0, 1] si capteur OK)")

                writer.writerow(row)
                fcsv.flush()

                time.sleep(0.2)   # ~5 Hz

        except KeyboardInterrupt:
            print(f"\nArrêt. Log sauvegardé dans {csv_name}")
