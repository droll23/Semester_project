#!/usr/bin/env python3
"""
pressure_adc.py — Lecture pression des vannes (ADC128D818 @ 0x1F)
==================================================================
Vref externe 5 V. Capteur linéaire 1-5 V → 0-6 bar :
    P [bar] = (V - V_zero) / (V_ref - V_zero) * P_max

V_ZERO peut être affiné par canal (mesure à 0 bar absolu) pour une
précision sub-1 % ; la spec donne 1.0 V par défaut.
"""

import time
from smbus2 import SMBus

PRESSURE_ADC_ADDR = 0x1F
ADC_VREF          = 5.0     # V — Vref EXTERNE (cf. init_pressure_adc)
RAW_MAX           = 4095.0
P_MAX_BAR         = 9.0
ADC_REG_READINGS  = 0x20

ADC_REG_ADV_CONFIG       = 0x0B
ADC_ADV_CONFIG_EXT_VREF  = 0b00000001   # Vref externe ON, Mode 0

# Tensions à 0 bar par canal (calibration banc). Spec : 1.0 V exact.
V_ZERO = [0.9405, 0.9344, 0.9967, 0.9704, 0.9877, 0.9711]


from geometry import VALVE_OF_MUSCLE
# Muscle m → canal ADC = vanne reliée à ce muscle.
ADC_PRESSURE_CH_OF_MUSCLE = list(VALVE_OF_MUSCLE)


def init_pressure_adc(bus: SMBus) -> None:
    """Init ADC pression avec Vref externe 5 V. Sans, surévaluation ×1.95."""
    bus.write_byte_data(PRESSURE_ADC_ADDR, 0x00, 0x00)        # Stop
    bus.write_byte_data(PRESSURE_ADC_ADDR, ADC_REG_ADV_CONFIG,
                        ADC_ADV_CONFIG_EXT_VREF)
    bus.write_byte_data(PRESSURE_ADC_ADDR, 0x03, 0xFF)        # Masquer IRQ
    bus.write_byte_data(PRESSURE_ADC_ADDR, 0x07, 0b00000001)  # Continu
    bus.write_byte_data(PRESSURE_ADC_ADDR, 0x08, 0b11000000)  # Désactiver CH6/7/temp
    bus.write_byte_data(PRESSURE_ADC_ADDR, 0x00, 0x01)        # Start
    time.sleep(0.1)


def _read_voltage(bus: SMBus, ch: int) -> float:
    data = bus.read_i2c_block_data(PRESSURE_ADC_ADDR, ADC_REG_READINGS + ch, 2)
    raw  = ((data[0] << 8) | data[1]) >> 4
    return (raw / RAW_MAX) * ADC_VREF


def _voltage_to_bar(v: float, ch: int) -> float:
    """V → bar : P = (V − V_zero[ch]) / (V_ref − V_zero[ch]) × P_max."""
    pressure = ((v - V_ZERO[ch]) / (ADC_VREF - V_ZERO[ch])) * P_MAX_BAR
    return max(0.0, min(P_MAX_BAR, pressure))


def read_pressures(bus: SMBus) -> list:
    """Retourne les pressions en bar indexées par muscle logique [0..5]."""
    by_ch = [_voltage_to_bar(_read_voltage(bus, ch), ch) for ch in range(6)]
    return [by_ch[ADC_PRESSURE_CH_OF_MUSCLE[m]] for m in range(6)]


if __name__ == "__main__":
    LABELS = ["A", "B", "C", "D", "E", "F"]
    bus = SMBus(1)
    init_pressure_adc(bus)
    print("Acquisition pression (Ctrl+C pour quitter)")
    print("Vref externe 5 V activée — capteur 1-5 V → 0-6 bar\n")
    print("  t(s)   " + "  ".join(f"M{m}_{LABELS[m]}(bar)" for m in range(6)))
    t0 = time.time()
    try:
        while True:
            p = read_pressures(bus)
            print(f"  {time.time()-t0:5.2f}   " + "  ".join(f"{v:>10.3f}" for v in p))
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()
