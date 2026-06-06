#!/usr/bin/env python3
"""
dac.py — DAC5578 (8 canaux, 8 bits) sur Raspberry Pi
=====================================================
Bus I2C 1, adresse 0x48, VREF externe 5.0 V. Pilote des vannes
proportionnelles 0-6 bar via conversion pression linéaire.
"""

import smbus2


DAC_ADDR     = 0x48   # Adresse I2C du DAC5578
DAC_VREF     = 5.0    # V (à vérifier au multimètre sur pin VREF)
DAC_MAX_CODE = 255    # 8 bits
P_MAX        = 6.0    # bar


def set_voltage(bus: smbus2.SMBus, channel: int, voltage: float) -> None:
    """Configure une tension sur le canal DAC (0 = A, 7 = H)."""
    if not (0 <= channel <= 7):
        raise ValueError(f"Canal doit être entre 0 et 7, reçu : {channel}")
    voltage = max(0.0, min(voltage, DAC_VREF))
    code = int((voltage / DAC_VREF) * DAC_MAX_CODE + 0.5)
    cmd = 0b00110000 | (channel & 0x07)
    msb = code & 0xFF
    lsb = 0x00          # Bits inutilisés (DAC 8 bits)
    bus.write_i2c_block_data(DAC_ADDR, cmd, [msb, lsb])


def set_muscle_pressure(bus: smbus2.SMBus, muscle: int, pressure: float) -> None:
    """Commande un muscle pneumatique en pression (0..P_MAX bar)."""
    if not (0 <= muscle <= 5):
        raise ValueError(f"Muscle doit être entre 0 et 5, reçu : {muscle}")
    voltage = (pressure / P_MAX) * DAC_VREF
    set_voltage(bus, channel=muscle, voltage=voltage)


def clear_all(bus: smbus2.SMBus) -> None:
    """Remet tous les canaux à 0 V via le registre CLR (0b01010000)."""
    bus.write_i2c_block_data(DAC_ADDR, 0b01010000, [0xFF, 0xFF])


# ═════════════════════════════════════════════════════════════════════════════
# Test principal
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Test DAC5578 @ 0x{DAC_ADDR:02X} — VREF={DAC_VREF} V, P_MAX={P_MAX} bar\n")
    with smbus2.SMBus(1) as bus:
        for i, v in enumerate([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]):
            set_voltage(bus, channel=i, voltage=v)
            print(f"  Canal {i} → {v:.2f} V")
        for i, p in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
            set_muscle_pressure(bus, muscle=i, pressure=p)
            print(f"  Muscle {i} → {p:.2f} bar")
        clear_all(bus)
        print("\nTous les canaux → 0 V")
