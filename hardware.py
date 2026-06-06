#!/usr/bin/env python3
"""
hardware.py — Ouverture et init des bus I2C
=============================================
Bus 1 : ADC positions (0x1D), ADC pressions (0x1F), DAC vannes.
Bus 2 : IMU. L'ADC pression est tolérant à l'échec (pressure_adc_ok=False).
"""
from __future__ import annotations

from smbus2 import SMBus

from imu import imu_init
from position_adc import init_adc
from pressure_adc import init_pressure_adc


class HardwareBuses:
    """Tient bus_imu (bus 2) et bus_adc (bus 1) ouverts pendant la session.
    pressure_adc_ok=False si l'init ADC pression échoue (P_mes désactivé)."""

    def __init__(self, imu_bus_num: int = 2, ctrl_bus_num: int = 1):
        self.bus_imu = SMBus(imu_bus_num)
        self.bus_adc = SMBus(ctrl_bus_num)

        imu_init(self.bus_imu)
        init_adc(self.bus_adc)

        try:
            init_pressure_adc(self.bus_adc)
            self.pressure_adc_ok = True
            print("[OK] ADC pression initialisé (0x1F)")
        except Exception as e:
            self.pressure_adc_ok = False
            print(f"[WARN] ADC pression KO : {e} — affichage P_mes désactivé")

        print(f"[OK] Bus I2C ouverts — IMU (bus {imu_bus_num}), "
              f"ADC/DAC (bus {ctrl_bus_num})")

    def close(self) -> None:
        for bus in (self.bus_imu, self.bus_adc):
            try:
                bus.close()
            except Exception:
                pass
