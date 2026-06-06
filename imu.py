#!/usr/bin/env python3
"""
imu.py — LSM9DS1 (accéléromètre + gyroscope) sur Raspberry Pi
==============================================================
Bus I2C 2, adresse AG 0x6B. Accel + gyro à 952 Hz ODR, lecture burst
22 bytes. Sorties : m/s² (accel), dps (gyro).
"""

import time
from dataclasses import dataclass
from smbus2 import SMBus


# ═════════════════════════════════════════════════════════════════════════════
# Configuration I2C
# ═════════════════════════════════════════════════════════════════════════════
I2C_BUS      = 2     # Bus I2C → /dev/i2c-2
IMU_I2C_ADDR = 0x6B  # Adresse LSM9DS1 AG (SDO_AG = VDD)


# ═════════════════════════════════════════════════════════════════════════════
# Registres AG (accéléromètre/gyroscope)
# ═════════════════════════════════════════════════════════════════════════════
REG_CTRL_REG8    = 0x22  # Config générale (bit 2: auto-increment)
REG_CTRL_REG1_G  = 0x10  # Gyroscope ODR et pleine échelle
REG_CTRL_REG6_XL = 0x20  # Accéléromètre ODR et pleine échelle
REG_STATUS       = 0x17  # Drapeaux data-ready
REG_BURST_START  = 0x18  # Début de lecture burst (OUT_X_L_G)
BURST_LENGTH     = 22    # Bytes de 0x18 à 0x2D (gyro + accel)

# Offsets dans le buffer de 22 bytes
GYRO_OFFSET  = 0   # data[0]  = OUT_X_L_G
ACCEL_OFFSET = 16  # data[16] = OUT_X_L_XL (0x28 - 0x18 = 16)


# ═════════════════════════════════════════════════════════════════════════════
# Sensibilités et constantes de conversion
# ═════════════════════════════════════════════════════════════════════════════
ACCEL_SENS_2G    = 0.000061  # g par LSB (±2g pleine échelle)
G_TO_MS2         = 9.80665   # m/s² par g
GYRO_SENS_245DPS = 0.00875   # dps par LSB (±245 dps pleine échelle)


# ═════════════════════════════════════════════════════════════════════════════
# Structures de données
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Vec3:
    """Vecteur 3 axes."""
    x: float
    y: float
    z: float

    def __repr__(self) -> str:
        return f"Vec3(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"


@dataclass
class IMUData:
    """Mesure combinée accéléromètre + gyroscope."""
    accel: Vec3   # Accélération linéaire en m/s²
    gyro:  Vec3   # Vitesse angulaire en degrés par seconde (dps)


# ═════════════════════════════════════════════════════════════════════════════
# Utilitaires de conversion
# ═════════════════════════════════════════════════════════════════════════════
def to_s16(low_byte: int, high_byte: int) -> int:
    """
    Convertit un low byte et un high byte en entier signé 16 bits.
    LSM9DS1 utilise little-endian: LSB d'abord, MSB ensuite.
    """
    unsigned = (high_byte << 8) | low_byte
    return unsigned if unsigned < 0x8000 else unsigned - 0x10000


# ═════════════════════════════════════════════════════════════════════════════
# Initialisation
# ═════════════════════════════════════════════════════════════════════════════
def imu_init(bus: SMBus) -> None:
    """Init LSM9DS1 : auto-increment + gyro 952 Hz/±245 dps + accel 952 Hz/±2 g."""
    bus.write_byte_data(IMU_I2C_ADDR, REG_CTRL_REG8,    0x04)   # auto-increment
    bus.write_byte_data(IMU_I2C_ADDR, REG_CTRL_REG1_G,  0xC0)   # gyro
    bus.write_byte_data(IMU_I2C_ADDR, REG_CTRL_REG6_XL, 0xE0)   # accel
    time.sleep(0.05)


# ═════════════════════════════════════════════════════════════════════════════
# Vérification data-ready
# ═════════════════════════════════════════════════════════════════════════════
def data_ready(bus: SMBus) -> bool:
    """True si gyro ET accel ont des données fraîches (STATUS_REG bits 0,1)."""
    return (bus.read_byte_data(IMU_I2C_ADDR, REG_STATUS) & 0x03) == 0x03


# ═════════════════════════════════════════════════════════════════════════════
# Lecture atomique burst (gyro + accel)
# ═════════════════════════════════════════════════════════════════════════════
def read_imu(bus: SMBus) -> IMUData:
    """Lit gyro + accel en un burst I2C de 22 bytes (0x18..0x2D).

    Layout : bytes 0-5 = gyro, 6-15 ignorés, 16-21 = accel.
    Retourne IMUData(accel m/s², gyro dps).
    """
    data = bus.read_i2c_block_data(IMU_I2C_ADDR, REG_BURST_START, BURST_LENGTH)

    # ── Gyroscope (bytes 0–5) ────────────────────────────────────────────────
    o = GYRO_OFFSET
    gx = to_s16(data[o + 0], data[o + 1]) * GYRO_SENS_245DPS
    gy = to_s16(data[o + 2], data[o + 3]) * GYRO_SENS_245DPS
    gz = to_s16(data[o + 4], data[o + 5]) * GYRO_SENS_245DPS

    # ── Accéléromètre (bytes 16–21) ──────────────────────────────────────────
    o = ACCEL_OFFSET
    scale = ACCEL_SENS_2G * G_TO_MS2
    ax = to_s16(data[o + 0], data[o + 1]) * scale
    ay = to_s16(data[o + 2], data[o + 3]) * scale
    az = to_s16(data[o + 4], data[o + 5]) * scale

    return IMUData(accel=Vec3(ax, ay, az), gyro=Vec3(gx, gy, gz))


# ═════════════════════════════════════════════════════════════════════════════
# Test principal
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Ouverture /dev/i2c-{I2C_BUS} @ 0x{IMU_I2C_ADDR:02X} ...")
    
    try:
        with SMBus(I2C_BUS) as bus:
            imu_init(bus)
            print("IMU initialisé — 952 Hz ODR, lecture burst activée.")
            print("Appuyez sur Ctrl+C pour arrêter.\n")

            count = 0
            t_start = time.monotonic()

            while True:
                while not data_ready(bus):
                    pass

                d = read_imu(bus)
                count += 1

                print(
                    f"[{count:6d}] "
                    f"Accel  ax={d.accel.x:8.3f}  ay={d.accel.y:8.3f}  az={d.accel.z:8.3f}  m/s²"
                    f"  |  "
                    f"Gyro   gx={d.gyro.x:8.3f}  gy={d.gyro.y:8.3f}  gz={d.gyro.z:8.3f}  dps"
                )

    except KeyboardInterrupt:
        elapsed = time.monotonic() - t_start
        hz = count / elapsed if elapsed > 0 else 0
        print(f"\nArrêt après {count} échantillons en {elapsed:.1f} s  ({hz:.0f} Hz).")
