#!/usr/bin/env python3
"""
platform_pose.py
================
Structures de données IMU minimales (Vec3, IMUSnap) partagées par le logging
et le contrôle.

L'IMU n'est plus utilisé pour reconstruire la pose : la cinématique directe
sur les longueurs de muscles le remplace. Les snapshots restent loggués pour
diagnostic.
"""
from __future__ import annotations


class Vec3:
    __slots__ = ("x", "y", "z")
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x, self.y, self.z = x, y, z


class IMUSnap:
    """Snapshot d'une lecture IMU brute ou corrigée (offsets retirés)."""
    __slots__ = ("accel", "gyro")
    def __init__(self, accel: Vec3, gyro: Vec3):
        self.accel, self.gyro = accel, gyro
