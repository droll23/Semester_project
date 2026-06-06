#!/usr/bin/env python3
"""
diagnose_position_noise.py — Bruit fil-pot (version minimale)
=============================================================
Plateforme IMMOBILE, FF/PID OFF. Acquiert pendant N secondes à la cadence
native ADC et sort, par muscle :
  - bruit RMS (mm)
  - présence de spikes (échantillons > 5σ) : oui / non

Usage : python3 diagnose_position_noise.py [--duration 30]
"""
from __future__ import annotations

import argparse
import time
from statistics import mean, pstdev

from smbus2 import SMBus

from position_adc import init_adc, read_positions, apply_position_correction


DEFAULT_DURATION_S = 30.0
POLL_INTERVAL_S    = 0.005


def acquire(duration_s: float) -> tuple:
    """Logue L_muscle pendant duration_s. Retourne (L_by_muscle, fs_eff)."""
    print(f"Acquisition {duration_s:.0f} s — plateforme IMMOBILE, FF/PID OFF.")
    L_log = [[] for _ in range(6)]

    with SMBus(1) as bus:
        init_adc(bus)
        time.sleep(0.3)

        # Warmup : offset au repos
        warmup_buf = []
        t_warmup = time.perf_counter() + 1.0
        while time.perf_counter() < t_warmup:
            try:
                warmup_buf.append(read_positions(bus))
            except Exception:
                pass
            time.sleep(0.05)
        if not warmup_buf:
            raise RuntimeError("Aucune lecture ADC pendant warmup.")
        raw_rest_offset = [
            sum(row[m] for row in warmup_buf) / len(warmup_buf)
            for m in range(6)
        ]

        # Acquisition avec déduplication
        t0 = time.perf_counter()
        prev_raw = None
        n_reads = 0
        while time.perf_counter() - t0 < duration_s:
            try:
                raw = read_positions(bus)
            except Exception:
                time.sleep(POLL_INTERVAL_S)
                continue
            if prev_raw is not None and all(raw[m] == prev_raw[m] for m in range(6)):
                time.sleep(POLL_INTERVAL_S)
                continue
            prev_raw = list(raw)
            L = apply_position_correction(raw, raw_rest_offset)
            for m in range(6):
                L_log[m].append(L[m])
            n_reads += 1

    fs_eff = n_reads / duration_s
    print(f"  {n_reads} échantillons, fs_eff = {fs_eff:.1f} Hz.\n")
    return L_log, fs_eff


def analyze(L_log: list) -> None:
    """Affiche RMS + présence de spikes par muscle."""
    print("─" * 50)
    print(f"{'Muscle':<8}{'RMS (mm)':<12}{'Spikes':<10}")
    print("─" * 50)
    for m in range(6):
        samples = L_log[m]
        if len(samples) < 4:
            print(f"M{m:<7}{'N/A':<12}{'N/A':<10}")
            continue
        mu = mean(samples)
        sigma = pstdev(samples)
        rms_mm = sigma * 1000.0
        if sigma > 1e-9:
            n_spikes = sum(1 for x in samples if abs(x - mu) > 5.0 * sigma)
        else:
            n_spikes = 0
        spike_str = f"OUI ({n_spikes})" if n_spikes > 0 else "non"
        print(f"M{m:<7}{rms_mm:<12.3f}{spike_str:<10}")
    print("─" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Bruit fil-pot : RMS + spikes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                        help="durée d'acquisition (s)")
    args = parser.parse_args()

    L_log, _ = acquire(args.duration)
    analyze(L_log)


if __name__ == "__main__":
    main()
