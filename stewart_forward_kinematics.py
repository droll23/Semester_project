#!/usr/bin/env python3
"""
stewart_forward_kinematics.py — FK Stewart par Levenberg-Marquardt
==================================================================
Donné les 6 longueurs muscle mesurées, retrouve la pose (translation +
orientation). Résolution numérique sur
les 6 équations de longueur :

    r_i(q) = ‖R(θ)·p_i + t − b_i‖ − L_mesuré_i = 0,  q = (tx,ty,tz,rx,ry,rz)

À chaque itération : (JᵀJ + λ·diag(JᵀJ))·Δq = Jᵀr. L'amortissement
diag(JᵀJ) (vs λ·I) rend la pénalisation invariante au changement d'échelle
entre translations et rotations.

Warm-start via initial_translation_m / initial_rotation_rad
(convergence en 1-3 itérations en passant la pose précédente).
NumPy seulement.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Géométrie (copiée de robot_control_gui_*.py — gardée identique pour cohérence)
# ─────────────────────────────────────────────────────────────────────────────
STEWART_PLATFORM_POINTS_MM = np.array([
    [-533.56,  437.96, 0.00],   # A → muscle 0
    [-646.06,  243.11, 0.00],   # B → muscle 1
    [-112.50, -681.05, 0.00],   # C → muscle 2
    [ 112.50, -681.05, 0.00],   # D → muscle 3
    [ 646.06,  243.11, 0.00],   # E → muscle 4
    [ 533.56,  437.96, 0.00],   # F → muscle 5
])

STEWART_BASE_POINTS_MM = np.array([
    [-160.56,  1199.07, 951.00],   # point 0 → muscle 0
    [-1125.62, -457.86, 949.62],   # point 1 → muscle 1
    [-955.07,  -743.91, 953.47],   # point 2 → muscle 2
    [ 953.53,  -737.09, 955.13],   # point 3 → muscle 3
    [ 1121.72, -455.12, 953.51],   # point 4 → muscle 4
    [ 160.60,  1199.41, 950.76],   # point 5 → muscle 5
])

STEWART_PLATFORM_POINTS_M = STEWART_PLATFORM_POINTS_MM / 1000.0
STEWART_BASE_POINTS_M     = STEWART_BASE_POINTS_MM     / 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _euler_rotation_matrix(rotation_rad: Iterable[float]) -> np.ndarray:
    """R(rx, ry, rz) = Rz(rz) · Ry(ry) · Rx(rx) — convention GUI."""
    rx, ry, rz = rotation_rad
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array([
        [cy * cz, -cx * sz + sx * sy * cz,  sx * sz + cx * sy * cz],
        [cy * sz,  cx * cz + sx * sy * sz, -sx * cz + cx * sy * sz],
        [-sy,      sx * cy,                 cx * cy],
    ])


def stewart_inverse_kinematics(translation_m, rotation_rad,
                               base_points=STEWART_BASE_POINTS_M,
                               platform_points=STEWART_PLATFORM_POINTS_M):
    """6 longueurs de muscle (m) : L_i = || R(θ) · p_i + t − b_i ||."""
    R = _euler_rotation_matrix(rotation_rad)
    t = np.asarray(translation_m, dtype=float)
    out = np.empty(6, dtype=float)
    for i in range(6):
        p_world = R @ platform_points[i] + t
        out[i] = np.linalg.norm(p_world - base_points[i])
    return out


def _length_jacobian(q: np.ndarray,
                     base_points: np.ndarray,
                     platform_points: np.ndarray,
                     eps_t: float = 1e-7,
                     eps_r: float = 1e-7) -> np.ndarray:
    """Jacobien 6×6 par différences finies centrées."""
    J = np.empty((6, 6), dtype=float)
    eps_vec = np.array([eps_t, eps_t, eps_t, eps_r, eps_r, eps_r])
    for k in range(6):
        e = eps_vec[k]
        qp = q.copy(); qp[k] += e
        qm = q.copy(); qm[k] -= e
        Lp = stewart_inverse_kinematics(qp[0:3], qp[3:6], base_points, platform_points)
        Lm = stewart_inverse_kinematics(qm[0:3], qm[3:6], base_points, platform_points)
        J[:, k] = (Lp - Lm) / (2.0 * e)
    return J


# ─────────────────────────────────────────────────────────────────────────────
# Forward kinematics — Levenberg-Marquardt « Marquardt 1963 / Grishin 2023 »
# ─────────────────────────────────────────────────────────────────────────────
def stewart_forward_kinematics(
        measured_lengths_m: Iterable[float],
        initial_translation_m: Iterable[float] = (0.0, 0.0, 0.0),
        initial_rotation_rad:  Iterable[float] = (0.0, 0.0, 0.0),
        base_points: np.ndarray = STEWART_BASE_POINTS_M,
        platform_points: np.ndarray = STEWART_PLATFORM_POINTS_M,
        max_iters: int = 30,
        tol_residual_m: float = 1e-7,
        tol_step: float = 1e-9,
        damping: float = 1e-3,
):
    """Pose (t, r) qui produit measured_lengths_m, via Marquardt 1963 :
    (JᵀJ + λ·diag(JᵀJ))·Δq = Jᵀr. λ × 4 si divergence, × 0.5 sinon.

    Retourne (t (3,), r (3,), info) avec info = {converged, iterations,
    residual_norm, max_residual, final_lambda}.
    """
    L_meas = np.asarray(measured_lengths_m, dtype=float)
    if L_meas.shape != (6,):
        raise ValueError(
            f"measured_lengths_m doit contenir 6 valeurs, reçu shape={L_meas.shape}"
        )

    q = np.array(list(initial_translation_m) + list(initial_rotation_rad),
                 dtype=float)
    if q.shape != (6,):
        raise ValueError("initial_translation_m + initial_rotation_rad doit faire 6 valeurs")

    lam = float(damping)
    converged = False
    iters_used = 0

    L = stewart_inverse_kinematics(q[0:3], q[3:6], base_points, platform_points)
    r = L - L_meas
    rn = float(np.linalg.norm(r))

    for it in range(max_iters):
        iters_used = it
        if rn < tol_residual_m:
            converged = True
            break

        J = _length_jacobian(q, base_points, platform_points)
        JTJ = J.T @ J
        JTr = J.T @ r

        # (JᵀJ + λ · diag(JᵀJ)) Δq = Jᵀ r
        diag_JTJ = np.diag(np.diag(JTJ))
        diag_floor = max(np.max(np.diag(JTJ)) * 1e-12, 1e-20)
        diag_JTJ = np.where(diag_JTJ > diag_floor, diag_JTJ,
                            diag_floor * np.eye(6))

        try:
            delta = np.linalg.solve(JTJ + lam * diag_JTJ, JTr)
        except np.linalg.LinAlgError:
            lam = max(lam * 100.0, 1e-3)
            try:
                delta = np.linalg.solve(JTJ + lam * diag_JTJ, JTr)
            except np.linalg.LinAlgError:
                break

        q_new = q - delta
        L_new = stewart_inverse_kinematics(q_new[0:3], q_new[3:6],
                                           base_points, platform_points)
        r_new = L_new - L_meas
        rn_new = float(np.linalg.norm(r_new))

        if rn_new < rn:
            q = q_new
            r = r_new
            rn = rn_new
            lam = max(lam * 0.5, 1e-12)
        else:
            lam *= 4.0
            if lam > 1e8:
                iters_used = it + 1
                break

        if float(np.linalg.norm(delta)) < tol_step:
            converged = True
            iters_used = it + 1
            break
    else:
        iters_used = max_iters

    return (
        q[0:3].copy(),
        q[3:6].copy(),
        {
            'converged'     : converged,
            'iterations'    : iters_used,
            'residual_norm' : rn,
            'max_residual'  : float(np.max(np.abs(r))),
            'final_lambda'  : lam,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test : roundtrip IK → FK doit retomber sur la pose d'origine
# ─────────────────────────────────────────────────────────────────────────────
def _selftest(verbose: bool = True) -> bool:
    rng = np.random.default_rng(42)
    test_poses = [
        ("repos",          0.0,  0.0,   0.0,    0.0,    0.0,    0.0),
        ("Z +5 cm",        0.0,  0.0,  +0.05,   0.0,    0.0,    0.0),
        ("Z -5 cm",        0.0,  0.0,  -0.05,   0.0,    0.0,    0.0),
        ("X +3 cm",       +0.03, 0.0,  0.0,    0.0,    0.0,    0.0),
        ("Y +3 cm",        0.0, +0.03, 0.0,    0.0,    0.0,    0.0),
        ("roll +5",        0.0,  0.0,  0.0,   +5.0,    0.0,    0.0),
        ("pitch -5",       0.0,  0.0,  0.0,    0.0,   -5.0,    0.0),
        ("yaw +10",        0.0,  0.0,  0.0,    0.0,    0.0,  +10.0),
        ("combo",         +0.02,-0.01,+0.03,  +3.0,   -2.0,   +5.0),
    ]
    for k in range(5):
        test_poses.append((
            f"random{k}",
            float(rng.uniform(-0.04, 0.04)),
            float(rng.uniform(-0.04, 0.04)),
            float(rng.uniform(-0.04, 0.04)),
            float(rng.uniform(-7.0, 7.0)),
            float(rng.uniform(-7.0, 7.0)),
            float(rng.uniform(-10.0, 10.0)),
        ))

    if verbose:
        print(f"{'pose':12s}  {'iters':>5s}  {'res(um)':>9s}  "
              f"{'err_t(um)':>10s}  {'err_r(udeg)':>12s}  status")
        print("-" * 70)

    last_t, last_r = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    all_ok = True
    for (label, tx, ty, tz, rx_d, ry_d, rz_d) in test_poses:
        t_true = (tx, ty, tz)
        r_true = (math.radians(rx_d), math.radians(ry_d), math.radians(rz_d))

        L = stewart_inverse_kinematics(t_true, r_true)
        t_est, r_est, info = stewart_forward_kinematics(
            L,
            initial_translation_m=last_t,
            initial_rotation_rad=last_r,
        )

        err_t = np.linalg.norm(np.array(t_true) - t_est)
        err_r = np.linalg.norm(np.array(r_true) - r_est)
        ok = (info['converged']
              and info['residual_norm'] < 1e-6
              and err_t < 1e-5
              and err_r < 1e-5)
        if not ok:
            all_ok = False

        if verbose:
            print(f"{label:12s}  {info['iterations']:5d}  "
                  f"{info['residual_norm']*1e6:9.3f}  "
                  f"{err_t*1e6:10.2f}  {math.degrees(err_r)*1e6:12.2f}  "
                  f"{'OK' if ok else 'FAIL'}")

        last_t, last_r = tuple(t_est), tuple(r_est)

    if verbose:
        print("-" * 70)
        print("RESULT:", "ALL OK" if all_ok else "FAILURES")
    return all_ok


if __name__ == "__main__":
    _selftest(verbose=True)
