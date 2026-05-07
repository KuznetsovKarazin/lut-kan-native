"""
exp_real_world_soc.py — LUT-KAN vs polynomial on battery SOC estimation.

Battery State of Charge (SOC) estimation from Open Circuit Voltage (OCV)
is a canonical embedded systems problem:

  - Input:  OCV (V), normalised to [-1, 1]
  - Output: SOC (0–1)
  - The OCV-SOC curve is monotonic but has a nonlinear middle plateau region
    (lithium-ion chemistry) — neither smooth nor sharply discontinuous.
  - Typical MCU application: BMS (Battery Management System) on Cortex-M4.

Dataset: synthetic OCV-SOC curve based on published Li-ion cell parameters
(Plett 2015, "Battery Management Systems Vol. 1"), plus Gaussian noise
mimicking real ADC measurements.

Comparison:
  1. Chebyshev LS (optimal degree, server-side, closed-form)
  2. B-spline cubic LS (optimal knots, server-side, closest to KAN)
  3. LUT K=4,L=8 with poly-init (quantize from poly, no gradient — baseline)
  4. LUT K=4,L=8 direct gradient training (our method)

Metrics:
  - Test MSE and MAE in SOC units (% error)
  - Inference cycles estimate (Cortex-M4, no FPU)
  - Model size (bytes, uint8 deployment)

Usage:
  python scripts/exp_real_world_soc.py
  python scripts/exp_real_world_soc.py --n_train 50   # tight calibration budget
  python scripts/exp_real_world_soc.py --n_train 200  # richer data
  python scripts/exp_real_world_soc.py --noise 0.005  # lower noise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import make_lsq_spline

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lut_native.baselines import (
    fit_chebyshev_ls,
    eval_chebyshev,
    sample_polynomial_to_lut,
    quantize_lut_uint8_asym,
    dequantize_lut,
)
from lut_native.core import lut_forward_numpy
from lut_native.training import TrainConfig, train_lut_edge


# ─────────────────────────────────────────────────────────────────────────────
# OCV-SOC curve (Li-ion, based on Plett 2015 parametric model)
# ─────────────────────────────────────────────────────────────────────────────

def ocv_soc_curve(soc: np.ndarray) -> np.ndarray:
    """
    OCV as a function of SOC for a typical LiFePO4 cell.
    Based on polynomial fit to published discharge data.
    OCV range: ~3.0V (SOC=0) to ~3.6V (SOC=1).
    The characteristic plateau at SOC=0.3–0.7 makes this non-trivially nonlinear.
    """
    # Published coefficients for LiFePO4 OCV model (Plett 2015)
    a = np.array([-1.031, 3.685, -1.432, 0.888, -0.2918, 0.002, 0.1088])
    ocv = (a[0] * np.exp(-35 * soc)
           + a[1]
           + a[2] * soc
           + a[3] * soc**2
           - a[4] * soc**3
           + a[5] / (soc + 0.01)
           + a[6] * np.exp(-4 * (soc - 1)))
    return ocv


def generate_soc_data(
    n_train: int,
    n_val: int,
    n_test: int,
    noise_std: float,
    seed: int = 42,
) -> tuple:
    """
    Generate OCV → SOC mapping data with measurement noise.

    x = OCV (normalised to [-1, 1])
    y = SOC (0 to 1)
    """
    rng = np.random.default_rng(seed)

    # True OCV range
    soc_min, soc_max = 0.02, 0.98
    ocv_min = ocv_soc_curve(np.array([soc_max]))[0]   # ~3.42V
    ocv_max = ocv_soc_curve(np.array([soc_min]))[0]   # ~3.55V

    def make_split(n, seed_offset):
        soc = rng.uniform(soc_min, soc_max, n)
        ocv = ocv_soc_curve(soc)
        ocv_noisy = ocv + rng.normal(0, noise_std, n)
        # Normalise OCV to [-1, 1]
        x = 2 * (ocv_noisy - ocv_min) / (ocv_max - ocv_min) - 1
        x = np.clip(x, -1, 1).astype(np.float32)
        y = soc.astype(np.float32)
        return x, y

    x_tr, y_tr = make_split(n_train, 0)
    x_val, y_val = make_split(n_val, 1)
    x_te, y_te = make_split(n_test, 2)
    return x_tr, y_tr, x_val, y_val, x_te, y_te, ocv_min, ocv_max


# ─────────────────────────────────────────────────────────────────────────────
# Inference cost estimates (Cortex-M4 without FPU)
# ─────────────────────────────────────────────────────────────────────────────

FLOAT_MUL = 15
FLOAT_ADD = 12
INT_OP    = 1

def lut_cycles(K: int, L: int) -> int:
    return 4 * INT_OP + FLOAT_MUL + FLOAT_ADD   # 31, constant

def cheby_cycles(degree: int) -> int:
    recurrence = (degree - 1) * (2 * FLOAT_MUL + FLOAT_ADD)
    dot = (degree + 1) * (FLOAT_MUL + FLOAT_ADD)
    return recurrence + dot

def bspline_cycles(n_inner: int, order: int = 3) -> int:
    knot_search = int(np.log2(max(n_inner + 2, 2))) * INT_OP
    deboor = order * (order + 1) // 2 * (2 * FLOAT_MUL + FLOAT_ADD)
    return knot_search + deboor


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run(n_train: int, noise_std: float, K: int, L: int, lut_epochs: int, n_seeds: int):
    n_val = min(n_train, max(20, n_train // 3))
    n_test = 400

    print(f"\n{'━'*65}")
    print(f"  Battery SOC estimation — LUT-KAN vs polynomial baselines")
    print(f"  n_train={n_train}  noise_std={noise_std:.4f}  K={K},L={L}")
    print(f"  Deployment: Cortex-M4 no FPU  |  Seeds: {n_seeds}")
    print(f"{'━'*65}\n")

    seed_results = {
        "lut_poly_init": [], "lut_gd": [],
        "cheby_ls": [], "bspline_ls": [],
    }

    for seed in range(n_seeds):
        d = generate_soc_data(n_train, n_val, n_test, noise_std, seed=seed)
        x_tr, y_tr, x_val, y_val, x_te, y_te, ocv_min, ocv_max = d

        # Sort for B-spline (requires non-decreasing x)
        idx_s = np.argsort(x_tr)
        x_s, y_s = x_tr[idx_s], y_tr[idx_s]

        # ── 1. Chebyshev LS (optimal degree via cross-validation) ────────────
        best_cheby_mse, best_deg, best_coeffs = 1e9, 0, None
        for deg in range(2, min(K * L, 50)):
            c = fit_chebyshev_ls(x_tr, y_tr, degree=deg)
            mse_v = float(np.mean((eval_chebyshev(x_val, c) - y_val) ** 2))
            if mse_v < best_cheby_mse:
                best_cheby_mse = mse_v
                best_deg = deg
                best_coeffs = c
        cheby_mse_te = float(np.mean((eval_chebyshev(x_te, best_coeffs) - y_te) ** 2))
        seed_results["cheby_ls"].append((cheby_mse_te, best_deg))

        # ── 2. B-spline cubic LS (cross-val n_inner, capped to n_train//8) ─
        bspline_mse_te = 1e9
        best_n_inner = 4
        max_ni = max(2, min(K * L - 4, n_train // 8))
        for ni in range(2, max_ni + 1):
            try:
                t = np.r_[[-1] * 4, np.linspace(-0.95, 0.95, ni), [1] * 4]
                spl = make_lsq_spline(x_s, y_s, t=t, k=3)
                mse_v = float(np.mean((spl(x_val) - y_val) ** 2))
                mse_te = float(np.mean((spl(x_te) - y_te) ** 2))
                if mse_v < bspline_mse_te:
                    bspline_mse_te = mse_te
                    best_n_inner = ni
            except Exception:
                pass
        seed_results["bspline_ls"].append((bspline_mse_te, best_n_inner))

        # ── 3. LUT: poly-init, no gradient ───────────────────────────────────
        lut_init = sample_polynomial_to_lut(best_coeffs, K=K, L=L)
        lut_pi_mse = float(np.mean((lut_forward_numpy(x_te, lut_init) - y_te) ** 2))
        seed_results["lut_poly_init"].append(lut_pi_mse)

        # ── 4. LUT: direct gradient training ─────────────────────────────────
        cfg = TrainConfig(epochs=lut_epochs, seed=seed, lambda_2=1.0, lr=1e-2)
        r = train_lut_edge(
            lut_init, x_tr, y_tr, x_val, y_val, x_te, y_te, -1.0, 1.0, cfg
        )
        seed_results["lut_gd"].append((r.mse_test_at_best, r.best_epoch))

    # ── Aggregate and report ─────────────────────────────────────────────────
    def _soc_err(mse: float) -> float:
        return float(np.sqrt(mse) * 100)   # RMSE in SOC% (SOC is 0–1)

    cheby_mses  = [r[0] for r in seed_results["cheby_ls"]]
    bspline_mses = [r[0] for r in seed_results["bspline_ls"]]
    lut_pi_mses = seed_results["lut_poly_init"]
    lut_gd_mses = [r[0] for r in seed_results["lut_gd"]]
    best_eps    = [r[1] for r in seed_results["lut_gd"]]

    best_cheby_deg    = int(np.median([r[1] for r in seed_results["cheby_ls"]]))
    best_bspline_ni   = int(np.median([r[1] for r in seed_results["bspline_ls"]]))

    cy_cheby   = cheby_cycles(best_cheby_deg)
    cy_bspline = bspline_cycles(best_bspline_ni)
    cy_lut     = lut_cycles(K, L)

    params_cheby   = best_cheby_deg + 1
    params_bspline = best_bspline_ni + 4
    params_lut     = K * L
    mem_cheby      = params_cheby * 4      # float32
    mem_bspline    = params_bspline * 4    # float32
    mem_lut        = K * L + 4 * K        # uint8 table + float32 knots

    print(f"  {'Method':28s} {'RMSE%':>7s} ±{'':>5s}  {'params':>7s}  {'bytes':>7s}  {'cycles':>7s}")
    print("  " + "─" * 65)

    def _row(name, mses, params, mem_b, cycles, extra=""):
        rmse_mean = _soc_err(float(np.mean(mses)))
        rmse_std  = _soc_err(float(np.std(mses))) if len(mses) > 1 else 0
        print(f"  {name:28s} {rmse_mean:>6.3f}% ±{rmse_std:<5.3f}  "
              f"{params:>7d}  {mem_b:>7d}  {cycles:>7d}  {extra}")

    _row(f"Chebyshev LS d={best_cheby_deg}",
         cheby_mses, params_cheby, mem_cheby, cy_cheby, "(server only)")
    _row(f"B-spline cubic n={best_bspline_ni}",
         bspline_mses, params_bspline, mem_bspline, cy_bspline, "(server only)")
    _row(f"LUT K={K},L={L} poly-init",
         lut_pi_mses, params_lut, mem_lut, cy_lut, "(no grad)")
    _row(f"LUT K={K},L={L} direct GD",
         lut_gd_mses, params_lut, mem_lut, cy_lut,
         f"best_ep≈{int(np.mean(best_eps))}")

    print()
    print("  Notes:")
    print(f"  - RMSE% is root-mean-squared error in SOC percentage points")
    print(f"  - Cycles estimated for Cortex-M4 without FPU (soft-float)")
    print(f"  - LUT memory: {K*L} bytes uint8 + {4*K} bytes float32 knots")
    print(f"  - LUT inference: {cy_lut} cycles = {cy_lut/64e6*1e6:.1f} µs at 64 MHz")
    print(f"  - B-spline inference: {cy_bspline} cycles = {cy_bspline/64e6*1e6:.1f} µs at 64 MHz")

    lut_vs_cheby = float(np.mean(cheby_mses)) / max(float(np.mean(lut_gd_mses)), 1e-30)
    lut_vs_bspline = float(np.mean(bspline_mses)) / max(float(np.mean(lut_gd_mses)), 1e-30)
    print()
    print(f"  LUT accuracy vs Chebyshev LS: {lut_vs_cheby:.1f}× "
          + ("(LUT better)" if lut_vs_cheby > 1 else "(Cheby better)"))
    print(f"  LUT accuracy vs B-spline LS:  {lut_vs_bspline:.1f}× "
          + ("(LUT better)" if lut_vs_bspline > 1 else "(B-spline better)"))
    print(f"  LUT speed vs Chebyshev:  {cy_cheby/cy_lut:.0f}× fewer cycles")
    print(f"  LUT speed vs B-spline:   {cy_bspline/cy_lut:.0f}× fewer cycles")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Battery SOC estimation: LUT-KAN vs poly")
    p.add_argument("--n_train",  type=int,   default=100,  help="Training samples (default 100)")
    p.add_argument("--noise",    type=float, default=0.002, help="OCV noise std in V (default 0.002)")
    p.add_argument("--K",        type=int,   default=4,    help="LUT segments (default 4)")
    p.add_argument("--L",        type=int,   default=16,   help="LUT cells/segment (default 16)")
    p.add_argument("--epochs",   type=int,   default=1500, help="LUT GD epochs (default 1500)")
    p.add_argument("--seeds",    type=int,   default=5,    help="Seeds (default 5)")
    args = p.parse_args()
    run(
        n_train=args.n_train,
        noise_std=args.noise,
        K=args.K, L=args.L,
        lut_epochs=args.epochs,
        n_seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
