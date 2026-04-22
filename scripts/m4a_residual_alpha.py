#!/usr/bin/env python3
"""
M4a: Residual LUT α sweep.

Test the "trust region" hypothesis:
    Naive direct-LUT training destroys poly-init because Adam has too much
    freedom. Constraining the effective LUT to be init + α·Δ with α << 1
    should prevent this.

Isolated ablation:
  - alpha=0.0   -> "trust region of size zero" -> training has no effect
                   (tests sanity: MSE stays at init)
  - alpha=0.05  -> very tight
  - alpha=0.1   -> tight
  - alpha=0.3   -> loose
  - alpha=1.0   -> equivalent to unconstrained direct-LUT (= KAN2 baseline)

For each α, per-seed: init a ResidualLUTKAN2Layer from trained PolyKAN2
coefficients, train 200 epochs, report best-val and final-val MSE.

Same setup as M3 to keep numbers comparable:
  - composition_1d: [1→4→1], deg=12 poly, K=16, L=32
  - feynman_2d:     [2→4→1], deg=8 poly,  K=16, L=32
  - lr = 5e-4 (same for both layers in M4a; M4c separates them)
  - lambda_2 = 0 (no smoothness reg; M4b adds anchor penalty)
  - 3 seeds

What M4a does NOT do:
  - separate LRs per layer (M4c)
  - init-anchor penalty (M4b)
  - progressive unfreezing (M4d)
  - visited-region masking (later phase)

Output: results/M4a_residual_alpha/{composition_1d,feynman_2d}/{summary.json, plot.png}

Runtime: ~5 min CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lut_native import (  # noqa: E402
    ResidualLUTKAN2Layer,
    ResidualTrainConfig,
    generate_data_2d,
    sample_polynomial_to_lut,
    train_residual_kan2,
)
from exp_H4_kan2_2d import PolyKAN2, train_poly_kan2
from m2b_composition import generate_data_1d


def _init_residual_from_poly(
    poly_c1, poly_c2,
    in_dim: int, hidden_dim: int, out_dim: int,
    K: int, L: int, alpha: float,
) -> ResidualLUTKAN2Layer:
    model = ResidualLUTKAN2Layer(
        in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
        K=K, L=L, alpha=alpha,
    )
    init_l1 = np.zeros((in_dim, hidden_dim, K, L), dtype=np.float32)
    for i in range(in_dim):
        for h in range(hidden_dim):
            init_l1[i, h] = sample_polynomial_to_lut(poly_c1[i, h], K=K, L=L)
    init_l2 = np.zeros((hidden_dim, out_dim, K, L), dtype=np.float32)
    for h in range(hidden_dim):
        for o in range(out_dim):
            init_l2[h, o] = sample_polynomial_to_lut(poly_c2[h, o], K=K, L=L)
    model.init_layer1_from_arrays(init_l1)
    model.init_layer2_from_arrays(init_l2)
    return model


def run_one_alpha(
    poly_coeffs_per_seed, seeds,
    in_dim, hidden, K, L, alpha, lr, epochs,
    x_tr, y_tr, x_v, y_v, x_te, y_te,
):
    per_seed = []
    trace_for_plot = None   # keep seed=0 trace for plotting
    for idx, seed in enumerate(seeds):
        c1, c2 = poly_coeffs_per_seed[idx]
        model = _init_residual_from_poly(c1, c2, in_dim, hidden, 1, K, L, alpha)
        cfg = ResidualTrainConfig(
            lambda_1=0.0, lambda_2=0.0,
            lr_l1=lr, lr_l2=lr,
            epochs=epochs, batch_size=128, seed=seed,
            eval_every_epochs=10,
        )
        res = train_residual_kan2(model, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)
        per_seed.append({
            "seed": seed,
            "mse_val_init": res.mse_val_init,
            "mse_val_at_best": res.mse_val_at_best,
            "mse_test_at_best": res.mse_test_at_best,
            "mse_val_final": res.mse_val_final,
            "best_epoch": res.best_epoch,
            "delta_l1_l2_norm_final": float(np.linalg.norm(res.delta_l1_final)),
            "delta_l2_l2_norm_final": float(np.linalg.norm(res.delta_l2_final)),
        })
        if idx == 0:
            trace_for_plot = res.trace
    arr_best = np.array([p["mse_test_at_best"] for p in per_seed])
    arr_init = np.array([p["mse_val_init"] for p in per_seed])
    return {
        "alpha": alpha,
        "per_seed": per_seed,
        "test_mse_best_mean": float(arr_best.mean()),
        "test_mse_best_std": float(arr_best.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "val_mse_init_mean": float(arr_init.mean()),
        "improvement_over_init_mean": float(arr_init.mean() / arr_best.mean()),
        "trace_seed0": trace_for_plot,
    }


def task_pipeline(
    task_name, x_tr, y_tr, x_v, y_v, x_te, y_te,
    in_dim, hidden, K, L, degree, alpha_vals, seeds,
    lut_epochs, poly_epochs, out_dir,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\nTask: {task_name}\n{'='*72}")

    # 1. Train polynomial baseline per seed
    poly_coeffs = []
    poly_test_mses = []
    for seed in seeds:
        pres = train_poly_kan2(
            in_dim=in_dim, hidden_dim=hidden, out_dim=1, degree=degree,
            x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
            lr=5e-3, epochs=poly_epochs, batch_size=128, seed=seed,
        )
        poly_coeffs.append((pres["coeffs_l1"], pres["coeffs_l2"]))
        poly_test_mses.append(pres["test_mse"])
        print(f"  PolyKAN2 seed={seed}: {pres['test_mse']:.3e}")
    poly_mse_mean = float(np.mean(poly_test_mses))
    print(f"  PolyKAN2 mean: {poly_mse_mean:.3e}")

    # 2. Sweep alpha
    results = []
    print(f"\n  Residual LUT alpha sweep (lr=5e-4, {lut_epochs} epochs, {len(seeds)} seeds):")
    print(f"  {'alpha':>6} {'init MSE':>10} {'best MSE':>14} {'best_ep(s0)':>12} "
          f"{'vs Poly':>10} {'improv':>10}")
    for alpha in alpha_vals:
        t0 = time.time()
        r = run_one_alpha(
            poly_coeffs, seeds,
            in_dim=in_dim, hidden=hidden, K=K, L=L, alpha=alpha,
            lr=5e-4, epochs=lut_epochs,
            x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
        )
        results.append(r)
        vs_poly = poly_mse_mean / r["test_mse_best_mean"]
        improv = r["improvement_over_init_mean"]
        print(f"  {alpha:>6.2f} {r['val_mse_init_mean']:>10.2e} "
              f"{r['test_mse_best_mean']:>10.2e}±{r['test_mse_best_std']:>2.0e} "
              f"{r['per_seed'][0]['best_epoch']:>12d} "
              f"{vs_poly:>9.2f}x {improv:>9.2f}x   (dt={time.time()-t0:.1f}s)")

    # 3. Save JSON
    out = {
        "task": task_name,
        "config": {
            "in_dim": in_dim, "hidden": hidden, "K": K, "L": L, "degree": degree,
            "alpha_vals": alpha_vals, "seeds": seeds,
            "lut_epochs": lut_epochs, "poly_epochs": poly_epochs,
            "lr": 5e-4,
        },
        "poly_baseline": {
            "test_mse_per_seed": poly_test_mses,
            "test_mse_mean": poly_mse_mean,
        },
        "results_per_alpha": results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> {out_dir / 'summary.json'}")

    # 4. Plot
    alphas = np.array([r["alpha"] for r in results])
    best_means = np.array([r["test_mse_best_mean"] for r in results])
    best_stds = np.array([r["test_mse_best_std"] for r in results])
    init_means = np.array([r["val_mse_init_mean"] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- Left: MSE vs alpha ---
    ax1.errorbar(alphas, best_means, yerr=best_stds, fmt="o-",
                 color="tab:blue", capsize=4, linewidth=1.8, markersize=7,
                 label="Residual-LUT best-val")
    ax1.axhline(init_means.mean(), color="tab:gray", linestyle=":",
                linewidth=1.2, label=f"Poly-init MSE = {init_means.mean():.2e}")
    ax1.axhline(poly_mse_mean, color="tab:green", linestyle="--",
                linewidth=1.2, label=f"PolyKAN2 = {poly_mse_mean:.2e}")
    ax1.set_xscale("symlog", linthresh=0.01)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\alpha$ (trust-region size)")
    ax1.set_ylabel("Test MSE (best-val)")
    ax1.set_title(f"{task_name}: MSE vs residual α")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8, loc="best")
    for a, m in zip(alphas, best_means):
        ax1.annotate(f"{m:.1e}", (a, m), textcoords="offset points",
                     xytext=(0, 8), fontsize=7, ha="center")

    # --- Right: val-MSE trace per alpha (seed 0) ---
    cm = plt.cm.viridis
    for i, r in enumerate(results):
        color = cm(i / max(len(results) - 1, 1))
        trace = r["trace_seed0"]
        ax2.semilogy(trace["epoch"], trace["mse_val"],
                     color=color, linewidth=1.3,
                     label=f"α={r['alpha']:g}")
    ax2.axhline(init_means.mean(), color="tab:gray", linestyle=":",
                alpha=0.6, label="poly-init")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val MSE")
    ax2.set_title(f"{task_name}: val-MSE trace (seed 0)")
    ax2.legend(fontsize=7, loc="best", ncol=2)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "plot.png", dpi=130)
    plt.close()
    print(f"  -> {out_dir / 'plot.png'}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/M4a_residual_alpha")
    ap.add_argument("--poly-epochs", type=int, default=400)
    ap.add_argument("--lut-epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=4)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.3, 1.0])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Task 1: composition 1D
    x_tr1, y_tr1, x_v1, y_v1, x_te1, y_te1 = generate_data_1d(
        1000, 400, 400, 42)
    r1 = task_pipeline(
        "composition_1d", x_tr1, y_tr1, x_v1, y_v1, x_te1, y_te1,
        in_dim=1, hidden=args.hidden, K=args.K, L=args.L, degree=12,
        alpha_vals=args.alphas, seeds=args.seeds,
        lut_epochs=args.lut_epochs, poly_epochs=args.poly_epochs,
        out_dir=out_dir / "composition_1d",
    )

    # Task 2: feynman_2d
    x_tr2, y_tr2, x_v2, y_v2, x_te2, y_te2 = generate_data_2d(
        "feynman_2d", n_train=1000, n_val=400, n_test=400, seed=42)
    r2 = task_pipeline(
        "feynman_2d", x_tr2, y_tr2, x_v2, y_v2, x_te2, y_te2,
        in_dim=2, hidden=args.hidden, K=args.K, L=args.L, degree=8,
        alpha_vals=args.alphas, seeds=args.seeds,
        lut_epochs=args.lut_epochs, poly_epochs=args.poly_epochs,
        out_dir=out_dir / "feynman_2d",
    )

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"composition_1d": r1, "feynman_2d": r2}, f, indent=2)
    print(f"\n{'='*72}\nM4a done\n{'='*72}")
    print(f"All results in: {out_dir}")


if __name__ == "__main__":
    main()
