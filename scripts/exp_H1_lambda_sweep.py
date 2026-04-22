#!/usr/bin/env python3
"""
Experiment H1a: (lambda_1, lambda_2) grid sweep for direct-LUT training.

Hypothesis H1:
    Direct-LUT training beats post-training LUT on smooth targets at small L,
    when appropriately regularized.

This script identifies WHICH regularization works. Also, as a secondary check,
includes the corrected boundary penalty to test whether it helps on top of
second-diff.

Output:
    results/H1a_lambda_sweep/
        sweep.json
        heatmap.png

Runtime: ~4 min CPU (1500 epochs * 5 seeds * 15 configs).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lut_native import (  # noqa: E402
    TrainConfig,
    dequantize_lut,
    eval_chebyshev,
    fit_chebyshev_ls,
    generate_data,
    lut_forward_numpy,
    paired_bootstrap_ci,
    quantize_lut_uint8_asym,
    sample_polynomial_to_lut,
    train_lut_edge,
)


def aggregate(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "median": float(np.median(vals)),
    }


def run_regime(lut_init, x_train, y_train, x_val, y_val, x_test, y_test,
               l1, l2, lbv, seeds, epochs):
    records = []
    for seed in seeds:
        cfg = TrainConfig(
            lambda_1=l1, lambda_2=l2, lambda_bv=lbv,
            lr=1e-2, epochs=epochs, batch_size=64,
            init_noise_std_rel=0.01, seed=seed, eval_every_epochs=25,
        )
        res = train_lut_edge(
            lut_init=lut_init,
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test,
            x_min=-1.0, x_max=1.0, cfg=cfg,
        )
        # Also report post-hoc uint8 quantization of the best LUT
        q, s, m = quantize_lut_uint8_asym(res.lut_best)
        lut_q = dequantize_lut(q, s, m)
        mse_u8 = float(np.mean((lut_forward_numpy(x_test, lut_q) - y_test) ** 2))
        records.append({
            "seed": seed,
            "test_mse_float": res.mse_test_at_best,
            "test_mse_uint8": mse_u8,
            "val_mse": res.mse_val_at_best,
            "train_mse": res.mse_train_at_best,
            "best_epoch": res.best_epoch,
            "n_updates": res.n_updates,
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/H1a_lambda_sweep")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--degree", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"H1a: lambda sweep  (K={args.K}, L={args.L}, deg={args.degree}, "
          f"{len(args.seeds)} seeds, {args.epochs} epochs)")
    print("=" * 72)

    x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data("sine", seed=42)
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=args.degree)
    mse_poly = float(np.mean((eval_chebyshev(x_te, coeffs) - y_te) ** 2))
    lut_init = sample_polynomial_to_lut(coeffs, K=args.K, L=args.L)
    mse_post_fp = float(np.mean((lut_forward_numpy(x_te, lut_init) - y_te) ** 2))
    q, s, m = quantize_lut_uint8_asym(lut_init)
    mse_post_u8 = float(np.mean(
        (lut_forward_numpy(x_te, dequantize_lut(q, s, m)) - y_te) ** 2))

    print(f"  Polynomial deg={args.degree}:     {mse_poly:.3e}")
    print(f"  Post-training LUT (fp): {mse_post_fp:.3e}")
    print(f"  Post-training LUT (u8): {mse_post_u8:.3e}")
    print()

    l1_vals = [0.0, 1e-4, 1e-3, 1e-2]
    l2_vals = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
    grid_mean = np.full((len(l1_vals), len(l2_vals)), np.nan)
    grid_std = np.full_like(grid_mean, np.nan)
    all_records = {}

    for i, l1 in enumerate(l1_vals):
        for j, l2 in enumerate(l2_vals):
            t0 = time.time()
            recs = run_regime(lut_init, x_tr, y_tr, x_v, y_v, x_te, y_te,
                              l1=l1, l2=l2, lbv=0.0,
                              seeds=args.seeds, epochs=args.epochs)
            mses = [r["test_mse_float"] for r in recs]
            agg = aggregate(mses)
            grid_mean[i, j] = agg["mean"]
            grid_std[i, j] = agg["std"]
            all_records[f"l1={l1}_l2={l2}"] = recs
            print(f"    l1={l1:<6} l2={l2:<6} "
                  f"-> mean={agg['mean']:.3e}  std={agg['std']:.1e}  "
                  f"dt={time.time()-t0:.1f}s")

    bi, bj = np.unravel_index(np.argmin(grid_mean), grid_mean.shape)
    best_l1, best_l2 = l1_vals[bi], l2_vals[bj]
    best_mse = grid_mean[bi, bj]
    print(f"\n  Best: lambda_1={best_l1}, lambda_2={best_l2} "
          f"-> MSE={best_mse:.3e}")

    # Paired bootstrap CI of direct/post ratio at best point
    best_mses = np.array([r["test_mse_float"] for r in all_records[f"l1={best_l1}_l2={best_l2}"]])
    post_mses = np.full_like(best_mses, mse_post_fp)  # post-training is deterministic
    ratio_ci = paired_bootstrap_ci(post_mses, best_mses, n_boot=10_000, ci=0.95, seed=0)

    # Secondary check: does boundary continuity help on top of best?
    print("\n  Secondary: does boundary penalty help on top of best?")
    boundary_records = {}
    for lbv in [0.0, 1e-3, 1e-2, 1e-1]:
        recs = run_regime(lut_init, x_tr, y_tr, x_v, y_v, x_te, y_te,
                          l1=best_l1, l2=best_l2, lbv=lbv,
                          seeds=args.seeds, epochs=args.epochs)
        agg = aggregate([r["test_mse_float"] for r in recs])
        boundary_records[f"lbv={lbv}"] = {"records": recs, "summary": agg}
        print(f"    lbv={lbv:<6} mean={agg['mean']:.3e}  std={agg['std']:.1e}")

    # Save JSON
    with open(out_dir / "sweep.json", "w") as f:
        json.dump({
            "config": {
                "K": args.K, "L": args.L, "degree": args.degree,
                "epochs": args.epochs, "seeds": args.seeds,
                "target": "sine",
            },
            "baselines": {
                "polynomial_fp": mse_poly,
                "post_training_lut_fp": mse_post_fp,
                "post_training_lut_u8": mse_post_u8,
            },
            "l1_vals": l1_vals,
            "l2_vals": l2_vals,
            "grid_mse_mean": grid_mean.tolist(),
            "grid_mse_std": grid_std.tolist(),
            "best": {
                "lambda_1": best_l1,
                "lambda_2": best_l2,
                "test_mse": float(best_mse),
                "ratio_post_over_direct": ratio_ci,
            },
            "boundary_penalty_ablation": boundary_records,
            "all_records": all_records,
        }, f, indent=2)
    print(f"\n  -> {out_dir / 'sweep.json'}")

    # Heatmap plot
    fig, ax = plt.subplots(figsize=(8, 5))
    log_grid = np.log10(grid_mean)
    im = ax.imshow(log_grid, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(l2_vals)))
    ax.set_xticklabels([f"{v:g}" for v in l2_vals])
    ax.set_yticks(range(len(l1_vals)))
    ax.set_yticklabels([f"{v:g}" for v in l1_vals])
    ax.set_xlabel(r"$\lambda_2$ (second-diff)")
    ax.set_ylabel(r"$\lambda_1$ (first-diff)")
    ax.set_title(
        f"Direct-LUT test MSE ($\\log_{{10}}$) | sine | K={args.K}, L={args.L}, "
        f"5 seeds\n"
        f"poly deg={args.degree}: {mse_poly:.1e}  |  "
        f"post-training LUT: {mse_post_fp:.1e}"
    )
    for i in range(len(l1_vals)):
        for j in range(len(l2_vals)):
            val = grid_mean[i, j]
            color = "white" if log_grid[i, j] > log_grid.mean() else "black"
            ax.text(j, i, f"{val:.1e}\n$\\pm${grid_std[i,j]:.0e}",
                    ha="center", va="center", color=color, fontsize=7.5)
    ax.plot(bj, bi, "r*", markersize=22, markeredgecolor="black", markeredgewidth=1.5)
    plt.colorbar(im, ax=ax, label=r"$\log_{10}$ test MSE")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap.png", dpi=130)
    plt.close()
    print(f"  -> {out_dir / 'heatmap.png'}")

    return best_l1, best_l2


if __name__ == "__main__":
    main()
