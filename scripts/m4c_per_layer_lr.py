#!/usr/bin/env python3
"""
M4c: Separate learning rates per layer.

Motivation (from M4a findings):
    - composition_1d converges at best_ep ≈ 140-150 (slow, lr might be low)
    - feynman_2d converges at best_ep ≈ 10-30 (fast, or task-specific)
    - Layer 1 and Layer 2 have different roles; uniform lr is probably suboptimal.

Ablation design: 5 configurations to separate layer-asymmetry effects from
overall lr effects. The "both slower" and "both faster" configs are
essential — without them we cannot distinguish "layer 1 needs smaller lr"
from "everything needs smaller lr".

    1. baseline:      lr_l1=5e-4, lr_l2=5e-4  (M4a reference)
    2. L1 slower:     lr_l1=1e-4, lr_l2=5e-4  (hypothesis: L1 is sensitive)
    3. L2 slower:     lr_l1=5e-4, lr_l2=1e-4  (hypothesis: L2 is sensitive)
    4. both slower:   lr_l1=1e-4, lr_l2=1e-4  (control: uniform lr reduction)
    5. both faster:   lr_l1=1e-3, lr_l2=1e-3  (control: uniform lr increase)

If (2) beats (1) and (2) beats (4): genuine L1-specific asymmetry.
If (2)=(4) but both beat (1): it's about overall slower lr, not L1 specifically.
Similar logic for L2 vs both-slower.

Fixed: alpha=0.1 (M4a best), lut_epochs=200, 3 seeds, two tasks.

Output: results/M4c_per_layer_lr/{composition_1d,feynman_2d}/{summary.json,plot.png}
Runtime: ~12 min CPU (2 tasks × 5 configs × 3 seeds × 200 epochs).
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
from exp_H4_kan2_2d import train_poly_kan2
from m2b_composition import generate_data_1d


CONFIGS = [
    ("baseline",    5e-4, 5e-4),
    ("L1_slower",   1e-4, 5e-4),
    ("L2_slower",   5e-4, 1e-4),
    ("both_slower", 1e-4, 1e-4),
    ("both_faster", 1e-3, 1e-3),
]


def _init_from_poly(poly_c1, poly_c2, in_dim, hidden, K, L, alpha):
    model = ResidualLUTKAN2Layer(
        in_dim=in_dim, hidden_dim=hidden, out_dim=1, K=K, L=L, alpha=alpha,
    )
    init_l1 = np.zeros((in_dim, hidden, K, L), dtype=np.float32)
    for i in range(in_dim):
        for h in range(hidden):
            init_l1[i, h] = sample_polynomial_to_lut(poly_c1[i, h], K=K, L=L)
    init_l2 = np.zeros((hidden, 1, K, L), dtype=np.float32)
    for h in range(hidden):
        init_l2[h, 0] = sample_polynomial_to_lut(poly_c2[h, 0], K=K, L=L)
    model.init_layer1_from_arrays(init_l1)
    model.init_layer2_from_arrays(init_l2)
    return model


def run_config(poly_coeffs_per_seed, seeds, in_dim, hidden, K, L, alpha,
               lr_l1, lr_l2, epochs,
               x_tr, y_tr, x_v, y_v, x_te, y_te):
    per_seed = []
    trace_seed0 = None
    for idx, seed in enumerate(seeds):
        c1, c2 = poly_coeffs_per_seed[idx]
        model = _init_from_poly(c1, c2, in_dim, hidden, K, L, alpha)
        cfg = ResidualTrainConfig(
            lambda_1=0.0, lambda_2=0.0,
            lambda_init_anchor_l1=0.0, lambda_init_anchor_l2=0.0,
            lr_l1=lr_l1, lr_l2=lr_l2,
            epochs=epochs, batch_size=128, seed=seed, eval_every_epochs=10,
        )
        res = train_residual_kan2(model, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)
        per_seed.append({
            "seed": seed,
            "mse_val_init": res.mse_val_init,
            "mse_val_at_best": res.mse_val_at_best,
            "mse_test_at_best": res.mse_test_at_best,
            "mse_val_final": res.mse_val_final,
            "best_epoch": res.best_epoch,
        })
        if idx == 0:
            trace_seed0 = res.trace
    arr = np.array([p["mse_test_at_best"] for p in per_seed])
    return {
        "lr_l1": lr_l1, "lr_l2": lr_l2,
        "per_seed": per_seed,
        "test_mse_mean": float(arr.mean()),
        "test_mse_std": float(arr.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "trace_seed0": trace_seed0,
    }


def task_pipeline(task_name, x_tr, y_tr, x_v, y_v, x_te, y_te,
                  in_dim, hidden, K, L, degree, alpha, seeds,
                  lut_epochs, poly_epochs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\n{task_name}\n{'='*72}")

    # Train polys
    poly_coeffs = []
    poly_mses = []
    for seed in seeds:
        pres = train_poly_kan2(
            in_dim=in_dim, hidden_dim=hidden, out_dim=1, degree=degree,
            x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
            lr=5e-3, epochs=poly_epochs, batch_size=128, seed=seed,
        )
        poly_coeffs.append((pres["coeffs_l1"], pres["coeffs_l2"]))
        poly_mses.append(pres["test_mse"])
        print(f"  PolyKAN2 seed={seed}: {pres['test_mse']:.3e}")
    poly_mean = float(np.mean(poly_mses))
    print(f"  PolyKAN2 mean: {poly_mean:.3e}")

    # Sweep configs
    print(f"\n  Per-layer lr ablation (alpha={alpha}, {lut_epochs} ep, {len(seeds)} seeds):")
    print(f"  {'config':<14} {'lr_l1':>8} {'lr_l2':>8} {'test MSE':>14} "
          f"{'best_ep':>8} {'vs Poly':>10}")
    results = []
    for name, lr1, lr2 in CONFIGS:
        t0 = time.time()
        r = run_config(poly_coeffs, seeds, in_dim, hidden, K, L, alpha,
                       lr1, lr2, lut_epochs,
                       x_tr, y_tr, x_v, y_v, x_te, y_te)
        r["name"] = name
        results.append(r)
        print(f"  {name:<14} {lr1:>8.0e} {lr2:>8.0e} "
              f"{r['test_mse_mean']:>8.2e}±{r['test_mse_std']:>3.0e} "
              f"{r['per_seed'][0]['best_epoch']:>8d} "
              f"{poly_mean/r['test_mse_mean']:>9.2f}x  "
              f"(dt={time.time()-t0:.1f}s)")

    # Save
    out = {
        "task": task_name,
        "config": {"in_dim": in_dim, "hidden": hidden, "K": K, "L": L,
                   "degree": degree, "alpha": alpha, "seeds": seeds,
                   "lut_epochs": lut_epochs, "poly_epochs": poly_epochs,
                   "configs": [{"name": n, "lr_l1": a, "lr_l2": b}
                               for (n, a, b) in CONFIGS]},
        "poly_baseline": {"test_mse_per_seed": poly_mses,
                          "test_mse_mean": poly_mean},
        "results": results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    names = [r["name"] for r in results]
    means = np.array([r["test_mse_mean"] for r in results])
    stds = np.array([r["test_mse_std"] for r in results])
    xs = np.arange(len(names))
    colors = plt.cm.tab10(np.arange(len(names)))
    ax1.bar(xs, means, yerr=stds, capsize=4, color=colors, edgecolor="black")
    ax1.axhline(poly_mean, color="k", linestyle="--", linewidth=1.2,
                label=f"PolyKAN2 = {poly_mean:.2e}")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, rotation=20, fontsize=9)
    ax1.set_yscale("log")
    ax1.set_ylabel("Test MSE (best-val)")
    ax1.set_title(f"{task_name}: per-layer lr ablation")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", which="both", alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m * 1.1, f"{m:.2e}", ha="center", fontsize=7.5)

    for i, r in enumerate(results):
        ax2.semilogy(r["trace_seed0"]["epoch"],
                     r["trace_seed0"]["mse_val"],
                     color=colors[i], linewidth=1.4, label=r["name"])
    ax2.axhline(poly_mean, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val MSE")
    ax2.set_title(f"{task_name}: val-MSE trace (seed 0)")
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "plot.png", dpi=130)
    plt.close()
    print(f"  -> {out_dir / 'plot.png'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/M4c_per_layer_lr")
    ap.add_argument("--poly-epochs", type=int, default=300)
    ap.add_argument("--lut-epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_tr1, y_tr1, x_v1, y_v1, x_te1, y_te1 = generate_data_1d(1000, 400, 400, 42)
    r1 = task_pipeline("composition_1d", x_tr1, y_tr1, x_v1, y_v1, x_te1, y_te1,
                       in_dim=1, hidden=4, K=16, L=32, degree=12,
                       alpha=args.alpha, seeds=args.seeds,
                       lut_epochs=args.lut_epochs, poly_epochs=args.poly_epochs,
                       out_dir=out_dir / "composition_1d")

    x_tr2, y_tr2, x_v2, y_v2, x_te2, y_te2 = generate_data_2d(
        "feynman_2d", n_train=1000, n_val=400, n_test=400, seed=42)
    r2 = task_pipeline("feynman_2d", x_tr2, y_tr2, x_v2, y_v2, x_te2, y_te2,
                       in_dim=2, hidden=4, K=16, L=32, degree=8,
                       alpha=args.alpha, seeds=args.seeds,
                       lut_epochs=args.lut_epochs, poly_epochs=args.poly_epochs,
                       out_dir=out_dir / "feynman_2d")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"composition_1d": r1, "feynman_2d": r2}, f, indent=2)
    print(f"\n{'='*72}\nM4c done\n{'='*72}\n")


if __name__ == "__main__":
    main()
