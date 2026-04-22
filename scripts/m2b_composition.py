#!/usr/bin/env python3
"""
M2b: Can multi-edge KAN solve a SIMPLER task?

The 2D feynman task is genuinely non-decomposable — maybe that's why nothing
works. Let's try the simplest possible case: [1 -> H -> 1] on a composed target

    y = tanh(2*sin(pi*x))

which IS decomposable: layer-1 can learn sin(pi*x), layer-2 can learn
tanh(2*·). Single-edge KAN cannot represent it exactly because the composition
is not separable along the polynomial basis, but 2-layer KAN trivially can.

If LUT-KAN can't even solve THIS, then multi-edge LUT training has a
fundamental problem. If it can, then layer-2 coverage is the constraint
we need to address (as the feynman_2d diagnostic suggests).

Setup:
    - target:       y = tanh(2 * sin(pi * x))
    - architecture: [1 -> 4 -> 1]
    - K=16, L=32
    - Compare: PolyKAN2 (deg=12) vs post-training LUT vs direct-LUT

Output: results/M2b_composition/summary.json, plots
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
    KAN2TrainConfig,
    LUTKAN2Layer,
    compute_kan2_coverage,
    coverage_report_to_dict,
    kan2_forward_numpy,
    sample_polynomial_to_lut,
    train_kan2,
)
# Reuse PolyKAN2 from exp_H4_kan2_2d
from exp_H4_kan2_2d import PolyKAN2, train_poly_kan2


def target_composition(x):
    return np.tanh(2.0 * np.sin(np.pi * x)).astype(np.float32)


def generate_data_1d(n_train=1000, n_val=400, n_test=400, seed=42):
    rng_t = np.random.RandomState(seed)
    rng_v = np.random.RandomState(seed + 10_000)
    x_tr = rng_t.uniform(-1.0, 1.0, n_train).reshape(-1, 1).astype(np.float32)
    x_v = rng_v.uniform(-1.0, 1.0, n_val).reshape(-1, 1).astype(np.float32)
    x_te = np.linspace(-1.0, 1.0, n_test, endpoint=False).reshape(-1, 1).astype(np.float32)
    y_tr = target_composition(x_tr.ravel())
    y_v = target_composition(x_v.ravel())
    y_te = target_composition(x_te.ravel())
    return x_tr, y_tr, x_v, y_v, x_te, y_te


def poly_init_lut_kan2(poly_coeffs_l1, poly_coeffs_l2, K, L, in_dim, hidden_dim, out_dim):
    model = LUTKAN2Layer(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, K=K, L=L)
    with torch.no_grad():
        for i in range(in_dim):
            for h in range(hidden_dim):
                lut = sample_polynomial_to_lut(poly_coeffs_l1[i, h], K=K, L=L)
                model.lut_l1.data[i, h] = torch.from_numpy(lut)
        for h in range(hidden_dim):
            for o in range(out_dim):
                lut = sample_polynomial_to_lut(poly_coeffs_l2[h, o], K=K, L=L)
                model.lut_l2.data[h, o] = torch.from_numpy(lut)
    return model


def eval_mse(model, x, y):
    with torch.no_grad():
        pred = model(torch.from_numpy(x.astype(np.float32))).cpu().numpy().ravel()
    return float(np.mean((pred - y.ravel()) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/M2b_composition")
    ap.add_argument("--poly-epochs", type=int, default=1200)
    ap.add_argument("--lut-epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=4)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--degree", type=int, default=12)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data_1d(
        n_train=1000, n_val=400, n_test=400, seed=42,
    )
    target_std2 = float(np.var(y_te))
    print(f"Target: y = tanh(2 * sin(pi * x)).  var(y_te) = {target_std2:.3e}")

    rows = []
    lr_vals = [5e-4, 1e-3, 5e-3]   # try three LUT learning rates

    # Train PolyKAN2 per seed
    poly_tests = []
    poly_coeffs_by_seed = []
    for seed in args.seeds:
        t0 = time.time()
        pres = train_poly_kan2(
            in_dim=1, hidden_dim=args.hidden, out_dim=1, degree=args.degree,
            x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
            lr=5e-3, epochs=args.poly_epochs, batch_size=128, seed=seed,
        )
        poly_tests.append(pres["test_mse"])
        poly_coeffs_by_seed.append((pres["coeffs_l1"], pres["coeffs_l2"]))
        print(f"  PolyKAN2 seed={seed}: test MSE = {pres['test_mse']:.3e}  dt={time.time()-t0:.1f}s")

    poly_test_mean = float(np.mean(poly_tests))
    poly_test_std = float(np.std(poly_tests, ddof=1))
    print(f"  PolyKAN2 mean: {poly_test_mean:.3e} ± {poly_test_std:.1e}")

    # Poly-init LUT-KAN2 at three learning rates
    for lr in lr_vals:
        per_seed_inits = []
        per_seed_bests = []
        per_seed_post_q = []   # post-training uint8 eval
        cov_after_last = None
        for seed in args.seeds:
            c1, c2 = poly_coeffs_by_seed[args.seeds.index(seed)]
            model = poly_init_lut_kan2(c1, c2, K=args.K, L=args.L,
                                       in_dim=1, hidden_dim=args.hidden, out_dim=1)
            mse_init = eval_mse(model, x_te, y_te)
            per_seed_inits.append(mse_init)

            # Post-training quantization (uint8)
            # Just evaluate the poly-init LUT as-is in float (no dequant error)
            # and also after uint8 round-trip for the "post-training-LUT" baseline
            from lut_native.baselines import quantize_lut_uint8_asym, dequantize_lut
            lut_q_l1 = np.zeros_like(model.lut_l1.detach().cpu().numpy())
            lut_q_l2 = np.zeros_like(model.lut_l2.detach().cpu().numpy())
            for i in range(1):
                for h in range(args.hidden):
                    q, s, m = quantize_lut_uint8_asym(model.lut_l1[i, h].detach().cpu().numpy())
                    lut_q_l1[i, h] = dequantize_lut(q, s, m)
            for h in range(args.hidden):
                for o in range(1):
                    q, s, m = quantize_lut_uint8_asym(model.lut_l2[h, o].detach().cpu().numpy())
                    lut_q_l2[h, o] = dequantize_lut(q, s, m)
            y_post_q = kan2_forward_numpy(x_te, lut_q_l1, lut_q_l2)
            mse_post_q = float(np.mean((y_post_q.ravel() - y_te.ravel()) ** 2))
            per_seed_post_q.append(mse_post_q)

            # Train direct-LUT
            cfg = KAN2TrainConfig(
                lambda_1=0.0, lambda_2=0.0, lr=lr,
                epochs=args.lut_epochs, batch_size=128,
                init_noise_std_absolute=0.0, seed=seed, eval_every_epochs=15,
            )
            res = train_kan2(model, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)
            per_seed_bests.append(res.mse_test_at_best)

            # Reload best for coverage
            model.lut_l1.data = torch.from_numpy(res.lut_l1_best)
            model.lut_l2.data = torch.from_numpy(res.lut_l2_best)
            cov_after_last = compute_kan2_coverage(model, x_tr)

        inits = np.array(per_seed_inits)
        bests = np.array(per_seed_bests)
        posts = np.array(per_seed_post_q)
        print(f"\n  lr={lr:.0e}:")
        print(f"    poly-init MSE    : {inits.mean():.3e} ± {inits.std(ddof=1):.1e}")
        print(f"    post-training u8 : {posts.mean():.3e} ± {posts.std(ddof=1):.1e}")
        print(f"    direct-LUT best  : {bests.mean():.3e} ± {bests.std(ddof=1):.1e}")
        print(f"    improvement      : {inits.mean()/bests.mean():.2f}x over poly-init")
        print(f"    vs PolyKAN2      : {poly_test_mean/bests.mean():.2f}x "
              f"({'LUT better' if poly_test_mean > bests.mean() else 'LUT worse'})")

        rows.append({
            "lr": lr,
            "poly_init_mse_per_seed": inits.tolist(),
            "post_training_u8_mse_per_seed": posts.tolist(),
            "direct_best_mse_per_seed": bests.tolist(),
            "poly_init_mean": float(inits.mean()),
            "poly_init_std": float(inits.std(ddof=1)),
            "post_u8_mean": float(posts.mean()),
            "post_u8_std": float(posts.std(ddof=1)),
            "direct_best_mean": float(bests.mean()),
            "direct_best_std": float(bests.std(ddof=1)),
            "coverage_after_last_seed": coverage_report_to_dict(cov_after_last),
        })

    out = {
        "config": {"hidden": args.hidden, "K": args.K, "L": args.L,
                   "degree": args.degree, "poly_epochs": args.poly_epochs,
                   "lut_epochs": args.lut_epochs, "seeds": args.seeds,
                   "target": "tanh(2*sin(pi*x))"},
        "target_std2": target_std2,
        "poly_kan2_test_mse_per_seed": poly_tests,
        "poly_kan2_test_mse_mean": poly_test_mean,
        "poly_kan2_test_mse_std": poly_test_std,
        "rows": rows,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_dir / 'summary.json'}")

    # Plot: comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Bar plot per method
    labels = ["Poly-init\n(= post-LUT fp)", "Post-train u8", "Direct-LUT"]
    pos_per_lr = [np.arange(3) + i * 0.28 for i in range(len(rows))]
    width = 0.25
    for r, pos, lr in zip(rows, pos_per_lr, lr_vals):
        data = [r["poly_init_mean"], r["post_u8_mean"], r["direct_best_mean"]]
        errs = [r["poly_init_std"], r["post_u8_std"], r["direct_best_std"]]
        ax1.bar(pos, data, width, yerr=errs, capsize=3, label=f"lr={lr:.0e}")
        for p, d in zip(pos, data):
            ax1.text(p, d * 1.15, f"{d:.1e}", ha="center", fontsize=6.5)
    ax1.axhline(poly_test_mean, color="k", linestyle="--", linewidth=1.2,
                label=f"PolyKAN2 = {poly_test_mean:.1e}")
    ax1.set_xticks(np.arange(3) + width)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Test MSE")
    ax1.set_yscale("log")
    ax1.set_title(f"Composition task: [1→{args.hidden}→1]  "
                  f"y=tanh(2·sin(πx))  ({len(args.seeds)} seeds)",
                  fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, which="both", alpha=0.3, axis="y")

    # Show one learned edge curve from layer 1
    x_dense = np.linspace(-1, 1, 400, dtype=np.float32)
    # Best seed's trained LUT for lr that gave best result
    best_row_idx = np.argmin([r["direct_best_mean"] for r in rows])
    best_row = rows[best_row_idx]
    cov = best_row["coverage_after_last_seed"]

    ax2.plot(x_dense, target_composition(x_dense), "k--", linewidth=1.5, label="target")
    # Overlay coverage bar chart for layer-1 vs layer-2
    KL = args.K * args.L
    metrics = ["visited\nfrac", "eff_supp\n/KL", "range\nutil"]
    l1_vals = [cov["layer1"]["visited_fraction_mean"],
               cov["layer1"]["effective_support_mean"] / KL,
               cov["layer1"]["range_utilization_mean"]]
    l2_vals = [cov["layer2"]["visited_fraction_mean"],
               cov["layer2"]["effective_support_mean"] / KL,
               cov["layer2"]["range_utilization_mean"]]
    xs = np.arange(len(metrics))
    ax2b = ax2  # reuse axes; just keep layout simple
    ax2.cla()
    ax2.bar(xs - 0.2, l1_vals, 0.4, label="Layer 1", color="tab:blue")
    ax2.bar(xs + 0.2, l2_vals, 0.4, label="Layer 2", color="tab:orange")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(metrics, fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.set_title(f"Coverage at lr={lr_vals[best_row_idx]:.0e}  "
                  f"(tanh(z) std={cov['hidden_activation_stats']['tanh_z_std']:.2f})",
                  fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    for x, v1, v2 in zip(xs, l1_vals, l2_vals):
        ax2.text(x - 0.2, v1 + 0.03, f"{v1:.2f}", ha="center", fontsize=7.5)
        ax2.text(x + 0.2, v2 + 0.03, f"{v2:.2f}", ha="center", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=130)
    plt.close()
    print(f"Saved: {out_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
