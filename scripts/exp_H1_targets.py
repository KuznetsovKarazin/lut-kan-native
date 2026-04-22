#!/usr/bin/env python3
"""
Experiment H1c: does the direct-LUT advantage generalize across target types?

Three targets:
    sine       — smooth, band-limited.  Polynomial wins in abs. terms.
    cusp       — C^0, polynomial has a Gibbs-like floor here.
    saturating — tanh(4x)+0.15x. Smooth but strongly non-linear.

For each, we compare:
    polynomial (deg=20)  -- reference
    post-training LUT    -- v2.1 pipeline
    direct-LUT           -- trained with best lambdas from H1a

Produces a single summary plot and a curves-overlay plot per target.

Runtime: ~3 min CPU.
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
    TARGETS,
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


def evaluate_target(target_name, K, L, degree, l1, l2, seeds, epochs):
    x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data(target_name, seed=42)

    # Polynomial baselines across several degrees (shows saturation)
    poly_mse = {}
    for d in [8, 12, 16, 20, 32]:
        c = fit_chebyshev_ls(x_tr, y_tr, degree=d)
        poly_mse[d] = float(np.mean((eval_chebyshev(x_te, c) - y_te) ** 2))

    # Reference polynomial for LUT init
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=degree)
    lut_init = sample_polynomial_to_lut(coeffs, K=K, L=L)

    mse_post_fp = float(np.mean((lut_forward_numpy(x_te, lut_init) - y_te) ** 2))
    q, s, m = quantize_lut_uint8_asym(lut_init)
    mse_post_u8 = float(np.mean(
        (lut_forward_numpy(x_te, dequantize_lut(q, s, m)) - y_te) ** 2))

    # Direct-LUT at best lambdas
    direct_fp, direct_u8 = [], []
    best_lut_for_plot = None
    best_seed = seeds[0]
    for seed in seeds:
        cfg = TrainConfig(lambda_1=l1, lambda_2=l2, lr=1e-2,
                          epochs=epochs, batch_size=64,
                          init_noise_std_rel=0.01, seed=seed,
                          eval_every_epochs=25)
        r = train_lut_edge(lut_init=lut_init,
                          x_train=x_tr, y_train=y_tr,
                          x_val=x_v, y_val=y_v,
                          x_test=x_te, y_test=y_te,
                          x_min=-1.0, x_max=1.0, cfg=cfg)
        direct_fp.append(r.mse_test_at_best)
        q2, s2, m2 = quantize_lut_uint8_asym(r.lut_best)
        direct_u8.append(float(np.mean(
            (lut_forward_numpy(x_te, dequantize_lut(q2, s2, m2)) - y_te) ** 2)))
        if seed == best_seed:
            best_lut_for_plot = r.lut_best

    direct_fp = np.array(direct_fp)
    direct_u8 = np.array(direct_u8)

    ratio_ci = paired_bootstrap_ci(
        np.full_like(direct_fp, mse_post_fp), direct_fp,
        n_boot=10_000, seed=0)

    return {
        "target": target_name,
        "description": TARGETS[target_name].description,
        "poly_mse_by_degree": poly_mse,
        "post_training_fp": mse_post_fp,
        "post_training_u8": mse_post_u8,
        "direct_fp_values": direct_fp.tolist(),
        "direct_u8_values": direct_u8.tolist(),
        "direct_fp_mean": float(direct_fp.mean()),
        "direct_fp_std": float(direct_fp.std(ddof=1)),
        "direct_u8_mean": float(direct_u8.mean()),
        "direct_u8_std": float(direct_u8.std(ddof=1)),
        "ratio_post_over_direct_fp": ratio_ci,
        "coeffs_deg20": coeffs.tolist(),
        "lut_init": lut_init.tolist(),
        "lut_direct_seed0": best_lut_for_plot.tolist() if best_lut_for_plot is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/H1c_targets")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--degree", type=int, default=20)
    ap.add_argument("--lambda-1", type=float, default=0.0)
    ap.add_argument("--lambda-2", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"H1c: target sweep  (K={args.K}, L={args.L}, deg={args.degree}, "
          f"l1={args.lambda_1}, l2={args.lambda_2})")
    print("=" * 72)

    results = {}
    for tname in ["sine", "cusp", "saturating"]:
        t0 = time.time()
        r = evaluate_target(tname, args.K, args.L, args.degree,
                            args.lambda_1, args.lambda_2,
                            args.seeds, args.epochs)
        results[tname] = r
        rci = r["ratio_post_over_direct_fp"]
        print(f"  {tname:<12} post={r['post_training_fp']:.2e}  "
              f"direct={r['direct_fp_mean']:.2e}±{r['direct_fp_std']:.1e}  "
              f"ratio={rci['ratio_mean']:.1f}x "
              f"[{rci['ratio_ci_lower']:.1f},{rci['ratio_ci_upper']:.1f}]  "
              f"dt={time.time()-t0:.1f}s")

    # JSON (strip big arrays for compactness — we save them separately)
    compact = {k: {kk: vv for kk, vv in v.items()
                   if kk not in ("coeffs_deg20", "lut_init", "lut_direct_seed0")}
               for k, v in results.items()}
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "config": {
                "K": args.K, "L": args.L, "degree": args.degree,
                "lambda_1": args.lambda_1, "lambda_2": args.lambda_2,
                "epochs": args.epochs, "seeds": args.seeds,
            },
            "results": compact,
        }, f, indent=2)

    # Curve overlay plot per target
    x_dense = np.linspace(-1.0, 1.0, 1000, endpoint=False).astype(np.float32)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, tname in zip(axes, ["sine", "cusp", "saturating"]):
        r = results[tname]
        y_true = TARGETS[tname].fn(x_dense)
        coeffs = np.array(r["coeffs_deg20"], dtype=np.float32)
        lut_init = np.array(r["lut_init"], dtype=np.float32)
        lut_direct = np.array(r["lut_direct_seed0"], dtype=np.float32)
        y_poly = eval_chebyshev(x_dense, coeffs)
        y_post = lut_forward_numpy(x_dense, lut_init)
        y_direct = lut_forward_numpy(x_dense, lut_direct)

        ax.plot(x_dense, y_true, "k--", linewidth=1.3, label="Target", alpha=0.8)
        ax.plot(x_dense, y_poly, color="tab:green", linewidth=1.1,
                label=f"Poly deg={args.degree}  ({r['poly_mse_by_degree'][args.degree]:.1e})",
                alpha=0.7)
        ax.plot(x_dense, y_post, color="tab:red", linewidth=1.1,
                label=f"Post-LUT  ({r['post_training_fp']:.1e})",
                alpha=0.7)
        ax.plot(x_dense, y_direct, color="tab:blue", linewidth=1.3,
                label=f"Direct-LUT  ({r['direct_fp_mean']:.1e})")
        ax.set_title(f"{tname}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc="best")
    fig.suptitle(
        f"Direct-LUT vs post-training LUT across target types  "
        f"(K={args.K}, L={args.L}, {len(args.seeds)} seeds; direct-LUT from seed 0 shown)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "curves.png", dpi=130)
    plt.close()
    print(f"  -> {out_dir / 'curves.png'}")

    # Bar plot of ratios
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    targets = list(results)
    ratios = [results[t]["ratio_post_over_direct_fp"]["ratio_mean"] for t in targets]
    lo = [results[t]["ratio_post_over_direct_fp"]["ratio_ci_lower"] for t in targets]
    hi = [results[t]["ratio_post_over_direct_fp"]["ratio_ci_upper"] for t in targets]
    yerr = np.array([[r-l for r,l in zip(ratios, lo)],
                     [h-r for r,h in zip(ratios, hi)]])
    bars = ax.bar(targets, ratios, yerr=yerr, capsize=6,
                  color=["tab:blue", "tab:orange", "tab:green"])
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="parity")
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, r * 1.05,
                f"{r:.1f}x", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Post-training / Direct-LUT  test-MSE ratio  (higher = direct better)")
    ax.set_yscale("log")
    ax.set_title(
        f"Direct-LUT advantage by target type  (K={args.K}, L={args.L})\n"
        f"Error bars: 95% paired bootstrap CI"
    )
    ax.grid(True, which="both", alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "ratio_bars.png", dpi=130)
    plt.close()
    print(f"  -> {out_dir / 'ratio_bars.png'}")


if __name__ == "__main__":
    main()
