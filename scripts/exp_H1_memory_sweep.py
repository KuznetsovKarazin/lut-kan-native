#!/usr/bin/env python3
"""
Experiment H1b/H3: memory sweep — test MSE vs L for direct-LUT vs post-training LUT.

Hypotheses:
    H1: Direct-LUT wins at small L (memory-constrained regime).
    H3: Advantage disappears (or inverts) at large L where interpolation is
        no longer the bottleneck. This is a honest negative result that
        defines the claim's boundary.

Output:
    results/H1b_memory_sweep/
        sweep.json
        plot.png

Runtime: ~4 min CPU.
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


def run_config(lut_init, data, l1, l2, seeds, epochs):
    x_tr, y_tr, x_v, y_v, x_te, y_te = data
    mses_fp, mses_u8 = [], []
    for seed in seeds:
        cfg = TrainConfig(
            lambda_1=l1, lambda_2=l2,
            lr=1e-2, epochs=epochs, batch_size=64,
            init_noise_std_rel=0.01, seed=seed, eval_every_epochs=25,
        )
        r = train_lut_edge(lut_init=lut_init,
                          x_train=x_tr, y_train=y_tr,
                          x_val=x_v, y_val=y_v,
                          x_test=x_te, y_test=y_te,
                          x_min=-1.0, x_max=1.0, cfg=cfg)
        mses_fp.append(r.mse_test_at_best)
        q, s, m = quantize_lut_uint8_asym(r.lut_best)
        lut_q = dequantize_lut(q, s, m)
        mses_u8.append(float(np.mean((lut_forward_numpy(x_te, lut_q) - y_te) ** 2)))
    return np.array(mses_fp), np.array(mses_u8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/H1b_memory_sweep")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--L-values", type=int, nargs="+",
                    default=[8, 16, 32, 64, 128])
    ap.add_argument("--degree", type=int, default=20)
    ap.add_argument("--lambda-1", type=float, default=0.0)
    ap.add_argument("--lambda-2", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"H1b: memory sweep (K={args.K}, L in {args.L_values}, "
          f"l1={args.lambda_1}, l2={args.lambda_2}, {len(args.seeds)} seeds)")
    print("=" * 72)

    data = generate_data("sine", seed=42)
    x_tr, y_tr, x_v, y_v, x_te, y_te = data
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=args.degree)
    mse_poly = float(np.mean((eval_chebyshev(x_te, coeffs) - y_te) ** 2))
    print(f"  Polynomial deg={args.degree} baseline: {mse_poly:.3e}\n")

    rows = []
    for L in args.L_values:
        lut_init = sample_polynomial_to_lut(coeffs, K=args.K, L=L)
        mse_post_fp = float(np.mean((lut_forward_numpy(x_te, lut_init) - y_te) ** 2))
        q, s, m = quantize_lut_uint8_asym(lut_init)
        mse_post_u8 = float(np.mean(
            (lut_forward_numpy(x_te, dequantize_lut(q, s, m)) - y_te) ** 2))

        t0 = time.time()
        direct_fp, direct_u8 = run_config(
            lut_init, data, args.lambda_1, args.lambda_2,
            args.seeds, args.epochs)

        # Paired bootstrap on log-ratio post/direct (post is deterministic so
        # "paired" here just means the post baseline applies to all seeds)
        ratio_ci = paired_bootstrap_ci(
            np.full_like(direct_fp, mse_post_fp), direct_fp,
            n_boot=10_000, seed=0,
        )

        row = {
            "L": L,
            "memory_bytes_float32": args.K * L * 4,
            "memory_bytes_uint8_lut": args.K * L + args.K * 4,  # q + scale(f16) + ymin(f16)
            "post_training_fp": mse_post_fp,
            "post_training_u8": mse_post_u8,
            "direct_fp_mean": float(direct_fp.mean()),
            "direct_fp_std": float(direct_fp.std(ddof=1)),
            "direct_fp_values": direct_fp.tolist(),
            "direct_u8_mean": float(direct_u8.mean()),
            "direct_u8_std": float(direct_u8.std(ddof=1)),
            "direct_u8_values": direct_u8.tolist(),
            "ratio_post_over_direct_fp": ratio_ci,
        }
        rows.append(row)
        print(f"    L={L:<4} post(fp)={mse_post_fp:.3e}  "
              f"direct(fp)={direct_fp.mean():.3e}"
              f"±{direct_fp.std(ddof=1):.1e}  "
              f"ratio={ratio_ci['ratio_mean']:.2f}x "
              f"[{ratio_ci['ratio_ci_lower']:.2f}, {ratio_ci['ratio_ci_upper']:.2f}]  "
              f"dt={time.time()-t0:.1f}s")

    with open(out_dir / "sweep.json", "w") as f:
        json.dump({
            "config": {
                "K": args.K, "L_values": args.L_values, "degree": args.degree,
                "lambda_1": args.lambda_1, "lambda_2": args.lambda_2,
                "epochs": args.epochs, "seeds": args.seeds,
                "target": "sine",
            },
            "polynomial_baseline_fp": mse_poly,
            "rows": rows,
        }, f, indent=2)
    print(f"\n  -> {out_dir / 'sweep.json'}")

    # Plot
    Ls = np.array([r["L"] for r in rows])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.axhline(mse_poly, color="k", linestyle="--", linewidth=1.2,
               label=f"Polynomial deg={args.degree} (MSE={mse_poly:.1e})", zorder=1)
    ax.plot(Ls, [r["post_training_fp"] for r in rows], "o-",
            label="Post-training LUT (v2.1, fp)",
            color="tab:red", markersize=8, linewidth=2, zorder=3)
    ax.errorbar(Ls, [r["direct_fp_mean"] for r in rows],
                yerr=[r["direct_fp_std"] for r in rows],
                fmt="s-", color="tab:blue", markersize=8, linewidth=2,
                capsize=5,
                label=f"Direct-LUT ($\\lambda_2$={args.lambda_2}, fp)", zorder=4)
    # Annotate crossover zone
    for i, r in enumerate(rows):
        txt = f"{r['ratio_post_over_direct_fp']['ratio_mean']:.1f}x"
        ax.annotate(txt, (Ls[i], r["direct_fp_mean"]),
                    textcoords="offset points", xytext=(0, -14),
                    fontsize=8, ha="center", color="tab:blue")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ls)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("L (samples per segment)")
    ax.set_ylabel("Test MSE")
    ax.set_title(
        f"Sine target: direct-LUT vs post-training LUT across memory budgets\n"
        f"K={args.K}, {len(args.seeds)} seeds, mini-batch SGD, "
        f"val-based best-model selection"
    )
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot.png", dpi=130)
    plt.close()
    print(f"  -> {out_dir / 'plot.png'}")


if __name__ == "__main__":
    main()
