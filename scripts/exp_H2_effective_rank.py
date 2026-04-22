#!/usr/bin/env python3
"""
Experiment H2: is the learned LUT secretly low-rank (i.e. polynomial-like)?

Rationale:
    When we train a LUT with strong curvature penalty (lambda_2 >> 0), we
    suspect the optimizer converges to a low-rank solution where all segments
    share a common smooth shape, differing only by scale/offset.

    If this is the case, the LUT's apparent 512 parameters (K*L = 16*32) are
    effectively using only ~d DOF, which would undermine the claim that LUT
    representation is fundamentally different from polynomial.

    We test this via the singular value spectrum of the de-meaned LUT.

    For the post-training LUT sampled from a polynomial of degree d, the rank
    should be ~min(d+1, K). For a free LUT trained without structure, the rank
    should be higher. This diagnostic tells us how "independent" the learned
    shapes across segments are.

Output:
    results/H2_effective_rank/
        summary.json
        spectrum.png

Runtime: ~2 min CPU.
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
    effective_rank,
    eval_chebyshev,
    fit_chebyshev_ls,
    generate_data,
    lut_forward_numpy,
    quantize_lut_uint8_asym,
    sample_polynomial_to_lut,
    singular_value_spectrum,
    train_lut_edge,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/H2_effective_rank")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--degree", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"H2: effective rank  (K={args.K}, L={args.L}, deg={args.degree})")
    print("=" * 72)

    x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data("sine", seed=42)
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=args.degree)
    lut_init = sample_polynomial_to_lut(coeffs, K=args.K, L=args.L)

    regimes = [
        ("post_training", 0.0, 0.0),
        ("direct_noreg", 0.0, 0.0),      # no reg, but still trained
        ("direct_l2_0.01", 0.0, 0.01),
        ("direct_l2_0.1", 0.0, 0.1),
        ("direct_l2_1.0", 0.0, 1.0),
    ]

    records = {}
    for name, l1, l2 in regimes:
        if name == "post_training":
            luts = [lut_init.copy()]  # deterministic, single "seed"
            test_mses = [float(np.mean((lut_forward_numpy(x_te, lut_init) - y_te) ** 2))]
        else:
            luts = []
            test_mses = []
            for seed in args.seeds:
                cfg = TrainConfig(lambda_1=l1, lambda_2=l2,
                                  lr=1e-2, epochs=args.epochs, batch_size=64,
                                  init_noise_std_rel=0.01, seed=seed,
                                  eval_every_epochs=25)
                r = train_lut_edge(lut_init=lut_init,
                                  x_train=x_tr, y_train=y_tr,
                                  x_val=x_v, y_val=y_v,
                                  x_test=x_te, y_test=y_te,
                                  x_min=-1.0, x_max=1.0, cfg=cfg)
                luts.append(r.lut_best)
                test_mses.append(r.mse_test_at_best)

        # Compute diagnostics
        ranks = [effective_rank(lut, energy_frac=0.99) for lut in luts]
        ranks_95 = [effective_rank(lut, energy_frac=0.95) for lut in luts]
        spectra = [singular_value_spectrum(lut) for lut in luts]
        records[name] = {
            "lambda_1": l1,
            "lambda_2": l2,
            "test_mse_mean": float(np.mean(test_mses)),
            "test_mse_std": float(np.std(test_mses, ddof=1)) if len(test_mses) > 1 else 0.0,
            "effective_rank_99_mean": float(np.mean(ranks)),
            "effective_rank_95_mean": float(np.mean(ranks_95)),
            "singular_values_seed0": spectra[0].tolist(),
        }
        print(f"  {name:<20}  MSE={np.mean(test_mses):.2e}  "
              f"eff_rank_99={np.mean(ranks):.1f}  eff_rank_95={np.mean(ranks_95):.1f}")

    # Reference: the polynomial basis itself, sampled onto the grid, has rank = degree+1
    # (minus 1 because we center per segment, but each segment's curvature component
    # has degree DOF in common across segments). For a full-rank target, rank = min(K, L).
    records["_reference_full_rank"] = min(args.K, args.L)
    records["_reference_degree_plus_1"] = args.degree + 1

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "config": {
                "K": args.K, "L": args.L, "degree": args.degree,
                "epochs": args.epochs, "seeds": args.seeds,
                "target": "sine",
            },
            "records": records,
        }, f, indent=2)
    print(f"\n  -> {out_dir / 'summary.json'}")

    # Plot singular value spectra
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    colors = {"post_training": "tab:red",
              "direct_noreg": "tab:gray",
              "direct_l2_0.01": "tab:purple",
              "direct_l2_0.1": "tab:orange",
              "direct_l2_1.0": "tab:blue"}

    for name, _, _ in regimes:
        sv = np.array(records[name]["singular_values_seed0"])
        label = f"{name}  (eff-rank@99={records[name]['effective_rank_99_mean']:.1f})"
        ax1.plot(np.arange(1, len(sv) + 1), sv, "o-",
                 label=label, color=colors[name], markersize=5)

    ax1.axvline(args.degree + 1, color="k", linestyle=":", alpha=0.5,
                label=f"polynomial degree+1 = {args.degree+1}")
    ax1.set_yscale("log")
    ax1.set_xlabel("Singular value index")
    ax1.set_ylabel("Singular value (log scale)")
    ax1.set_title("Singular value spectrum of (LUT - per-segment mean)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, which="both", alpha=0.3)

    # Cumulative energy
    for name, _, _ in regimes:
        sv = np.array(records[name]["singular_values_seed0"])
        cum = np.cumsum(sv ** 2) / (sv ** 2).sum()
        ax2.plot(np.arange(1, len(sv) + 1), cum, "o-",
                 color=colors[name], markersize=5, label=name)
    ax2.axhline(0.99, color="k", linestyle=":", alpha=0.5, label="99% energy")
    ax2.axhline(0.95, color="gray", linestyle=":", alpha=0.5, label="95% energy")
    ax2.axvline(args.degree + 1, color="k", linestyle=":", alpha=0.3)
    ax2.set_xlabel("Singular value index")
    ax2.set_ylabel("Cumulative energy fraction")
    ax2.set_title("Cumulative spectrum energy")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Effective rank of learned LUT  (K={args.K}, L={args.L}, sine target)"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "spectrum.png", dpi=130)
    plt.close()
    print(f"  -> {out_dir / 'spectrum.png'}")


if __name__ == "__main__":
    main()
