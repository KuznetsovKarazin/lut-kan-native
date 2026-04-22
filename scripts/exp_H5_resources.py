#!/usr/bin/env python3
"""
Experiment H5: Resource accounting benchmark.

For a fixed KAN architecture [2 -> 4 -> 1] on feynman_2d, reports:
  - static memory (bytes)
  - per-sample int/float ops (from resources.py analytic counts)
  - measured CPU latency (via timeit)

for three evaluators at matched architecture. Note that polynomial and LUT
naturally have very different memory profiles; we report both separately
and as a Pareto plot.

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lut_native import (  # noqa: E402
    generate_data_2d,
    lut_memory_bytes,
    lut_ops_per_sample,
    multi_edge_kan_ops,
    multi_edge_lut_memory_bytes,
    polynomial_memory_bytes,
    polynomial_ops_per_sample,
    time_forward,
)
from lut_native.kan2 import (  # noqa: E402
    kan2_forward_numpy,
    polynomial_kan2_forward_numpy,
)
from exp_H4_kan2_2d import (  # noqa: E402
    luts_from_polynomial_kan,
    quantize_kan_luts,
    train_poly_kan2,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/H5_resources")
    ap.add_argument("--poly-epochs", type=int, default=500)
    ap.add_argument("--hidden-dim", type=int, default=4)
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--K-values", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--L-values", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--n-reps", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("H5: resource benchmark")
    print(f"  Architecture: [2 -> {args.hidden_dim} -> 1], poly deg={args.degree}")
    print(f"  K values: {args.K_values}, L values: {args.L_values}")
    print("=" * 72)

    x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data_2d("feynman_2d", seed=42)
    n_edges = 2 * args.hidden_dim + args.hidden_dim * 1

    # ─── Train one poly-KAN as reference / LUT-init source ────────────────
    print("\n[A] Train polynomial KAN once (seed=0) for init...")
    t0 = time.time()
    best_poly = train_poly_kan2(
        in_dim=2, hidden_dim=args.hidden_dim, out_dim=1, degree=args.degree,
        x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
        lr=5e-3, epochs=args.poly_epochs, batch_size=128, seed=0,
    )
    print(f"    test_MSE={best_poly['test_mse']:.3e}, dt={time.time()-t0:.1f}s")

    # ─── Benchmark latency for each evaluator ──────────────────────────────
    print("\n[B] CPU latency measurement")
    x_bench = x_te.astype(np.float32)  # (400, 2)

    # Polynomial KAN
    def poly_fn(x):
        return polynomial_kan2_forward_numpy(
            x, best_poly["coeffs_l1"], best_poly["coeffs_l2"])

    poly_lat = time_forward(poly_fn, x_bench, n_reps=args.n_reps)
    print(f"    Polynomial KAN (deg={args.degree}):")
    print(f"      {poly_lat['median_us_per_sample']:.2f} us/sample "
          f"(median of {args.n_reps} reps)")

    # LUT-KAN variants at different (K, L)
    rows = []
    for K in args.K_values:
        for L in args.L_values:
            lut_l1, lut_l2 = luts_from_polynomial_kan(
                best_poly["coeffs_l1"], best_poly["coeffs_l2"], K=K, L=L)
            lut_l1_q, lut_l2_q = quantize_kan_luts(lut_l1, lut_l2)

            # Verify accuracy so user knows each config is meaningful
            y_pred = kan2_forward_numpy(x_bench, lut_l1_q, lut_l2_q)
            mse = float(np.mean((y_pred.ravel() - y_te.ravel()) ** 2))

            def lut_fn(x, l1=lut_l1_q, l2=lut_l2_q):
                return kan2_forward_numpy(x, l1, l2)

            lat = time_forward(lut_fn, x_bench, n_reps=args.n_reps)
            mem = multi_edge_lut_memory_bytes(n_edges, K, L, dtype="uint8",
                                              meta_dtype="float16")
            ops = multi_edge_kan_ops(
                in_dim=2, hidden_dim=args.hidden_dim, out_dim=1,
                eval_ops_per_edge=lut_ops_per_sample(K, L, "uint8"),
            )
            row = {
                "method": f"LUT-KAN K={K} L={L}",
                "K": K, "L": L,
                "memory_bytes": mem,
                "ops_int": ops.int_ops,
                "ops_float": ops.float_ops,
                "test_mse_uint8": mse,
                "latency_median_us": lat["median_us_per_sample"],
                "latency_std_us": lat["std_us_per_sample"],
            }
            rows.append(row)
            print(f"    LUT-KAN K={K:>3} L={L:>3}: mem={mem:>6} B, "
                  f"{lat['median_us_per_sample']:.2f} us/sample, "
                  f"MSE={mse:.2e}")

    # Polynomial row
    poly_mem = n_edges * polynomial_memory_bytes(args.degree, dtype="float32")
    poly_ops = multi_edge_kan_ops(
        in_dim=2, hidden_dim=args.hidden_dim, out_dim=1,
        eval_ops_per_edge=polynomial_ops_per_sample(args.degree),
    )
    poly_row = {
        "method": f"PolyKAN deg={args.degree}",
        "K": None, "L": None,
        "memory_bytes": poly_mem,
        "ops_int": poly_ops.int_ops,
        "ops_float": poly_ops.float_ops,
        "test_mse_uint8": best_poly["test_mse"],
        "latency_median_us": poly_lat["median_us_per_sample"],
        "latency_std_us": poly_lat["std_us_per_sample"],
    }

    # ─── Save JSON ────────────────────────────────────────────────────────
    with open(out_dir / "resources.json", "w") as f:
        json.dump({
            "config": {
                "architecture": f"2->{args.hidden_dim}->1",
                "degree": args.degree,
                "K_values": args.K_values, "L_values": args.L_values,
                "n_edges": n_edges,
                "n_samples_per_batch": int(x_bench.shape[0]),
                "n_reps": args.n_reps,
            },
            "polynomial": poly_row,
            "lut_variants": rows,
        }, f, indent=2)
    print(f"\n-> {out_dir / 'resources.json'}")

    # ─── Plot: Pareto (memory vs latency, MSE as size) ─────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Latency vs memory
    for r in rows:
        size_marker = 60 + 400 * (1.0 / max(r["test_mse_uint8"], 1e-6)) ** 0.3
        ax1.scatter(r["memory_bytes"], r["latency_median_us"],
                    color="tab:blue", alpha=0.7, s=size_marker,
                    edgecolor="black", linewidth=0.5)
        ax1.annotate(f"K={r['K']},L={r['L']}\n{r['test_mse_uint8']:.1e}",
                     (r["memory_bytes"], r["latency_median_us"]),
                     textcoords="offset points", xytext=(8, 5), fontsize=7.5)
    size_p = 60 + 400 * (1.0 / max(poly_row["test_mse_uint8"], 1e-6)) ** 0.3
    ax1.scatter(poly_row["memory_bytes"], poly_row["latency_median_us"],
                color="tab:green", alpha=0.7, s=size_p,
                edgecolor="black", linewidth=0.5, marker="^", zorder=3)
    ax1.annotate(f"PolyKAN deg={args.degree}\n{poly_row['test_mse_uint8']:.1e}",
                 (poly_row["memory_bytes"], poly_row["latency_median_us"]),
                 textcoords="offset points", xytext=(8, -10), fontsize=7.5,
                 color="tab:green", fontweight="bold")
    ax1.set_xlabel("Memory footprint (bytes)")
    ax1.set_ylabel("CPU latency (us/sample, median)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("Latency vs memory\n(marker size ~ 1/MSE; labels: K,L / MSE)")
    ax1.grid(True, which="both", alpha=0.3)

    # Memory vs MSE
    for r in rows:
        ax2.scatter(r["memory_bytes"], r["test_mse_uint8"],
                    color="tab:blue", alpha=0.7, s=80,
                    edgecolor="black", linewidth=0.5,
                    label="LUT-KAN" if r == rows[0] else None)
        ax2.annotate(f"K={r['K']},L={r['L']}",
                     (r["memory_bytes"], r["test_mse_uint8"]),
                     textcoords="offset points", xytext=(8, 0), fontsize=7)
    ax2.scatter(poly_row["memory_bytes"], poly_row["test_mse_uint8"],
                color="tab:green", alpha=0.7, s=120, marker="^",
                edgecolor="black", linewidth=0.5,
                label=f"PolyKAN deg={args.degree}", zorder=3)
    ax2.set_xlabel("Memory footprint (bytes)")
    ax2.set_ylabel("Test MSE")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("MSE vs memory")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Resource benchmark: {n_edges} edges ({args.hidden_dim} hidden), feynman_2d",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "pareto.png", dpi=130)
    plt.close()
    print(f"-> {out_dir / 'pareto.png'}")


if __name__ == "__main__":
    main()
