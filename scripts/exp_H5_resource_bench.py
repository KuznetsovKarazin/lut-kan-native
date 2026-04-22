#!/usr/bin/env python3
"""
Experiment H5: resource accounting + CPU latency.

Answers:
  - At the best direct-LUT config (single-edge, sine), how does it compare
    to the polynomial baseline on: bytes, ops, latency?
  - What do those numbers extrapolate to for multi-edge 2D KAN?

Outputs a table in CSV and a bar-plot figure.

Runtime: ~30 seconds.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lut_native import (  # noqa: E402
    eval_chebyshev,
    fit_chebyshev_ls,
    generate_data,
    lut_forward_numpy,
    lut_memory_bytes,
    lut_ops_per_sample,
    multi_edge_kan_ops,
    multi_edge_lut_memory_bytes,
    polynomial_memory_bytes,
    polynomial_ops_per_sample,
    sample_polynomial_to_lut,
    time_forward,
)


def main():
    out_dir = Path("results/H5_resource_bench")
    out_dir.mkdir(parents=True, exist_ok=True)

    x_tr, y_tr, _, _, _, _ = generate_data("sine", seed=42)
    coeffs_deg20 = fit_chebyshev_ls(x_tr, y_tr, degree=20)
    coeffs_deg16 = fit_chebyshev_ls(x_tr, y_tr, degree=16)

    # Benchmark inputs: larger N reduces per-call-overhead in latency estimate
    N = 20_000
    x_big = np.linspace(-1, 1, N, endpoint=False, dtype=np.float32)

    # Build single-edge LUT configs
    lut_k16_l32 = sample_polynomial_to_lut(coeffs_deg20, K=16, L=32)
    lut_k16_l16 = sample_polynomial_to_lut(coeffs_deg20, K=16, L=16)
    lut_k8_l16 = sample_polynomial_to_lut(coeffs_deg20, K=8, L=16)

    rows = []

    def bench(name, mem_bytes, ops, fn, extra=""):
        lat = time_forward(fn, x_big, n_reps=20)
        rows.append({
            "method": name,
            "memory_bytes": int(mem_bytes),
            "ops_float": int(ops.float_ops),
            "ops_int": int(ops.int_ops),
            "ops_total": int(ops.total_ops()),
            "reads_bytes": int(ops.memory_reads_bytes),
            "latency_ns_per_sample": float(lat["median_us_per_sample"] * 1000),
            "latency_std_ns": float(lat["std_us_per_sample"] * 1000),
            "extra": extra,
        })

    # --- Single-edge configs ---
    bench("poly_deg20_f32",
          polynomial_memory_bytes(20),
          polynomial_ops_per_sample(20),
          lambda x: eval_chebyshev(x, coeffs_deg20),
          extra="single-edge, sine")
    bench("poly_deg16_f32",
          polynomial_memory_bytes(16),
          polynomial_ops_per_sample(16),
          lambda x: eval_chebyshev(x, coeffs_deg16),
          extra="single-edge, sine (paper v2.1 default)")
    bench("lut_K16_L32_u8",
          lut_memory_bytes(16, 32),
          lut_ops_per_sample(16, 32),
          lambda x: lut_forward_numpy(x, lut_k16_l32),
          extra="single-edge, sine (paper v2.1 default)")
    bench("lut_K16_L16_u8",
          lut_memory_bytes(16, 16),
          lut_ops_per_sample(16, 16),
          lambda x: lut_forward_numpy(x, lut_k16_l16),
          extra="single-edge, sine")
    bench("lut_K8_L16_u8",
          lut_memory_bytes(8, 16),
          lut_ops_per_sample(8, 16),
          lambda x: lut_forward_numpy(x, lut_k8_l16),
          extra="single-edge, sine (MCU-friendly small)")

    # --- Multi-edge resource-only (no latency; model architecture) ---
    # Record as rows without latency
    configs_me = [
        ("poly_KAN_h4_d8_2d", 2, 4, 1, "poly", 8),
        ("poly_KAN_h8_d8_2d", 2, 8, 1, "poly", 8),
        ("lut_KAN_h4_K4L4_2d", 2, 4, 1, "lut", (4, 4)),
        ("lut_KAN_h4_K4L8_2d", 2, 4, 1, "lut", (4, 8)),
        ("lut_KAN_h8_K4L4_2d", 2, 8, 1, "lut", (4, 4)),
    ]
    for name, in_dim, hidden, out_dim, kind, detail in configs_me:
        if kind == "poly":
            deg = detail
            n_edges = in_dim * hidden + hidden * out_dim
            mem = n_edges * polynomial_memory_bytes(deg)
            ops = multi_edge_kan_ops(in_dim, hidden, out_dim, polynomial_ops_per_sample(deg))
        else:
            K, L = detail
            n_edges = in_dim * hidden + hidden * out_dim
            mem = multi_edge_lut_memory_bytes(n_edges, K, L)
            ops = multi_edge_kan_ops(in_dim, hidden, out_dim, lut_ops_per_sample(K, L))
        rows.append({
            "method": name,
            "memory_bytes": int(mem),
            "ops_float": int(ops.float_ops),
            "ops_int": int(ops.int_ops),
            "ops_total": int(ops.total_ops()),
            "reads_bytes": int(ops.memory_reads_bytes),
            "latency_ns_per_sample": None,
            "latency_std_ns": None,
            "extra": f"multi-edge {in_dim}->{hidden}->{out_dim}",
        })

    # --- Save ---
    with open(out_dir / "resource_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(out_dir / "resource_table.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Print table
    print(f"{'method':<25} {'bytes':>7} {'ops_f':>6} {'ops_i':>6} {'ops_tot':>8} {'lat_ns':>10}")
    print("-" * 70)
    for r in rows:
        lat_str = f"{r['latency_ns_per_sample']:.2f}" if r["latency_ns_per_sample"] is not None else "  -"
        print(f"{r['method']:<25} {r['memory_bytes']:>7} {r['ops_float']:>6} "
              f"{r['ops_int']:>6} {r['ops_total']:>8} {lat_str:>10}")

    # --- Plot: single-edge latency comparison ---
    single_edge = [r for r in rows if r["latency_ns_per_sample"] is not None]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    names = [r["method"] for r in single_edge]
    ax = axes[0]
    mems = [r["memory_bytes"] for r in single_edge]
    colors = ["tab:green" if "poly" in n else "tab:blue" for n in names]
    ax.bar(names, mems, color=colors)
    ax.set_ylabel("Memory (bytes)")
    ax.set_title("Memory footprint")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax = axes[1]
    ops_float = [r["ops_float"] for r in single_edge]
    ops_int = [r["ops_int"] for r in single_edge]
    x = np.arange(len(names))
    ax.bar(x - 0.2, ops_float, width=0.4, label="float ops", color="tab:orange")
    ax.bar(x + 0.2, ops_int, width=0.4, label="int ops", color="tab:purple")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, fontsize=8)
    ax.set_ylabel("Ops per sample")
    ax.set_title("Op counts (per sample)\nFloat ops are expensive on MCU")
    ax.legend(fontsize=9)

    ax = axes[2]
    lats = [r["latency_ns_per_sample"] for r in single_edge]
    lats_err = [r["latency_std_ns"] for r in single_edge]
    ax.bar(names, lats, yerr=lats_err, color=colors, capsize=5)
    ax.set_ylabel("CPU latency (ns/sample)")
    ax.set_title(f"Measured CPU latency (N={N}, 20 reps)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    fig.suptitle("Single-edge sine: resource profile\n"
                 "(Polynomial in green, LUT in blue)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "single_edge_profile.png", dpi=130)
    plt.close()

    print(f"\n  -> {out_dir / 'resource_table.csv'}")
    print(f"  -> {out_dir / 'resource_table.json'}")
    print(f"  -> {out_dir / 'single_edge_profile.png'}")


if __name__ == "__main__":
    main()
