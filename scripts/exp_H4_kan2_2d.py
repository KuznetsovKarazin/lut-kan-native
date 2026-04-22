#!/usr/bin/env python3
"""
Experiment H4: Multi-edge KAN on a genuine 2D target.

Target: feynman_2d: f(x, y) = sin(pi*x) + 0.5*cos(2*pi*x*y)
  - cross-variable structure (product term) means single-edge can't fit it
  - [2 -> H -> 1] KAN architecture is the minimal reasonable fit

Compares three methods at matched architecture:
  A) Polynomial KAN [2->H->1]: deg-d Chebyshev per edge, trained end-to-end
  B) Post-training LUT-KAN: take (A)'s coeffs, sample on LUT grid, quantize
  C) Direct LUT-KAN: initialize from (A)'s LUTs, continue training with l2 reg

Reports:
  - Test MSE (best-val) across seeds with paired-bootstrap CI
  - Memory footprint (bytes) at matched K, L, degree choices
  - Per-sample op counts (integer + float) for each method

Runtime: ~8 min CPU.
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
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lut_native import (  # noqa: E402
    KAN2TrainConfig,
    LUTKAN2Layer,
    chebyshev_basis,
    dequantize_lut,
    eval_chebyshev,
    generate_data_2d,
    lut_forward_numpy,
    lut_memory_bytes,
    lut_ops_per_sample,
    multi_edge_kan_ops,
    multi_edge_lut_memory_bytes,
    paired_bootstrap_ci,
    polynomial_memory_bytes,
    polynomial_ops_per_sample,
    quantize_lut_uint8_asym,
    sample_polynomial_to_lut,
    train_kan2,
)
from lut_native.kan2 import kan2_forward_numpy, polynomial_kan2_forward_numpy


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial KAN [in_dim -> hidden_dim -> out_dim] trained end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class PolyKAN2(nn.Module):
    """Two-layer polynomial KAN with Chebyshev basis and tanh squash.

    Forward matches polynomial_kan2_forward_numpy so that coefficients from
    here can be sampled to LUTs for direct comparison.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, degree: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.degree = degree
        # Small random init to break symmetry
        self.c_l1 = nn.Parameter(torch.randn(in_dim, hidden_dim, degree + 1) * 0.1)
        self.c_l2 = nn.Parameter(torch.randn(hidden_dim, out_dim, degree + 1) * 0.1)

    def _cheb_basis(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D) -> (N, D, degree+1)"""
        xn = torch.clamp(x, -1.0, 1.0)
        T = [torch.ones_like(xn), xn]
        for n in range(2, self.degree + 1):
            T.append(2 * xn * T[-1] - T[-2])
        return torch.stack(T, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = self._cheb_basis(x)                             # (N, in_dim, deg+1)
        z = torch.einsum("nid,ihd->nh", T, self.c_l1)       # (N, hidden)
        a = torch.tanh(z)
        Ta = self._cheb_basis(a)                            # (N, hidden, deg+1)
        y = torch.einsum("nhd,hod->no", Ta, self.c_l2)      # (N, out_dim)
        return y


def train_poly_kan2(
    in_dim, hidden_dim, out_dim, degree,
    x_tr, y_tr, x_v, y_v, x_te, y_te,
    lr=5e-3, epochs=2000, batch_size=128, seed=0, eval_every=50,
):
    torch.manual_seed(seed)
    model = PolyKAN2(in_dim, hidden_dim, out_dim, degree)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).view(-1, out_dim)
    y_v_t = torch.from_numpy(y_v.astype(np.float32)).view(-1, out_dim)
    y_e_t = torch.from_numpy(y_te.astype(np.float32)).view(-1, out_dim)
    x_tr_t = torch.from_numpy(x_tr.astype(np.float32))
    x_v_t = torch.from_numpy(x_v.astype(np.float32))
    x_e_t = torch.from_numpy(x_te.astype(np.float32))
    N = x_tr_t.shape[0]

    best_val = float("inf")
    best_c1 = model.c_l1.detach().cpu().numpy().copy()
    best_c2 = model.c_l2.detach().cpu().numpy().copy()
    best_epoch = 0

    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            optim.zero_grad()
            loss = ((model(x_tr_t[idx]) - y_tr_t[idx]) ** 2).mean()
            loss.backward()
            optim.step()
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            with torch.no_grad():
                mse_v = ((model(x_v_t) - y_v_t) ** 2).mean().item()
            if mse_v < best_val:
                best_val = mse_v
                best_c1 = model.c_l1.detach().cpu().numpy().copy()
                best_c2 = model.c_l2.detach().cpu().numpy().copy()
                best_epoch = ep + 1

    # Final test MSE with best coeffs
    y_pred = polynomial_kan2_forward_numpy(x_te, best_c1, best_c2)
    mse_test = float(np.mean((y_pred.ravel() - y_te.ravel()) ** 2))
    return {
        "coeffs_l1": best_c1, "coeffs_l2": best_c2,
        "test_mse": mse_test, "val_mse": best_val, "best_epoch": best_epoch,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Post-training LUT-KAN: sample poly-KAN coeffs onto LUT grid
# ─────────────────────────────────────────────────────────────────────────────

def luts_from_polynomial_kan(coeffs_l1, coeffs_l2, K, L,
                             x_min_l1=-1.0, x_max_l1=1.0):
    """Sample polynomial KAN coefficients onto LUT grids.

    coeffs_l1: (in_dim, hidden_dim, degree+1)
    coeffs_l2: (hidden_dim, out_dim, degree+1)
    Returns (lut_l1, lut_l2) with shapes (in, hidden, K, L), (hidden, out, K, L).
    Layer 2 LUTs are on [-1, 1] (after tanh).
    """
    in_dim, hidden_dim, _ = coeffs_l1.shape
    _, out_dim, _ = coeffs_l2.shape
    lut_l1 = np.empty((in_dim, hidden_dim, K, L), dtype=np.float32)
    lut_l2 = np.empty((hidden_dim, out_dim, K, L), dtype=np.float32)
    for i in range(in_dim):
        for h in range(hidden_dim):
            lut_l1[i, h] = sample_polynomial_to_lut(
                coeffs_l1[i, h], K=K, L=L, x_min=x_min_l1, x_max=x_max_l1)
    for h in range(hidden_dim):
        for o in range(out_dim):
            lut_l2[h, o] = sample_polynomial_to_lut(
                coeffs_l2[h, o], K=K, L=L, x_min=-1.0, x_max=1.0)
    return lut_l1, lut_l2


def quantize_kan_luts(lut_l1, lut_l2):
    """Apply per-segment uint8 quantization to all edge LUTs, return dequantized
    float arrays (what the MCU kernel would actually evaluate)."""
    def _process(lut_block):
        out = np.empty_like(lut_block)
        a, b, K, L = lut_block.shape
        for i in range(a):
            for j in range(b):
                q, s, m = quantize_lut_uint8_asym(lut_block[i, j])
                out[i, j] = dequantize_lut(q, s, m)
        return out
    return _process(lut_l1), _process(lut_l2)


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/H4_kan2_2d")
    ap.add_argument("--poly-epochs", type=int, default=1500)
    ap.add_argument("--lut-epochs", type=int, default=800)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--hidden-dim", type=int, default=4)
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--lambda-2", type=float, default=0.1,
                    help="curvature penalty for direct-LUT (reduced from 1.0 "
                         "because 8 edges sum, per-LUT reg has stronger effect)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"H4: multi-edge KAN on 2D target")
    print(f"  Architecture: [2 -> {args.hidden_dim} -> 1], K={args.K}, L={args.L}")
    print(f"  Poly degree (for baseline): {args.degree}")
    print(f"  Direct-LUT lambda_2: {args.lambda_2}")
    print(f"  Seeds: {args.seeds}")
    print("=" * 72)

    x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data_2d("feynman_2d", seed=42)

    # ─── A. Polynomial KAN baseline (5 seeds, pick best-val per seed) ──────
    print("\n[A] Polynomial KAN training")
    poly_results = []
    for seed in args.seeds:
        t0 = time.time()
        r = train_poly_kan2(
            in_dim=2, hidden_dim=args.hidden_dim, out_dim=1, degree=args.degree,
            x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
            lr=5e-3, epochs=args.poly_epochs, batch_size=128, seed=seed,
        )
        poly_results.append(r)
        print(f"    seed={seed}  test_MSE={r['test_mse']:.3e}  "
              f"best_ep={r['best_epoch']}  dt={time.time()-t0:.1f}s")

    poly_mses = np.array([r["test_mse"] for r in poly_results])
    best_poly = min(poly_results, key=lambda r: r["val_mse"])

    # ─── B. Post-training LUT-KAN ─────────────────────────────────────────
    print("\n[B] Post-training LUT-KAN (sample best poly-KAN's coefficients)")
    coeffs_l1 = best_poly["coeffs_l1"]
    coeffs_l2 = best_poly["coeffs_l2"]
    post_lut_l1, post_lut_l2 = luts_from_polynomial_kan(
        coeffs_l1, coeffs_l2, K=args.K, L=args.L)
    y_post_fp = kan2_forward_numpy(x_te, post_lut_l1, post_lut_l2)
    mse_post_fp = float(np.mean((y_post_fp.ravel() - y_te.ravel()) ** 2))

    post_lut_l1_q, post_lut_l2_q = quantize_kan_luts(post_lut_l1, post_lut_l2)
    y_post_u8 = kan2_forward_numpy(x_te, post_lut_l1_q, post_lut_l2_q)
    mse_post_u8 = float(np.mean((y_post_u8.ravel() - y_te.ravel()) ** 2))
    print(f"    test_MSE (fp):   {mse_post_fp:.3e}")
    print(f"    test_MSE (uint8): {mse_post_u8:.3e}")

    # ─── C. Direct-LUT-KAN ─────────────────────────────────────────────────
    print(f"\n[C] Direct-LUT-KAN (l2={args.lambda_2})")
    direct_results = []
    for seed in args.seeds:
        t0 = time.time()
        model = LUTKAN2Layer(
            in_dim=2, hidden_dim=args.hidden_dim, out_dim=1,
            K=args.K, L=args.L,
        )
        # Initialize from post-training LUTs (this gives direct-LUT a fair
        # starting point; without it, training would have to re-discover
        # basic structure from scratch).
        model.init_layer1_from_arrays(post_lut_l1)
        model.init_layer2_from_arrays(post_lut_l2)
        cfg = KAN2TrainConfig(
            lambda_1=0.0, lambda_2=args.lambda_2, lr=5e-4,
            epochs=args.lut_epochs, batch_size=128,
            init_noise_std_absolute=0.001,  # tiny — we already have a great init
            seed=seed, eval_every_epochs=15,
        )
        res = train_kan2(model, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)
        direct_results.append(res)
        print(f"    seed={seed}  test_MSE={res.mse_test_at_best:.3e}  "
              f"best_ep={res.best_epoch}  dt={time.time()-t0:.1f}s")

    direct_mses = np.array([r.mse_test_at_best for r in direct_results])

    # Also: uint8-quantized direct-LUT
    direct_u8_mses = []
    for res in direct_results:
        l1_q, l2_q = quantize_kan_luts(res.lut_l1_best, res.lut_l2_best)
        y = kan2_forward_numpy(x_te, l1_q, l2_q)
        direct_u8_mses.append(float(np.mean((y.ravel() - y_te.ravel()) ** 2)))
    direct_u8_mses = np.array(direct_u8_mses)

    # ─── Statistics ────────────────────────────────────────────────────────
    # Pair direct-LUT seeds with post-training-LUT baseline (deterministic)
    ratio_post_over_direct = paired_bootstrap_ci(
        np.full_like(direct_mses, mse_post_fp), direct_mses,
        n_boot=10_000, seed=0,
    )
    # Direct-LUT vs polynomial KAN, paired by seed index
    ratio_poly_over_direct = paired_bootstrap_ci(
        poly_mses, direct_mses, n_boot=10_000, seed=0,
    )

    # ─── Resource accounting ──────────────────────────────────────────────
    n_edges = 2 * args.hidden_dim + args.hidden_dim * 1  # L1 + L2 edges
    # Polynomial: (deg+1) float32 per edge
    poly_mem = n_edges * polynomial_memory_bytes(args.degree, dtype="float32")
    poly_ops = multi_edge_kan_ops(
        in_dim=2, hidden_dim=args.hidden_dim, out_dim=1,
        eval_ops_per_edge=polynomial_ops_per_sample(args.degree),
    )
    lut_mem = multi_edge_lut_memory_bytes(
        n_edges=n_edges, K=args.K, L=args.L,
        dtype="uint8", meta_dtype="float16",
    )
    lut_ops = multi_edge_kan_ops(
        in_dim=2, hidden_dim=args.hidden_dim, out_dim=1,
        eval_ops_per_edge=lut_ops_per_sample(args.K, args.L, dtype="uint8"),
    )

    # ─── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Method':<36} {'MSE (5 seeds)':<28} {'Memory':<10} {'Ops':<12}")
    print("-" * 86)
    print(f"{'Polynomial KAN (deg=' + str(args.degree) + ')':<36} "
          f"{poly_mses.mean():.2e} +/- {poly_mses.std(ddof=1):.1e}    "
          f"{poly_mem:>6} B  {poly_ops.float_ops + poly_ops.int_ops:>6} ops")
    print(f"{'Post-training LUT-KAN (fp)':<36} "
          f"{mse_post_fp:.2e} (deterministic)        "
          f"{n_edges * args.K * args.L * 4:>6} B  {lut_ops.float_ops + lut_ops.int_ops:>6} ops")
    print(f"{'Post-training LUT-KAN (uint8)':<36} "
          f"{mse_post_u8:.2e} (deterministic)        "
          f"{lut_mem:>6} B  {lut_ops.float_ops + lut_ops.int_ops:>6} ops")
    print(f"{'Direct LUT-KAN (fp)':<36} "
          f"{direct_mses.mean():.2e} +/- {direct_mses.std(ddof=1):.1e}    "
          f"{n_edges * args.K * args.L * 4:>6} B  {lut_ops.float_ops + lut_ops.int_ops:>6} ops")
    print(f"{'Direct LUT-KAN (uint8)':<36} "
          f"{direct_u8_mses.mean():.2e} +/- {direct_u8_mses.std(ddof=1):.1e}    "
          f"{lut_mem:>6} B  {lut_ops.float_ops + lut_ops.int_ops:>6} ops")
    print()
    print(f"Direct/Post ratio (fp):    "
          f"{ratio_post_over_direct['ratio_mean']:.2f}x  "
          f"[{ratio_post_over_direct['ratio_ci_lower']:.2f}, "
          f"{ratio_post_over_direct['ratio_ci_upper']:.2f}]  (higher = direct better)")
    print(f"Direct/Poly-KAN ratio:     "
          f"{ratio_poly_over_direct['ratio_mean']:.2f}x  "
          f"[{ratio_poly_over_direct['ratio_ci_lower']:.2f}, "
          f"{ratio_poly_over_direct['ratio_ci_upper']:.2f}]  (higher = direct better)")

    # ─── Save JSON ────────────────────────────────────────────────────────
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "config": {
                "target": "feynman_2d",
                "architecture": f"2->{args.hidden_dim}->1",
                "degree": args.degree, "K": args.K, "L": args.L,
                "lambda_2": args.lambda_2,
                "poly_epochs": args.poly_epochs,
                "lut_epochs": args.lut_epochs,
                "seeds": args.seeds,
                "n_edges": n_edges,
            },
            "results": {
                "polynomial_kan": {
                    "test_mse_mean": float(poly_mses.mean()),
                    "test_mse_std": float(poly_mses.std(ddof=1)),
                    "test_mse_values": poly_mses.tolist(),
                    "memory_bytes": poly_mem,
                    "ops_per_sample": {
                        "int_ops": poly_ops.int_ops,
                        "float_ops": poly_ops.float_ops,
                    },
                },
                "post_training_lut_kan": {
                    "test_mse_fp": mse_post_fp,
                    "test_mse_uint8": mse_post_u8,
                    "memory_bytes_uint8": lut_mem,
                    "memory_bytes_fp32": n_edges * args.K * args.L * 4,
                    "ops_per_sample": {
                        "int_ops": lut_ops.int_ops,
                        "float_ops": lut_ops.float_ops,
                    },
                },
                "direct_lut_kan": {
                    "test_mse_fp_mean": float(direct_mses.mean()),
                    "test_mse_fp_std": float(direct_mses.std(ddof=1)),
                    "test_mse_fp_values": direct_mses.tolist(),
                    "test_mse_uint8_mean": float(direct_u8_mses.mean()),
                    "test_mse_uint8_std": float(direct_u8_mses.std(ddof=1)),
                    "test_mse_uint8_values": direct_u8_mses.tolist(),
                    "memory_bytes_uint8": lut_mem,
                    "memory_bytes_fp32": n_edges * args.K * args.L * 4,
                    "ops_per_sample": {
                        "int_ops": lut_ops.int_ops,
                        "float_ops": lut_ops.float_ops,
                    },
                },
            },
            "statistics": {
                "ratio_post_over_direct_fp": ratio_post_over_direct,
                "ratio_poly_over_direct_fp": ratio_poly_over_direct,
            },
        }, f, indent=2)
    print(f"\n-> {out_dir / 'summary.json'}")

    # ─── Plot ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: bar chart of MSEs
    methods = ["PolyKAN", "PostLUT-KAN\n(fp)", "PostLUT-KAN\n(uint8)",
               "DirectLUT-KAN\n(fp)", "DirectLUT-KAN\n(uint8)"]
    values = [poly_mses.mean(), mse_post_fp, mse_post_u8,
              direct_mses.mean(), direct_u8_mses.mean()]
    errors = [poly_mses.std(ddof=1), 0, 0,
              direct_mses.std(ddof=1), direct_u8_mses.std(ddof=1)]
    colors = ["tab:green", "tab:red", "firebrick", "tab:blue", "steelblue"]
    x = np.arange(len(methods))
    ax1.bar(x, values, yerr=errors, capsize=4, color=colors)
    for i, v in enumerate(values):
        ax1.text(i, v * 1.15, f"{v:.1e}", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=8)
    ax1.set_ylabel("Test MSE")
    ax1.set_yscale("log")
    ax1.set_title(f"2D target (feynman_2d), [2->{args.hidden_dim}->1], K={args.K}, L={args.L}\n"
                  f"Error bars: std over {len(args.seeds)} seeds")
    ax1.grid(True, axis="y", which="both", alpha=0.3)

    # Right: memory vs MSE scatter
    ax2.scatter([poly_mem], [poly_mses.mean()], color="tab:green", s=120, marker="^",
                label=f"PolyKAN (deg={args.degree})", zorder=3)
    ax2.scatter([lut_mem], [mse_post_u8], color="tab:red", s=120, marker="o",
                label="PostLUT-KAN (uint8)", zorder=3)
    ax2.scatter([lut_mem], [direct_u8_mses.mean()], color="tab:blue", s=120, marker="s",
                label="DirectLUT-KAN (uint8)", zorder=3)
    ax2.errorbar([poly_mem], [poly_mses.mean()], yerr=[poly_mses.std(ddof=1)],
                 color="tab:green", capsize=4)
    ax2.errorbar([lut_mem], [direct_u8_mses.mean()], yerr=[direct_u8_mses.std(ddof=1)],
                 color="tab:blue", capsize=4)
    ax2.set_xlabel("Memory footprint (bytes)")
    ax2.set_ylabel("Test MSE")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_title(f"MSE vs memory budget")
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=130)
    plt.close()
    print(f"-> {out_dir / 'comparison.png'}")

    return poly_mses, direct_mses, mse_post_fp


if __name__ == "__main__":
    main()
