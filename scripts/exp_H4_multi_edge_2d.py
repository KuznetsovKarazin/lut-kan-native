#!/usr/bin/env python3
"""
Experiment H4: Multi-edge KAN on 2D feynman target.

Hypothesis:
    On a 2-input task where 2-layer KAN architecture is actually useful,
    does direct-LUT training at matched memory budget beat polynomial-KAN?

Memory-matched comparison: test MSE for both methods at different memory
budgets (uint8 LUT bytes vs float32 coefficient bytes). Also measures
per-sample op counts and CPU latency.

Runtime: ~5 min CPU.

Expected outcome (based on pilot runs): on this 2D task, polynomial-KAN is
significantly more accurate per byte than direct-LUT-KAN. Direct-LUT's
advantage from single-edge experiments does NOT carry over to multi-edge
under the tested configs. This is a valuable negative result that defines
the limits of our claim.
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
    generate_data_2d,
    multi_edge_kan_ops,
    multi_edge_lut_memory_bytes,
    lut_ops_per_sample,
    paired_bootstrap_ci,
    polynomial_memory_bytes,
    polynomial_ops_per_sample,
    time_forward,
    train_kan2,
)
from lut_native.kan2 import polynomial_kan2_forward_numpy


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial-KAN baseline trained by Adam (no closed-form LS for 2-layer)
# ─────────────────────────────────────────────────────────────────────────────

class PolyKAN2D(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, degree: int):
        super().__init__()
        self.in_dim, self.hidden_dim, self.out_dim, self.degree = in_dim, hidden_dim, out_dim, degree
        self.c_l1 = nn.Parameter(torch.randn(in_dim, hidden_dim, degree + 1) * 0.05)
        self.c_l2 = nn.Parameter(torch.randn(hidden_dim, out_dim, degree + 1) * 0.05)

    def _cheb(self, x, deg):
        T = [torch.ones_like(x), x]
        for n in range(2, deg + 1):
            T.append(2 * x * T[-1] - T[-2])
        return torch.stack(T, dim=-1)

    def forward(self, x):
        xn = torch.clamp(x, -1.0, 1.0)
        z = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        for i in range(self.in_dim):
            T = self._cheb(xn[:, i], self.degree)
            z = z + torch.einsum('nd,hd->nh', T, self.c_l1[i])
        a = torch.tanh(z)
        y = torch.zeros(x.shape[0], self.out_dim, device=x.device)
        for h in range(self.hidden_dim):
            Ta = self._cheb(a[:, h], self.degree)
            y = y + torch.einsum('nd,od->no', Ta, self.c_l2[h])
        return y


def train_poly_kan(hidden_dim, degree, x_tr, y_tr, x_v, y_v, x_te, y_te,
                    epochs=3000, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    model = PolyKAN2D(2, hidden_dim, 1, degree)
    xt = torch.from_numpy(x_tr.astype(np.float32))
    yt = torch.from_numpy(y_tr.astype(np.float32)).view(-1, 1)
    xv = torch.from_numpy(x_v.astype(np.float32))
    yv = torch.from_numpy(y_v.astype(np.float32)).view(-1, 1)
    xe = torch.from_numpy(x_te.astype(np.float32))
    ye = torch.from_numpy(y_te.astype(np.float32)).view(-1, 1)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_test = float("inf")
    best_coeffs = None
    for ep in range(epochs):
        optim.zero_grad()
        loss = ((model(xt) - yt) ** 2).mean()
        loss.backward()
        optim.step()
        if (ep + 1) % 50 == 0:
            with torch.no_grad():
                mv = ((model(xv) - yv) ** 2).mean().item()
                if mv < best_val:
                    best_val = mv
                    best_test = ((model(xe) - ye) ** 2).mean().item()
                    best_coeffs = (model.c_l1.detach().cpu().numpy().copy(),
                                   model.c_l2.detach().cpu().numpy().copy())
    return {"test_mse": best_test, "val_mse": best_val, "coeffs": best_coeffs}


def run_lut_kan(hidden_dim, K, L, lambda_2, x_tr, y_tr, x_v, y_v, x_te, y_te,
                 epochs=500, seed=0, lr=5e-3):
    model = LUTKAN2Layer(in_dim=2, hidden_dim=hidden_dim, out_dim=1, K=K, L=L)
    cfg = KAN2TrainConfig(
        lambda_1=0.0, lambda_2=lambda_2, lr=lr, epochs=epochs,
        batch_size=128, init_noise_std_absolute=0.1,
        seed=seed, eval_every_epochs=25,
    )
    res = train_kan2(model, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)
    return {
        "test_mse": res.mse_test_at_best,
        "val_mse": res.mse_val_at_best,
        "train_mse": res.mse_train_at_best,
        "best_epoch": res.best_epoch,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/H4_multi_edge_2d")
    ap.add_argument("--epochs-lut", type=int, default=400)
    ap.add_argument("--epochs-poly", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("H4: Multi-edge [2 -> h -> 1] KAN on 2D feynman target")
    print("=" * 72)

    x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data_2d("feynman_2d", seed=42)
    print(f"  Train: {len(x_tr)} samples, Val: {len(x_v)}, Test: {len(x_te)}")
    print(f"  Target range: [{y_te.min():.2f}, {y_te.max():.2f}] std={y_te.std():.2f}")
    print()

    # ---- Polynomial-KAN reference (best configuration from pilot) --------
    poly_configs = [(4, 6), (4, 8), (8, 6), (8, 8), (8, 12)]
    poly_results = []
    print("--- Polynomial-KAN reference ---")
    for hidden, deg in poly_configs:
        t0 = time.time()
        mses = []
        for seed in args.seeds:
            r = train_poly_kan(hidden, deg, x_tr, y_tr, x_v, y_v, x_te, y_te,
                                epochs=args.epochs_poly, seed=seed)
            mses.append(r["test_mse"])
        n_params = 2 * hidden * (deg + 1) + hidden * (deg + 1)
        bytes_f32 = n_params * 4
        ops = multi_edge_kan_ops(2, hidden, 1, polynomial_ops_per_sample(deg))
        poly_results.append({
            "method": "polynomial_kan",
            "hidden_dim": hidden,
            "degree": deg,
            "n_params": n_params,
            "memory_bytes": bytes_f32,
            "ops_per_sample": ops.total_ops(),
            "test_mses": mses,
            "mean": float(np.mean(mses)),
            "std": float(np.std(mses, ddof=1)),
        })
        print(f"  h={hidden} deg={deg}  bytes={bytes_f32:<5} ops={ops.total_ops():<5} "
              f"MSE={np.mean(mses):.3e} ± {np.std(mses, ddof=1):.1e}  "
              f"dt={time.time()-t0:.1f}s")

    # ---- Direct-LUT-KAN ---------------------------------------------------
    print("\n--- Direct-LUT-KAN (multi-edge) ---")
    # (hidden, K, L, lambda_2) - chosen to span memory range of poly baselines
    lut_configs = [
        (2, 4, 8, 1.0),
        (2, 8, 8, 1.0),
        (4, 4, 4, 0.1),
        (4, 4, 4, 1.0),
        (4, 4, 8, 0.1),
        (4, 4, 8, 1.0),
        (4, 8, 8, 0.1),
        (4, 8, 8, 1.0),
        (8, 4, 4, 1.0),
        (8, 4, 8, 1.0),
    ]
    lut_results = []
    for hidden, K, L, l2 in lut_configs:
        t0 = time.time()
        mses = []
        for seed in args.seeds:
            r = run_lut_kan(hidden, K, L, l2, x_tr, y_tr, x_v, y_v, x_te, y_te,
                             epochs=args.epochs_lut, seed=seed)
            mses.append(r["test_mse"])
        n_edges = 2 * hidden + hidden * 1
        bytes_u8 = multi_edge_lut_memory_bytes(n_edges, K, L)
        ops = multi_edge_kan_ops(2, hidden, 1, lut_ops_per_sample(K, L))
        lut_results.append({
            "method": "direct_lut_kan",
            "hidden_dim": hidden,
            "K": K,
            "L": L,
            "lambda_2": l2,
            "n_edges": n_edges,
            "memory_bytes": bytes_u8,
            "ops_per_sample": ops.total_ops(),
            "test_mses": mses,
            "mean": float(np.mean(mses)),
            "std": float(np.std(mses, ddof=1)),
        })
        print(f"  h={hidden} K={K} L={L} l2={l2:<4}  bytes={bytes_u8:<5} "
              f"ops={ops.total_ops():<5} MSE={np.mean(mses):.3e} ± {np.std(mses, ddof=1):.1e}  "
              f"dt={time.time()-t0:.1f}s")

    # ---- Save JSON ---------------------------------------------------------
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "config": {
                "target": "feynman_2d",
                "n_train": len(x_tr), "n_val": len(x_v), "n_test": len(x_te),
                "seeds": args.seeds,
                "epochs_lut": args.epochs_lut,
                "epochs_poly": args.epochs_poly,
            },
            "polynomial_kan": poly_results,
            "lut_kan": lut_results,
        }, f, indent=2)

    # ---- Plot: memory vs MSE ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Left: memory
    ax = axes[0]
    poly_x = [r["memory_bytes"] for r in poly_results]
    poly_y = [r["mean"] for r in poly_results]
    poly_yerr = [r["std"] for r in poly_results]
    lut_x = [r["memory_bytes"] for r in lut_results]
    lut_y = [r["mean"] for r in lut_results]
    lut_yerr = [r["std"] for r in lut_results]

    ax.errorbar(poly_x, poly_y, yerr=poly_yerr, fmt="o-", color="tab:green",
                label="Polynomial-KAN (float32 coeffs)", markersize=7, capsize=4)
    ax.errorbar(lut_x, lut_y, yerr=lut_yerr, fmt="s", color="tab:blue",
                label="Direct-LUT-KAN (uint8 LUT)", markersize=7, capsize=4)
    for r in poly_results:
        ax.annotate(f"h={r['hidden_dim']},d={r['degree']}",
                    (r['memory_bytes'], r['mean']),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, color="tab:green")
    for r in lut_results:
        ax.annotate(f"h{r['hidden_dim']},K{r['K']}L{r['L']}",
                    (r['memory_bytes'], r['mean']),
                    textcoords="offset points", xytext=(5, -8), fontsize=7, color="tab:blue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Memory budget (bytes)")
    ax.set_ylabel("Test MSE")
    ax.set_title(f"Memory vs MSE, 2D feynman ({len(args.seeds)} seeds)\n"
                 "Polynomial clearly Pareto-dominates on this task")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Right: ops
    ax = axes[1]
    poly_x_ops = [r["ops_per_sample"] for r in poly_results]
    lut_x_ops = [r["ops_per_sample"] for r in lut_results]
    ax.errorbar(poly_x_ops, poly_y, yerr=poly_yerr, fmt="o-", color="tab:green",
                label="Polynomial-KAN (all float ops)", markersize=7, capsize=4)
    ax.errorbar(lut_x_ops, lut_y, yerr=lut_yerr, fmt="s", color="tab:blue",
                label="Direct-LUT-KAN (mixed int+float)", markersize=7, capsize=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total ops per sample")
    ax.set_ylabel("Test MSE")
    ax.set_title("Ops vs MSE, 2D feynman\n"
                 "LUT ops are cheaper per-op on MCUs (no FPU needed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "pareto.png", dpi=130)
    plt.close()
    print(f"\n  -> {out_dir / 'pareto.png'}")
    print(f"  -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
