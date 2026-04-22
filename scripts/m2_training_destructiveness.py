#!/usr/bin/env python3
"""
M2 follow-up: training destroys a good poly-init.

From the first diagnostic we saw that:
  - poly_init LUT-KAN2 starts at MSE ~2e-3 (close to polynomial's ~5e-4)
  - After 300 epochs of training, it STAYS at ~2e-3 but the trace shows a
    spike up to ~0.5 in epochs 1-50
  - The best-val weights are saved in ~epoch 5-10, essentially the init

Question: is training actively HARMFUL, or just unnecessary at best?

Hypotheses:
  H1. LR too high: Adam at lr=1e-2 on the good init sends weights far from
      the minimum in the first few batches. Smaller lr should stabilize.
  H2. Second-diff reg too weak: the LUT drifts toward overfitting noise;
      curvature penalty should hold it near init.
  H3. Val-based selection is doing its job — the problem is that training
      just can't improve over poly-init in this setting.

We sweep (lr, lambda_2) on poly-init scenario with same dataset / seeds,
and report (a) initial MSE, (b) best-val MSE achieved, (c) final MSE,
(d) "improvement over init" = mse_init - mse_best.

Run: python scripts/m2_training_destructiveness.py
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

from lut_native import (  # noqa: E402
    KAN2TrainConfig,
    LUTKAN2Layer,
    PolyKAN2Layer,
    generate_data_2d,
    sample_polynomial_to_lut,
    train_kan2,
    train_poly_kan2,
)


def run_trial(poly_model, x_tr, y_tr, x_v, y_v, x_te, y_te,
              lr, lambda_2, epochs, seed, K=16, L=32, hidden=4):
    """Poly-init a fresh LUT-KAN2 and train it. Return MSEs."""
    model = LUTKAN2Layer(in_dim=2, hidden_dim=hidden, out_dim=1, K=K, L=L)

    # Poly-init
    with torch.no_grad():
        c1 = poly_model.c_l1.detach().cpu().numpy()
        for i in range(2):
            for h in range(hidden):
                lut = sample_polynomial_to_lut(c1[i, h], K=K, L=L)
                model.lut_l1.data[i, h] = torch.from_numpy(lut)
        c2 = poly_model.c_l2.detach().cpu().numpy()
        for h in range(hidden):
            lut = sample_polynomial_to_lut(c2[h, 0], K=K, L=L)
            model.lut_l2.data[h, 0] = torch.from_numpy(lut)

    # Pre-training MSE
    with torch.no_grad():
        pred_init = model(torch.from_numpy(x_te.astype(np.float32))).cpu().numpy().ravel()
    mse_init = float(np.mean((pred_init - y_te) ** 2))

    cfg = KAN2TrainConfig(
        lambda_1=0.0, lambda_2=lambda_2,
        lr=lr, epochs=epochs, batch_size=64,
        init_noise_std_absolute=0.01,   # small noise to keep near poly-init
        seed=seed, eval_every_epochs=max(1, epochs // 30),
    )
    res = train_kan2(model, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)

    return {
        "lr": lr, "lambda_2": lambda_2, "epochs": epochs, "seed": seed,
        "mse_init": mse_init,
        "mse_val_best": float(res.mse_val_at_best),
        "mse_test_best": float(res.mse_test_at_best),
        "mse_val_final": float(res.mse_val_final),
        "best_epoch": int(res.best_epoch),
        "improvement_abs": mse_init - float(res.mse_test_at_best),
        "improvement_rel": (mse_init - float(res.mse_test_at_best)) / mse_init,
        "trace_epochs": res.trace["epoch"],
        "trace_val_mse": res.trace["mse_val"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/M2_destruct")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--poly-epochs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data_2d(
        "feynman_2d", n_train=1000, n_val=400, n_test=400, seed=42,
    )

    # Train poly KAN2
    print(f"Training PolyKAN2 ({args.poly_epochs} epochs)...")
    poly = PolyKAN2Layer(in_dim=2, hidden_dim=4, out_dim=1, degree=8)
    poly_res = train_poly_kan2(poly, x_tr, y_tr, x_v, y_v, x_te, y_te,
                               lr=5e-3, epochs=args.poly_epochs, seed=args.seed)
    poly_test = poly_res["mse_test_at_best"]
    print(f"  PolyKAN2 test MSE: {poly_test:.3e}")

    lr_vals = [1e-4, 1e-3, 1e-2]
    l2_vals = [0.0, 0.01, 0.1, 1.0]

    print(f"\nSweeping poly-init LUT-KAN2, lr × lambda_2:")
    print(f"{'lr':>8} {'l2':>6} {'MSE init':>10} {'MSE best':>10} {'MSE final':>10} "
          f"{'best_ep':>8} {'improv%':>8}")

    results = []
    for lr in lr_vals:
        for l2 in l2_vals:
            t0 = time.time()
            r = run_trial(poly, x_tr, y_tr, x_v, y_v, x_te, y_te,
                          lr=lr, lambda_2=l2,
                          epochs=args.epochs, seed=args.seed)
            r["wall_time_sec"] = time.time() - t0
            results.append(r)
            print(f"{lr:>8.0e} {l2:>6g} {r['mse_init']:>10.2e} "
                  f"{r['mse_test_best']:>10.2e} {r['mse_val_final']:>10.2e} "
                  f"{r['best_epoch']:>8d} {r['improvement_rel']*100:>7.1f}%")

    out = {
        "config": {"epochs": args.epochs, "seed": args.seed},
        "poly_kan2_baseline_mse": poly_test,
        "lr_vals": lr_vals, "l2_vals": l2_vals,
        "results": results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_dir/'summary.json'}")

    # Plot: grid heatmap of best-vs-init
    grid_init = np.zeros((len(lr_vals), len(l2_vals)))
    grid_best = np.zeros_like(grid_init)
    grid_final = np.zeros_like(grid_init)
    for r in results:
        i = lr_vals.index(r["lr"]); j = l2_vals.index(r["lambda_2"])
        grid_init[i, j] = r["mse_init"]
        grid_best[i, j] = r["mse_test_best"]
        grid_final[i, j] = r["mse_val_final"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ["Init MSE", "Best-val MSE", "Final-epoch MSE"]
    grids = [grid_init, grid_best, grid_final]
    for ax, g, t in zip(axes, grids, titles):
        im = ax.imshow(np.log10(g), aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(l2_vals)))
        ax.set_xticklabels([f"{v:g}" for v in l2_vals])
        ax.set_yticks(range(len(lr_vals)))
        ax.set_yticklabels([f"{v:.0e}" for v in lr_vals])
        ax.set_xlabel(r"$\lambda_2$")
        ax.set_ylabel("lr")
        ax.set_title(t)
        for i in range(len(lr_vals)):
            for j in range(len(l2_vals)):
                ax.text(j, i, f"{g[i,j]:.1e}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if np.log10(g[i,j]) > np.log10(g).mean() else "black")
        plt.colorbar(im, ax=ax, label=r"$\log_{10}$ MSE")
    fig.suptitle(f"Does training improve on poly-init?  "
                 f"(dashed: PolyKAN2 test MSE = {poly_test:.1e})",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "grid.png", dpi=130)
    plt.close()
    print(f"Saved: {out_dir/'grid.png'}")

    # Also a "destructiveness": does final > init?
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for r in results:
        label = f"lr={r['lr']:.0e}, l2={r['lambda_2']:g}"
        ax.semilogy(r["trace_epochs"], r["trace_val_mse"],
                    label=label, alpha=0.7, linewidth=1.2)
    ax.axhline(poly_test, color="k", linestyle="--", alpha=0.6,
               linewidth=1.5, label=f"PolyKAN2 test MSE = {poly_test:.1e}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MSE")
    ax.set_title("All poly-init training traces — does training help at all?")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "traces.png", dpi=130)
    plt.close()
    print(f"Saved: {out_dir/'traces.png'}")

    # Headline
    print("\n" + "=" * 72)
    best_cfg = min(results, key=lambda r: r["mse_test_best"])
    print("HEADLINE")
    print(f"  PolyKAN2 (no LUT):               {poly_test:.3e}")
    print(f"  Best poly-init LUT-KAN2 across the grid:")
    print(f"      lr={best_cfg['lr']:.0e}, l2={best_cfg['lambda_2']:g}")
    print(f"      init MSE     = {best_cfg['mse_init']:.3e}")
    print(f"      best-val MSE = {best_cfg['mse_test_best']:.3e}")
    print(f"      improvement over init: "
          f"{best_cfg['improvement_rel']*100:+.1f}%")

    # Across the grid: how many configs improved init, how many worsened it
    n_helped = sum(1 for r in results if r["mse_test_best"] < r["mse_init"])
    n_neutral = sum(1 for r in results if abs(r["mse_test_best"] - r["mse_init"]) / r["mse_init"] < 0.05)
    n_hurt = sum(1 for r in results if r["mse_test_best"] > r["mse_init"] * 1.05)
    print(f"\n  Across {len(results)} configs:")
    print(f"      improved init:   {n_helped}")
    print(f"      within 5%:       {n_neutral}")
    print(f"      worsened init:   {n_hurt}")


if __name__ == "__main__":
    main()
