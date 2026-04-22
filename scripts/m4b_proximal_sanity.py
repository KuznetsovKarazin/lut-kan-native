#!/usr/bin/env python3
"""
M4b: Quick sanity — does adding a proximal penalty on Delta help?

Prediction (based on M4a findings): NO, because:
  - M4a showed trust-region size α ∈ [0.05, 1.0] gives identical best-val
    MSE (differing only in path, not destination).
  - Proximal penalty λ·||Δ||² is another mechanism for the same thing
    (gravitate toward init).

We run 1 task (feynman_2d) × 1 seed with best M4a config (α=0.1) and
sweep lambda_init ∈ {0, 0.1, 1.0}. If all three give the same best-val
MSE within noise, we confirm the prediction and move on. If unexpectedly
one helps, we flag for follow-up.

Runtime: ~1 min CPU.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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


def main():
    out_dir = Path("results/M4b_proximal_sanity")
    out_dir.mkdir(parents=True, exist_ok=True)

    K, L, hidden = 16, 32, 4
    alpha = 0.1   # best from M4a
    lr = 5e-4
    epochs = 200
    seeds = [0, 1]
    task_name = "feynman_2d"

    print(f"M4b sanity: feynman_2d, alpha={alpha}, lr={lr}, epochs={epochs}")

    x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data_2d(
        "feynman_2d", n_train=1000, n_val=400, n_test=400, seed=42,
    )

    # Train poly per seed
    poly_coeffs = []
    poly_mses = []
    for s in seeds:
        pres = train_poly_kan2(
            in_dim=2, hidden_dim=hidden, out_dim=1, degree=8,
            x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
            lr=5e-3, epochs=300, batch_size=128, seed=s,
        )
        poly_coeffs.append((pres["coeffs_l1"], pres["coeffs_l2"]))
        poly_mses.append(pres["test_mse"])

    poly_mean = float(np.mean(poly_mses))
    print(f"  PolyKAN2 baseline mean over {len(seeds)} seeds: {poly_mean:.3e}")

    lambda_vals = [0.0, 0.1, 1.0]
    print(f"\n  {'λ_init':>8} {'best MSE (mean±std)':>22}  {'vs Poly':>10} "
          f"{'|Δ|₂ final L1':>14} {'|Δ|₂ final L2':>14}")

    results = []
    for lam in lambda_vals:
        best_mses = []
        delta_l1_norms = []
        delta_l2_norms = []
        t0 = time.time()
        for seed, (c1, c2) in zip(seeds, poly_coeffs):
            model = _init_from_poly(c1, c2, in_dim=2, hidden=hidden,
                                    K=K, L=L, alpha=alpha)
            cfg = ResidualTrainConfig(
                lambda_1=0.0, lambda_2=0.0,
                lambda_init_anchor_l1=lam,
                lambda_init_anchor_l2=lam,
                lr_l1=lr, lr_l2=lr,
                epochs=epochs, batch_size=128, seed=seed,
                eval_every_epochs=10,
            )
            res = train_residual_kan2(model, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)
            best_mses.append(res.mse_test_at_best)
            delta_l1_norms.append(float(np.linalg.norm(res.delta_l1_final)))
            delta_l2_norms.append(float(np.linalg.norm(res.delta_l2_final)))

        arr = np.array(best_mses)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(seeds) > 1 else 0.0
        print(f"  {lam:>8.2f} {mean:>10.3e} ± {std:>7.1e} "
              f"{poly_mean/mean:>9.2f}x "
              f"{np.mean(delta_l1_norms):>14.3f} {np.mean(delta_l2_norms):>14.3f}  "
              f"(dt={time.time()-t0:.1f}s)")
        results.append({
            "lambda_init": lam,
            "best_mse_per_seed": best_mses,
            "best_mse_mean": mean,
            "best_mse_std": std,
            "delta_l1_norm_mean": float(np.mean(delta_l1_norms)),
            "delta_l2_norm_mean": float(np.mean(delta_l2_norms)),
            "vs_poly": poly_mean / mean,
        })

    # Decision rule: call it null if all three are within ~15% of each other
    means = [r["best_mse_mean"] for r in results]
    spread = max(means) / min(means)
    verdict = "null (proximal does not change best-val MSE)" if spread < 1.15 \
        else "unexpected — spread > 15%, follow up"
    print(f"\nSpread: {spread:.3f}x  →  {verdict}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "config": {"task": task_name, "alpha": alpha, "lr": lr,
                       "epochs": epochs, "seeds": seeds,
                       "lambda_vals": lambda_vals},
            "poly_baseline_mean": poly_mean,
            "results": results,
            "spread": spread,
            "verdict": verdict,
        }, f, indent=2)
    print(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
