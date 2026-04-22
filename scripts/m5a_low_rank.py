#!/usr/bin/env python3
"""
M5a: Low-rank residual sweep.

THE FINAL MULTI-EDGE EXPERIMENT per the agreed stop criterion.

Hypothesis: unstructured residual (M4) didn't help because per-cell
perturbations are too noisy. Low-rank residual (Δ = U V^T per edge) forces
globally-coherent perturbations which may be what the task actually benefits
from.

Sweep: per-edge rank r ∈ {1, 2, 4, 8}.
  - rank 1:    48 params/edge  vs full 512  (90% reduction)
  - rank 2:    96 params/edge  vs full 512  (81% reduction)
  - rank 4:   192 params/edge  vs full 512  (63% reduction)
  - rank 8:   384 params/edge  vs full 512  (25% reduction)

Baseline: M4c best (full delta, L1_slower) = 1.22× on feynman_2d.

Stop check:
  - If best rank gives statistically significant gain over M4c's 1.22× → explore further
  - If flat or worse → close multi-edge, switch to single-edge strengthening

Setup:
  - alpha = 0.1 (from M4a)
  - lr_l1 = 1e-4, lr_l2 = 5e-4 (from M4c)
  - 200 epochs, 3 seeds
  - Both tasks

Output: results/M5a_low_rank/{composition_1d,feynman_2d}/{summary.json,plot.png}
Runtime: ~12 min CPU.
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
    LowRankResidualLUTKAN2Layer,
    LowRankTrainConfig,
    ResidualLUTKAN2Layer,
    ResidualTrainConfig,
    generate_data_2d,
    sample_polynomial_to_lut,
    train_low_rank_kan2,
    train_residual_kan2,
)
from exp_H4_kan2_2d import train_poly_kan2
from m2b_composition import generate_data_1d


def _build_init_luts(poly_c1, poly_c2, in_dim, hidden, K, L):
    init_l1 = np.zeros((in_dim, hidden, K, L), dtype=np.float32)
    for i in range(in_dim):
        for h in range(hidden):
            init_l1[i, h] = sample_polynomial_to_lut(poly_c1[i, h], K=K, L=L)
    init_l2 = np.zeros((hidden, 1, K, L), dtype=np.float32)
    for h in range(hidden):
        init_l2[h, 0] = sample_polynomial_to_lut(poly_c2[h, 0], K=K, L=L)
    return init_l1, init_l2


def run_low_rank_at(rank, poly_coeffs_per_seed, seeds, in_dim, hidden, K, L,
                    alpha, lr_l1, lr_l2, epochs,
                    x_tr, y_tr, x_v, y_v, x_te, y_te):
    per_seed = []
    trace_seed0 = None
    for idx, seed in enumerate(seeds):
        c1, c2 = poly_coeffs_per_seed[idx]
        init_l1, init_l2 = _build_init_luts(c1, c2, in_dim, hidden, K, L)
        model = LowRankResidualLUTKAN2Layer(
            in_dim=in_dim, hidden_dim=hidden, out_dim=1, K=K, L=L,
            alpha=alpha, rank_l1=rank, rank_l2=rank,
        )
        model.init_layer1_from_arrays(init_l1)
        model.init_layer2_from_arrays(init_l2)

        cfg = LowRankTrainConfig(
            lambda_1=0.0, lambda_2=0.0,
            lr_l1=lr_l1, lr_l2=lr_l2,
            epochs=epochs, batch_size=128, seed=seed, eval_every_epochs=10,
        )
        res = train_low_rank_kan2(model, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)
        per_seed.append({
            "seed": seed,
            "mse_val_init": res.mse_val_init,
            "mse_test_at_best": res.mse_test_at_best,
            "mse_val_at_best": res.mse_val_at_best,
            "mse_val_final": res.mse_val_final,
            "best_epoch": res.best_epoch,
            "n_delta_params": model.n_delta_params(),
        })
        if idx == 0:
            trace_seed0 = res.trace
    arr = np.array([p["mse_test_at_best"] for p in per_seed])
    return {
        "rank": rank,
        "per_seed": per_seed,
        "test_mse_mean": float(arr.mean()),
        "test_mse_std": float(arr.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "n_delta_params_per_seed": per_seed[0]["n_delta_params"],
        "trace_seed0": trace_seed0,
    }


def run_full_rank_reference(poly_coeffs_per_seed, seeds, in_dim, hidden, K, L,
                            alpha, lr_l1, lr_l2, epochs,
                            x_tr, y_tr, x_v, y_v, x_te, y_te):
    """Full-rank reference (= M4c best config) using ResidualLUTKAN2Layer."""
    per_seed = []
    trace_seed0 = None
    for idx, seed in enumerate(seeds):
        c1, c2 = poly_coeffs_per_seed[idx]
        init_l1, init_l2 = _build_init_luts(c1, c2, in_dim, hidden, K, L)
        model = ResidualLUTKAN2Layer(
            in_dim=in_dim, hidden_dim=hidden, out_dim=1, K=K, L=L, alpha=alpha,
        )
        model.init_layer1_from_arrays(init_l1)
        model.init_layer2_from_arrays(init_l2)

        cfg = ResidualTrainConfig(
            lambda_1=0.0, lambda_2=0.0,
            lr_l1=lr_l1, lr_l2=lr_l2,
            epochs=epochs, batch_size=128, seed=seed, eval_every_epochs=10,
        )
        res = train_residual_kan2(model, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)
        per_seed.append({
            "seed": seed,
            "mse_val_init": res.mse_val_init,
            "mse_test_at_best": res.mse_test_at_best,
            "mse_val_at_best": res.mse_val_at_best,
            "mse_val_final": res.mse_val_final,
            "best_epoch": res.best_epoch,
            "n_delta_params": (in_dim * hidden + hidden * 1) * K * L,
        })
        if idx == 0:
            trace_seed0 = res.trace
    arr = np.array([p["mse_test_at_best"] for p in per_seed])
    return {
        "rank": "full",
        "per_seed": per_seed,
        "test_mse_mean": float(arr.mean()),
        "test_mse_std": float(arr.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "n_delta_params_per_seed": per_seed[0]["n_delta_params"],
        "trace_seed0": trace_seed0,
    }


def paired_bootstrap_vs_reference(ref_per_seed, cand_per_seed, n_boot=10000, seed=0):
    """Paired bootstrap: is candidate better than reference on per-seed diffs?"""
    ref = np.array([p["mse_test_at_best"] for p in ref_per_seed])
    cand = np.array([p["mse_test_at_best"] for p in cand_per_seed])
    diffs = ref - cand   # positive = candidate better
    rng = np.random.RandomState(seed)
    boots = []
    n = len(diffs)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots.append(diffs[idx].mean())
    boots = np.array(boots)
    return {
        "diff_mean": float(diffs.mean()),
        "diff_ci_lo": float(np.percentile(boots, 2.5)),
        "diff_ci_hi": float(np.percentile(boots, 97.5)),
        "p_positive": float((boots > 0).mean()),
        "all_seeds_positive": bool(all(d > 0 for d in diffs)),
        "ratio_mean": float(ref.mean() / cand.mean()),
    }


def task_pipeline(task_name, x_tr, y_tr, x_v, y_v, x_te, y_te,
                  in_dim, hidden, K, L, degree, alpha, seeds,
                  lut_epochs, poly_epochs, ranks, lr_l1, lr_l2, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\n{task_name}\n{'='*72}")

    # Polynomial baselines
    poly_coeffs = []
    poly_mses = []
    for seed in seeds:
        pres = train_poly_kan2(
            in_dim=in_dim, hidden_dim=hidden, out_dim=1, degree=degree,
            x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
            lr=5e-3, epochs=poly_epochs, batch_size=128, seed=seed,
        )
        poly_coeffs.append((pres["coeffs_l1"], pres["coeffs_l2"]))
        poly_mses.append(pres["test_mse"])
        print(f"  PolyKAN2 seed={seed}: {pres['test_mse']:.3e}")
    poly_mean = float(np.mean(poly_mses))
    print(f"  PolyKAN2 mean: {poly_mean:.3e}")

    print(f"\n  Low-rank residual sweep (alpha={alpha}, lr_l1={lr_l1}, lr_l2={lr_l2}):")
    print(f"  {'rank':>6} {'params/Δ':>9} {'test MSE':>18} {'best_ep':>8} "
          f"{'vs Poly':>10}")

    results = []
    # Full-rank reference first (= M4c best config, for apples-to-apples diff)
    t0 = time.time()
    ref = run_full_rank_reference(
        poly_coeffs, seeds, in_dim, hidden, K, L, alpha, lr_l1, lr_l2,
        lut_epochs, x_tr, y_tr, x_v, y_v, x_te, y_te,
    )
    results.append(ref)
    print(f"  {ref['rank']:>6} {ref['n_delta_params_per_seed']:>9d} "
          f"{ref['test_mse_mean']:>10.3e}±{ref['test_mse_std']:>5.1e} "
          f"{ref['per_seed'][0]['best_epoch']:>8d} "
          f"{poly_mean/ref['test_mse_mean']:>9.2f}x   (dt={time.time()-t0:.1f}s)")

    # Low-rank variants
    for rank in ranks:
        t0 = time.time()
        r = run_low_rank_at(
            rank, poly_coeffs, seeds, in_dim, hidden, K, L,
            alpha, lr_l1, lr_l2, lut_epochs,
            x_tr, y_tr, x_v, y_v, x_te, y_te,
        )
        results.append(r)
        print(f"  {r['rank']:>6} {r['n_delta_params_per_seed']:>9d} "
              f"{r['test_mse_mean']:>10.3e}±{r['test_mse_std']:>5.1e} "
              f"{r['per_seed'][0]['best_epoch']:>8d} "
              f"{poly_mean/r['test_mse_mean']:>9.2f}x   (dt={time.time()-t0:.1f}s)")

    # Paired comparison of each low-rank variant against full-rank reference
    print(f"\n  Paired comparison vs full-rank reference:")
    print(f"  {'rank':>6} {'ratio (ref/cand)':>17} {'CI on diff':>30} "
          f"{'all seeds +':>12}")
    comparisons = {}
    for r in results[1:]:
        c = paired_bootstrap_vs_reference(ref["per_seed"], r["per_seed"])
        comparisons[r["rank"]] = c
        print(f"  {r['rank']:>6} {c['ratio_mean']:>17.3f} "
              f"[{c['diff_ci_lo']:+.2e}, {c['diff_ci_hi']:+.2e}]  "
              f"{str(c['all_seeds_positive']):>12}")

    # Save
    out = {
        "task": task_name,
        "config": {"in_dim": in_dim, "hidden": hidden, "K": K, "L": L,
                   "degree": degree, "alpha": alpha, "seeds": seeds,
                   "lut_epochs": lut_epochs, "poly_epochs": poly_epochs,
                   "lr_l1": lr_l1, "lr_l2": lr_l2, "ranks": ranks},
        "poly_baseline": {"test_mse_per_seed": poly_mses, "test_mse_mean": poly_mean},
        "results": results,
        "comparisons_vs_full": comparisons,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    labels = [str(r["rank"]) for r in results]
    means = np.array([r["test_mse_mean"] for r in results])
    stds = np.array([r["test_mse_std"] for r in results])
    colors = ["tab:gray"] + list(plt.cm.viridis(np.linspace(0.15, 0.85, len(results) - 1)))
    ax1.bar(np.arange(len(labels)), means, yerr=stds, capsize=4,
            color=colors, edgecolor="black")
    ax1.axhline(poly_mean, color="k", linestyle="--", linewidth=1.2,
                label=f"PolyKAN2 = {poly_mean:.2e}")
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_xlabel("Per-edge delta rank  (full = K*L)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Test MSE (best-val)")
    ax1.set_title(f"{task_name}: low-rank residual sweep")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3, axis="y")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m * 1.1, f"{m:.2e}", ha="center", fontsize=7.5)

    for i, r in enumerate(results):
        tr = r["trace_seed0"]
        ax2.semilogy(tr["epoch"], tr["mse_val"], color=colors[i],
                     linewidth=1.4, label=f"rank={r['rank']}")
    ax2.axhline(poly_mean, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val MSE")
    ax2.set_title(f"{task_name}: val-MSE trace (seed 0)")
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "plot.png", dpi=130)
    plt.close()
    print(f"  -> {out_dir / 'plot.png'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/M5a_low_rank")
    ap.add_argument("--poly-epochs", type=int, default=300)
    ap.add_argument("--lut-epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--lr-l1", type=float, default=1e-4)
    ap.add_argument("--lr-l2", type=float, default=5e-4)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_tr1, y_tr1, x_v1, y_v1, x_te1, y_te1 = generate_data_1d(1000, 400, 400, 42)
    r1 = task_pipeline("composition_1d", x_tr1, y_tr1, x_v1, y_v1, x_te1, y_te1,
                       in_dim=1, hidden=4, K=16, L=32, degree=12,
                       alpha=args.alpha, seeds=args.seeds,
                       lut_epochs=args.lut_epochs, poly_epochs=args.poly_epochs,
                       ranks=args.ranks, lr_l1=args.lr_l1, lr_l2=args.lr_l2,
                       out_dir=out_dir / "composition_1d")

    x_tr2, y_tr2, x_v2, y_v2, x_te2, y_te2 = generate_data_2d(
        "feynman_2d", n_train=1000, n_val=400, n_test=400, seed=42)
    r2 = task_pipeline("feynman_2d", x_tr2, y_tr2, x_v2, y_v2, x_te2, y_te2,
                       in_dim=2, hidden=4, K=16, L=32, degree=8,
                       alpha=args.alpha, seeds=args.seeds,
                       lut_epochs=args.lut_epochs, poly_epochs=args.poly_epochs,
                       ranks=args.ranks, lr_l1=args.lr_l1, lr_l2=args.lr_l2,
                       out_dir=out_dir / "feynman_2d")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"composition_1d": r1, "feynman_2d": r2}, f, indent=2)
    print(f"\n{'='*72}\nM5a done\n{'='*72}")


if __name__ == "__main__":
    main()
