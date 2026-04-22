#!/usr/bin/env python3
"""
M3: Does z-score layer-2 normalization help multi-edge direct-LUT?

Motivation (from M2 findings):
    Layer 2 coverage is intrinsically ~40% because tanh(z) has std ≈ 0.3 on our
    tasks. Training cannot fix this — coverage is a property of the input
    distribution. Z-score normalization ((z - mu) / sigma) should expand the
    distribution to fill [x_min_l2, x_max_l2] more uniformly.

Prediction:
    - Layer-2 visited fraction should go from 0.4 → ~0.9 after calibration.
    - MSE should improve because more LUT cells contribute.
    - If MSE does NOT improve despite coverage going up, that proves the
      "coverage is the problem" hypothesis was wrong.

What we do:
    Three conditions on two tasks:
      (a) activation='tanh' (baseline, matches prior experiments)
      (b) activation='zscore', x_min_l2=-3, x_max_l2=+3 (3σ window)
      (c) activation='zscore', x_min_l2=-2, x_max_l2=+2 (2σ — more aggressive,
          some clipping, but denser cell usage)

    Tasks:
      - composition 1D: y = tanh(2*sin(pi*x)), [1→4→1]
      - feynman_2d: y = sin(pi*x1) + 0.5*cos(2*pi*x1*x2), [2→4→1]

    For each (activation, task): initialize from a trained PolyKAN2,
    train direct-LUT with lr=5e-4, λ₂=0, seeds 0-2.

Output: results/M3_zscore/summary.json + per-task comparison PNG.

Runtime: ~8-10 min CPU.
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
    KAN2TrainConfig,
    LUTKAN2Layer,
    PolyKAN2Zscore,
    compute_kan2_coverage,
    coverage_report_to_dict,
    generate_data_2d,
    sample_polynomial_to_lut,
    train_kan2,
    train_poly_kan2_zscore,
)
from exp_H4_kan2_2d import PolyKAN2, train_poly_kan2
from m2b_composition import generate_data_1d, target_composition  # noqa: F401


def poly_init_lut_kan2(
    poly_coeffs_l1, poly_coeffs_l2,
    K, L,
    in_dim, hidden_dim, out_dim,
    activation, x_min_l2, x_max_l2,
    z_mean=None, z_std=None,
):
    """Initialize a LUTKAN2Layer from polynomial coefficients.

    For activation='zscore', layer-2 coefficients MUST have been fit on the
    same [x_min_l2, x_max_l2] domain (i.e. produced by PolyKAN2Zscore with
    the same domain). Also, if z_mean/z_std are provided, they're copied into
    the model's buffers so subsequent training uses the matching normalization.
    """
    model = LUTKAN2Layer(
        in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, K=K, L=L,
        activation=activation, x_min_l2=x_min_l2, x_max_l2=x_max_l2,
    )
    with torch.no_grad():
        for i in range(in_dim):
            for h in range(hidden_dim):
                lut = sample_polynomial_to_lut(poly_coeffs_l1[i, h], K=K, L=L)
                model.lut_l1.data[i, h] = torch.from_numpy(lut)
        for h in range(hidden_dim):
            for o in range(out_dim):
                # Layer-2 LUT sampled on the layer-2 domain. eval_chebyshev
                # rescales [x_min_l2, x_max_l2] back to [-1, 1] internally, so
                # for zscore this only makes sense when coeffs were fit by
                # PolyKAN2Zscore (which uses the same rescaling).
                lut = sample_polynomial_to_lut(
                    poly_coeffs_l2[h, o], K=K, L=L,
                    x_min=x_min_l2, x_max=x_max_l2,
                )
                model.lut_l2.data[h, o] = torch.from_numpy(lut)

        if activation == "zscore" and z_mean is not None and z_std is not None:
            model._z_mean.copy_(torch.from_numpy(np.asarray(z_mean, dtype=np.float32)))
            model._z_std.copy_(torch.from_numpy(np.asarray(z_std, dtype=np.float32)))
            model._calibrated.fill_(True)
    return model


def eval_mse_test(model, x_te, y_te):
    with torch.no_grad():
        pred = model(torch.from_numpy(x_te.astype(np.float32))).cpu().numpy().ravel()
    return float(np.mean((pred - y_te.ravel()) ** 2))


def run_one_condition(
    task_name: str, x_tr, y_tr, x_v, y_v, x_te, y_te,
    in_dim: int, hidden: int,
    poly_coeffs: list,   # list of (c_l1, c_l2) per seed for tanh-fit
    poly_coeffs_zscore_by_domain: dict,  # {(x_min, x_max): list of (c_l1, c_l2, z_mean, z_std)}
    seeds: list,
    activation: str, x_min_l2: float, x_max_l2: float,
    K: int, L: int, lr: float, lut_epochs: int,
):
    """Returns dict with per-seed results + last-seed coverage."""
    per_seed_inits = []
    per_seed_bests = []
    cov_before_last = None
    cov_after_last = None
    z_mean_last = None
    z_std_last = None
    frac_clipped_last = None

    # Pick the right coefficient source per activation mode.
    if activation == "tanh":
        source_list = poly_coeffs
    else:
        key = (x_min_l2, x_max_l2)
        if key not in poly_coeffs_zscore_by_domain:
            raise RuntimeError(f"No zscore poly coeffs fit for domain {key}")
        source_list = poly_coeffs_zscore_by_domain[key]

    for seed_idx, seed in enumerate(seeds):
        entry = source_list[seed_idx]
        if activation == "tanh":
            c1, c2 = entry
            z_mean, z_std = None, None
        else:
            c1, c2, z_mean, z_std = entry

        model = poly_init_lut_kan2(
            c1, c2, K=K, L=L,
            in_dim=in_dim, hidden_dim=hidden, out_dim=1,
            activation=activation, x_min_l2=x_min_l2, x_max_l2=x_max_l2,
            z_mean=z_mean, z_std=z_std,
        )

        if activation == "zscore":
            frac_clipped_last = None   # could recompute; skip for brevity
            z_mean_last = z_mean.tolist() if z_mean is not None else None
            z_std_last = z_std.tolist() if z_std is not None else None

        mse_init = eval_mse_test(model, x_te, y_te)
        per_seed_inits.append(mse_init)

        cov_before_last = compute_kan2_coverage(model, x_tr)

        cfg = KAN2TrainConfig(
            lambda_1=0.0, lambda_2=0.0, lr=lr,
            epochs=lut_epochs, batch_size=128,
            init_noise_std_absolute=0.0, seed=seed, eval_every_epochs=15,
            calibrate_at_start=False,   # already calibrated (or tanh, no calib needed)
        )
        res = train_kan2(model, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)
        per_seed_bests.append(res.mse_test_at_best)

        model.lut_l1.data = torch.from_numpy(res.lut_l1_best)
        model.lut_l2.data = torch.from_numpy(res.lut_l2_best)
        cov_after_last = compute_kan2_coverage(model, x_tr)

    inits = np.array(per_seed_inits)
    bests = np.array(per_seed_bests)
    return {
        "activation": activation,
        "x_min_l2": x_min_l2, "x_max_l2": x_max_l2,
        "task": task_name,
        "poly_init_mse_per_seed": inits.tolist(),
        "direct_best_mse_per_seed": bests.tolist(),
        "poly_init_mean": float(inits.mean()),
        "poly_init_std": float(inits.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "direct_best_mean": float(bests.mean()),
        "direct_best_std": float(bests.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "improvement_over_init": float(inits.mean() / bests.mean()),
        "coverage_before_last_seed": coverage_report_to_dict(cov_before_last),
        "coverage_after_last_seed": coverage_report_to_dict(cov_after_last),
        "calibration_stats": {
            "z_mean_per_hidden": z_mean_last,
            "z_std_per_hidden": z_std_last,
            "frac_clipped_after_zscore": frac_clipped_last,
        } if activation == "zscore" else None,
    }


def plot_task_comparison(task_name, conditions, poly_baseline, out_path):
    """One figure per task: MSE bars + layer-2 coverage bars side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # --- MSE bars ---
    labels = [c["name"] for c in conditions]
    init_means = [c["result"]["poly_init_mean"] for c in conditions]
    init_stds = [c["result"]["poly_init_std"] for c in conditions]
    best_means = [c["result"]["direct_best_mean"] for c in conditions]
    best_stds = [c["result"]["direct_best_std"] for c in conditions]

    xs = np.arange(len(labels))
    w = 0.35
    ax1.bar(xs - w/2, init_means, w, yerr=init_stds, capsize=3,
            label="Poly-init (before training)", color="lightgray",
            edgecolor="black")
    ax1.bar(xs + w/2, best_means, w, yerr=best_stds, capsize=3,
            label="Direct-LUT best-val", color="tab:blue", edgecolor="black")
    ax1.axhline(poly_baseline, color="k", linestyle="--", linewidth=1.3,
                label=f"PolyKAN2 = {poly_baseline:.1e}")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, rotation=0, fontsize=9)
    ax1.set_yscale("log")
    ax1.set_ylabel("Test MSE")
    ax1.set_title(f"{task_name}: MSE")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, which="both", alpha=0.3, axis="y")
    for x, v in zip(xs - w/2, init_means):
        ax1.text(x, v * 1.15, f"{v:.1e}", ha="center", fontsize=7)
    for x, v in zip(xs + w/2, best_means):
        ax1.text(x, v * 1.15, f"{v:.1e}", ha="center", fontsize=7)

    # --- Layer-2 coverage bars (before training) ---
    KL_frac_before = [
        c["result"]["coverage_before_last_seed"]["layer2"]["effective_support_mean"]
        / (c["result"]["coverage_before_last_seed"]["layer2"]["K"]
           * c["result"]["coverage_before_last_seed"]["layer2"]["L"])
        for c in conditions
    ]
    vis_before = [
        c["result"]["coverage_before_last_seed"]["layer2"]["visited_fraction_mean"]
        for c in conditions
    ]
    range_util_before = [
        c["result"]["coverage_before_last_seed"]["layer2"]["range_utilization_mean"]
        for c in conditions
    ]
    ax2.bar(xs - w, vis_before, w, label="visited fraction",
            color="tab:orange", edgecolor="black")
    ax2.bar(xs, KL_frac_before, w, label="eff_support / KL",
            color="tab:green", edgecolor="black")
    ax2.bar(xs + w, range_util_before, w, label="range utilization",
            color="tab:purple", edgecolor="black")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, rotation=0, fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Layer-2 coverage metric")
    ax2.set_title(f"{task_name}: Layer-2 coverage")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3, axis="y")
    for x, v in zip(xs - w, vis_before):
        ax2.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
    for x, v in zip(xs, KL_frac_before):
        ax2.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
    for x, v in zip(xs + w, range_util_before):
        ax2.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/M3_zscore")
    ap.add_argument("--poly-epochs", type=int, default=800)
    ap.add_argument("--lut-epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=4)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions_def = [
        {"name": "tanh (baseline)", "activation": "tanh",
         "x_min_l2": -1.0, "x_max_l2": 1.0},
        {"name": "zscore 3σ", "activation": "zscore",
         "x_min_l2": -3.0, "x_max_l2": 3.0},
        {"name": "zscore 2σ", "activation": "zscore",
         "x_min_l2": -2.0, "x_max_l2": 2.0},
    ]

    all_results = {}

    def _fit_zscore_polys(in_dim, hidden, degree, x_tr, y_tr, x_v, y_v, x_te, y_te,
                          epochs, zscore_conditions, seeds):
        """For each distinct (x_min_l2, x_max_l2) domain, fit a PolyKAN2Zscore
        per seed. Returns dict keyed by (x_min, x_max) -> list of tuples."""
        out = {}
        domains = sorted({(c["x_min_l2"], c["x_max_l2"]) for c in zscore_conditions
                          if c["activation"] == "zscore"})
        for (x_min, x_max) in domains:
            entries = []
            print(f"  Fitting PolyKAN2Zscore for domain [{x_min}, {x_max}]:")
            for seed in seeds:
                pres = train_poly_kan2_zscore(
                    in_dim=in_dim, hidden_dim=hidden, out_dim=1, degree=degree,
                    x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
                    x_min_l2=x_min, x_max_l2=x_max,
                    lr=5e-3, epochs=epochs, batch_size=128, seed=seed,
                )
                entries.append((pres["coeffs_l1"], pres["coeffs_l2"],
                                pres["z_mean"], pres["z_std"]))
                print(f"    seed={seed}: test MSE = {pres['test_mse']:.3e}")
            out[(x_min, x_max)] = entries
        return out

    # ========================================================================
    # Task 1: composition 1D
    # ========================================================================
    print("=" * 72)
    print("Task: composition 1D (y = tanh(2*sin(pi*x)))")
    print("=" * 72)
    x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data_1d(
        n_train=1000, n_val=400, n_test=400, seed=42,
    )

    # tanh-PolyKAN2 per seed
    poly_coeffs_1d = []
    poly_mses_1d = []
    for seed in args.seeds:
        pres = train_poly_kan2(
            in_dim=1, hidden_dim=args.hidden, out_dim=1, degree=12,
            x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
            lr=5e-3, epochs=args.poly_epochs, batch_size=128, seed=seed,
        )
        poly_coeffs_1d.append((pres["coeffs_l1"], pres["coeffs_l2"]))
        poly_mses_1d.append(pres["test_mse"])
        print(f"  PolyKAN2 (tanh) seed={seed}: {pres['test_mse']:.3e}")
    poly_base_1d = float(np.mean(poly_mses_1d))
    print(f"  PolyKAN2 (tanh) mean: {poly_base_1d:.3e}")

    # zscore-PolyKAN2 per domain per seed
    poly_z_1d = _fit_zscore_polys(
        in_dim=1, hidden=args.hidden, degree=12,
        x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
        epochs=args.poly_epochs,
        zscore_conditions=conditions_def, seeds=args.seeds,
    )

    task1_results = []
    for cond_def in conditions_def:
        t0 = time.time()
        r = run_one_condition(
            task_name="composition_1d",
            x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
            in_dim=1, hidden=args.hidden,
            poly_coeffs=poly_coeffs_1d,
            poly_coeffs_zscore_by_domain=poly_z_1d,
            seeds=args.seeds,
            activation=cond_def["activation"],
            x_min_l2=cond_def["x_min_l2"], x_max_l2=cond_def["x_max_l2"],
            K=args.K, L=args.L, lr=5e-4, lut_epochs=args.lut_epochs,
        )
        task1_results.append({"name": cond_def["name"], "result": r})
        cov_before = r["coverage_before_last_seed"]["layer2"]
        print(f"  {cond_def['name']:<22}  init={r['poly_init_mean']:.3e}  "
              f"best={r['direct_best_mean']:.3e} (±{r['direct_best_std']:.1e})  "
              f"L2_vis={cov_before['visited_fraction_mean']:.2f}  "
              f"L2_eff/KL={cov_before['effective_support_mean']/(args.K*args.L):.2f}  "
              f"dt={time.time()-t0:.1f}s")
    all_results["composition_1d"] = {
        "poly_baseline_tanh": poly_base_1d,
        "poly_baseline_tanh_per_seed": poly_mses_1d,
        "conditions": task1_results,
    }
    plot_task_comparison("composition 1D", task1_results, poly_base_1d,
                         out_dir / "composition_1d.png")
    print(f"  -> {out_dir / 'composition_1d.png'}")

    # ========================================================================
    # Task 2: feynman 2D
    # ========================================================================
    print("\n" + "=" * 72)
    print("Task: feynman_2d (y = sin(pi*x1) + 0.5*cos(2*pi*x1*x2))")
    print("=" * 72)
    x_tr2, y_tr2, x_v2, y_v2, x_te2, y_te2 = generate_data_2d(
        "feynman_2d", n_train=1000, n_val=400, n_test=400, seed=42,
    )

    poly_coeffs_2d = []
    poly_mses_2d = []
    for seed in args.seeds:
        pres = train_poly_kan2(
            in_dim=2, hidden_dim=args.hidden, out_dim=1, degree=8,
            x_tr=x_tr2, y_tr=y_tr2, x_v=x_v2, y_v=y_v2, x_te=x_te2, y_te=y_te2,
            lr=5e-3, epochs=args.poly_epochs, batch_size=128, seed=seed,
        )
        poly_coeffs_2d.append((pres["coeffs_l1"], pres["coeffs_l2"]))
        poly_mses_2d.append(pres["test_mse"])
        print(f"  PolyKAN2 (tanh) seed={seed}: {pres['test_mse']:.3e}")
    poly_base_2d = float(np.mean(poly_mses_2d))
    print(f"  PolyKAN2 (tanh) mean: {poly_base_2d:.3e}")

    poly_z_2d = _fit_zscore_polys(
        in_dim=2, hidden=args.hidden, degree=8,
        x_tr=x_tr2, y_tr=y_tr2, x_v=x_v2, y_v=y_v2, x_te=x_te2, y_te=y_te2,
        epochs=args.poly_epochs,
        zscore_conditions=conditions_def, seeds=args.seeds,
    )

    task2_results = []
    for cond_def in conditions_def:
        t0 = time.time()
        r = run_one_condition(
            task_name="feynman_2d",
            x_tr=x_tr2, y_tr=y_tr2, x_v=x_v2, y_v=y_v2, x_te=x_te2, y_te=y_te2,
            in_dim=2, hidden=args.hidden,
            poly_coeffs=poly_coeffs_2d,
            poly_coeffs_zscore_by_domain=poly_z_2d,
            seeds=args.seeds,
            activation=cond_def["activation"],
            x_min_l2=cond_def["x_min_l2"], x_max_l2=cond_def["x_max_l2"],
            K=args.K, L=args.L, lr=5e-4, lut_epochs=args.lut_epochs,
        )
        task2_results.append({"name": cond_def["name"], "result": r})
        cov_before = r["coverage_before_last_seed"]["layer2"]
        print(f"  {cond_def['name']:<22}  init={r['poly_init_mean']:.3e}  "
              f"best={r['direct_best_mean']:.3e} (±{r['direct_best_std']:.1e})  "
              f"L2_vis={cov_before['visited_fraction_mean']:.2f}  "
              f"L2_eff/KL={cov_before['effective_support_mean']/(args.K*args.L):.2f}  "
              f"dt={time.time()-t0:.1f}s")
    all_results["feynman_2d"] = {
        "poly_baseline_tanh": poly_base_2d,
        "poly_baseline_tanh_per_seed": poly_mses_2d,
        "conditions": task2_results,
    }
    plot_task_comparison("feynman 2D", task2_results, poly_base_2d,
                         out_dir / "feynman_2d.png")
    print(f"  -> {out_dir / 'feynman_2d.png'}")

    # ========================================================================
    # Save JSON
    # ========================================================================
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "config": {"K": args.K, "L": args.L, "hidden": args.hidden,
                       "poly_epochs": args.poly_epochs,
                       "lut_epochs": args.lut_epochs, "seeds": args.seeds},
            "results_by_task": all_results,
        }, f, indent=2)
    print(f"\nSaved: {out_dir / 'summary.json'}")

    # ========================================================================
    # Headline table
    # ========================================================================
    print("\n" + "=" * 72)
    print("HEADLINE")
    print("=" * 72)
    for task_name, data in all_results.items():
        print(f"\n{task_name}  (PolyKAN2(tanh) baseline = {data['poly_baseline_tanh']:.3e})")
        print(f"{'condition':<22} {'init MSE':>10} {'best MSE':>12} "
              f"{'L2_vis':>8} {'L2_eff/KL':>10} {'vs Poly':>10}")
        for c in data["conditions"]:
            r = c["result"]
            cov = r["coverage_before_last_seed"]["layer2"]
            KL = cov["K"] * cov["L"]
            print(f"{c['name']:<22} {r['poly_init_mean']:>10.2e} "
                  f"{r['direct_best_mean']:>12.2e} "
                  f"{cov['visited_fraction_mean']:>8.2f} "
                  f"{cov['effective_support_mean']/KL:>10.2f} "
                  f"{data['poly_baseline_tanh']/r['direct_best_mean']:>9.2f}x")


if __name__ == "__main__":
    main()
