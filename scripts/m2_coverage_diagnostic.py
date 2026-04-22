#!/usr/bin/env python3
"""
M2 coverage diagnostic for multi-edge LUT-KAN.

The existing repo has three H4 scripts that give apparently contradictory results:
  - exp_H4_kan2_2d.py (poly-init LUT-KAN2):     MSE ~1.7e-4 (1.66x better than poly)
  - exp_H4_multi_edge_2d.py (sweep):            Poly Pareto-dominates LUT everywhere
  - exp_H4_multiedge.py (random / identity init): null result, MSE ~5e-1

Hypothesis: the difference is initialization. Poly-init puts LUTs into a region
where gradients are meaningful; random / identity init leaves them in a cold
coverage regime where most cells never receive updates.

This script replays both scenarios with seeded determinism, and measures
coverage before/after training in both layers. If the hypothesis holds,
random-init will show (a) low visited_fraction, (b) tight tanh(z) concentration
near 0 → low layer-2 coverage, (c) MSE stuck near trivial baseline.

Run:
    python scripts/m2_coverage_diagnostic.py --out results/M2_coverage
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lut_native import (  # noqa: E402
    KAN2TrainConfig,
    LUTKAN2Layer,
    PolyKAN2Layer,
    compute_kan2_coverage,
    coverage_report_to_dict,
    eval_chebyshev,
    fit_chebyshev_ls,
    generate_data_2d,
    sample_polynomial_to_lut,
    train_kan2,
    train_poly_kan2,
)


def _init_from_poly_kan(lut_model: LUTKAN2Layer, poly_model: PolyKAN2Layer):
    """Sample each polynomial edge into a LUT grid matching lut_model's dims."""
    K, L = lut_model.K, lut_model.L
    # Layer 1: (in_dim, hidden_dim) polynomial coeffs -> LUTs
    with torch.no_grad():
        coeffs1 = poly_model.c_l1.detach().cpu().numpy()   # (in_dim, hidden_dim, degree+1)
        for i in range(lut_model.in_dim):
            for h in range(lut_model.hidden_dim):
                lut = sample_polynomial_to_lut(coeffs1[i, h], K=K, L=L)
                lut_model.lut_l1.data[i, h] = torch.from_numpy(lut)
        coeffs2 = poly_model.c_l2.detach().cpu().numpy()   # (hidden_dim, out_dim, degree+1)
        for h in range(lut_model.hidden_dim):
            for o in range(lut_model.out_dim):
                lut = sample_polynomial_to_lut(coeffs2[h, o], K=K, L=L)
                lut_model.lut_l2.data[h, o] = torch.from_numpy(lut)


def _init_identity_scaled(lut_model: LUTKAN2Layer):
    """Each edge = 0.5*x, matching exp_H4_multiedge.py's identity init."""
    K, L = lut_model.K, lut_model.L
    id_coeffs = np.array([0.0, 0.5] + [0.0] * 19, dtype=np.float32)
    id_lut = sample_polynomial_to_lut(id_coeffs, K=K, L=L)
    with torch.no_grad():
        for i in range(lut_model.in_dim):
            for h in range(lut_model.hidden_dim):
                lut_model.lut_l1.data[i, h] = torch.from_numpy(id_lut)
        for h in range(lut_model.hidden_dim):
            for o in range(lut_model.out_dim):
                lut_model.lut_l2.data[h, o] = torch.from_numpy(id_lut)


def make_lut_kan2(K, L, hidden_dim=4, in_dim=2, out_dim=1) -> LUTKAN2Layer:
    return LUTKAN2Layer(
        in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, K=K, L=L,
    )


def run_scenario(
    name: str,
    init_strategy: str,        # "poly_init" | "random_init" | "identity_init"
    x_tr, y_tr, x_v, y_v, x_te, y_te,
    K: int, L: int, hidden: int,
    lambda_2: float, epochs: int, seed: int,
    poly_model: PolyKAN2Layer = None,
) -> dict:
    """Run one training scenario, measure coverage before and after, return report."""
    # Build the LUT model
    model = make_lut_kan2(K=K, L=L, hidden_dim=hidden, in_dim=2, out_dim=1)

    if init_strategy == "poly_init":
        assert poly_model is not None, "poly_init needs a trained poly_model"
        _init_from_poly_kan(model, poly_model)
        init_noise_abs = 0.01
    elif init_strategy == "identity_init":
        _init_identity_scaled(model)
        init_noise_abs = 0.05
    elif init_strategy == "random_init":
        # Default zero-init + noise (matches how LUTKAN2Layer starts)
        init_noise_abs = 0.1
    else:
        raise ValueError(init_strategy)

    # Coverage BEFORE training
    cov_before = compute_kan2_coverage(model, x_tr)
    mse_before = float(
        ((torch.from_numpy(y_tr.reshape(-1, 1)) -
          model(torch.from_numpy(x_tr.astype(np.float32)))) ** 2).mean().item()
    )

    # Train
    cfg = KAN2TrainConfig(
        lambda_1=0.0, lambda_2=lambda_2,
        lr=1e-2, epochs=epochs, batch_size=64,
        init_noise_std_absolute=init_noise_abs,
        seed=seed, eval_every_epochs=max(1, epochs // 20),
    )
    t0 = time.time()
    result = train_kan2(model, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)
    dt = time.time() - t0

    # Reload best weights for post-training coverage measurement
    model.lut_l1.data = torch.from_numpy(result.lut_l1_best)
    model.lut_l2.data = torch.from_numpy(result.lut_l2_best)
    cov_after = compute_kan2_coverage(model, x_tr)

    return {
        "name": name,
        "init_strategy": init_strategy,
        "K": K, "L": L, "hidden": hidden,
        "lambda_2": lambda_2,
        "epochs": epochs, "seed": seed,
        "wall_time_sec": dt,
        "mse_train_before": mse_before,
        "mse_train_at_best": float(result.mse_train_at_best),
        "mse_val_at_best": float(result.mse_val_at_best),
        "mse_test_at_best": float(result.mse_test_at_best),
        "best_epoch": int(result.best_epoch),
        "coverage_before": coverage_report_to_dict(cov_before),
        "coverage_after": coverage_report_to_dict(cov_after),
        "trace_epochs": result.trace["epoch"],
        "trace_val_mse": result.trace["mse_val"],
    }


def plot_comparison(scenarios: list, out_path: Path, target_std2: float):
    """Side-by-side view of coverage metrics and MSE trace across scenarios."""
    n = len(scenarios)
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, n, height_ratios=[1.2, 1, 1.5], hspace=0.55, wspace=0.35)

    colors = ["tab:red", "tab:blue", "tab:green"]
    for i, scn in enumerate(scenarios):
        color = colors[i % len(colors)]
        l1_before = scn["coverage_before"]["layer1"]
        l1_after = scn["coverage_after"]["layer1"]
        l2_before = scn["coverage_before"]["layer2"]
        l2_after = scn["coverage_after"]["layer2"]

        # Row 0: bars with coverage metrics before/after, layer 1 vs layer 2
        ax = fig.add_subplot(gs[0, i])
        metrics = ["visited\nfraction", "effective\nsupport\n(/ K*L)",
                   "range\nutilization"]
        KL = scn["K"] * scn["L"]
        vals_before = [
            l1_before["visited_fraction_mean"],
            l1_before["effective_support_mean"] / KL,
            l1_before["range_utilization_mean"],
        ]
        vals_after = [
            l1_after["visited_fraction_mean"],
            l1_after["effective_support_mean"] / KL,
            l1_after["range_utilization_mean"],
        ]
        xs = np.arange(len(metrics))
        w = 0.35
        ax.bar(xs - w/2, vals_before, w, label="before", color="lightgray",
               edgecolor="black")
        ax.bar(xs + w/2, vals_after, w, label="after", color=color,
               edgecolor="black")
        ax.set_xticks(xs)
        ax.set_xticklabels(metrics, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{scn['name']}\nLayer 1 coverage",
                     fontsize=9, color=color)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)

        # Row 1: Layer 2 coverage
        ax = fig.add_subplot(gs[1, i])
        vals_before2 = [
            l2_before["visited_fraction_mean"],
            l2_before["effective_support_mean"] / KL,
            l2_before["range_utilization_mean"],
        ]
        vals_after2 = [
            l2_after["visited_fraction_mean"],
            l2_after["effective_support_mean"] / KL,
            l2_after["range_utilization_mean"],
        ]
        ax.bar(xs - w/2, vals_before2, w, label="before", color="lightgray",
               edgecolor="black")
        ax.bar(xs + w/2, vals_after2, w, label="after", color=color,
               edgecolor="black")
        ax.set_xticks(xs)
        ax.set_xticklabels(metrics, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title("Layer 2 coverage", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        # Row 2: MSE trace (and hidden activation hist beside it in a nested)
        ax = fig.add_subplot(gs[2, i])
        ax.semilogy(scn["trace_epochs"], scn["trace_val_mse"],
                    color=color, linewidth=1.5)
        ax.axhline(target_std2, color="k", linestyle="--", alpha=0.4,
                   label=f"trivial (std² = {target_std2:.1e})")
        ax.axhline(scn["mse_test_at_best"], color=color, linestyle=":",
                   alpha=0.6, label=f"test@best = {scn['mse_test_at_best']:.1e}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MSE")
        ax.set_title(f"Training trace\ntanh(z) |·|>0.99: "
                     f"{scn['coverage_after']['hidden_activation_stats']['frac_abs_gt_0_99']:.2f} "
                     f"|·|<0.1: "
                     f"{scn['coverage_after']['hidden_activation_stats']['frac_abs_lt_0_1']:.2f}",
                     fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("M2: multi-edge LUT-KAN coverage diagnostic "
                 "(does training actually reach all LUT cells?)",
                 fontsize=11)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/M2_coverage")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--poly-epochs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"M2 coverage diagnostic   K={args.K}, L={args.L}, hidden={args.hidden}")
    print("=" * 72)

    x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data_2d(
        "feynman_2d", n_train=1000, n_val=400, n_test=400, seed=42,
    )
    target_std2 = float(np.var(y_te))
    print(f"Target: feynman_2d.  Test variance (trivial baseline MSE): {target_std2:.3e}")

    # --- First, train the polynomial KAN2 to convergence ---
    print(f"\n[1/4] Training PolyKAN2 (epochs={args.poly_epochs}, seed={args.seed})...")
    poly = PolyKAN2Layer(in_dim=2, hidden_dim=args.hidden, out_dim=1, degree=8)
    poly_res = train_poly_kan2(
        poly, x_tr, y_tr, x_v, y_v, x_te, y_te,
        lr=5e-3, epochs=args.poly_epochs, seed=args.seed,
    )
    print(f"   PolyKAN2 test MSE: {poly_res['mse_test_at_best']:.3e}")

    scenarios = []

    # --- Scenario A: poly-init LUT-KAN2 ---
    print(f"\n[2/4] LUT-KAN2 (poly-init, lambda_2=0.0)...")
    s_a = run_scenario(
        name="poly_init, l2=0.0",
        init_strategy="poly_init",
        x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
        K=args.K, L=args.L, hidden=args.hidden,
        lambda_2=0.0, epochs=args.epochs, seed=args.seed,
        poly_model=poly,
    )
    scenarios.append(s_a)
    print(f"   test MSE: {s_a['mse_test_at_best']:.3e}")
    print(f"   L1 visited after: {s_a['coverage_after']['layer1']['visited_fraction_mean']:.2f}")
    print(f"   L2 visited after: {s_a['coverage_after']['layer2']['visited_fraction_mean']:.2f}")

    # --- Scenario B: identity-init LUT-KAN2 ---
    print(f"\n[3/4] LUT-KAN2 (identity-init, lambda_2=0.0)...")
    s_b = run_scenario(
        name="identity_init, l2=0.0",
        init_strategy="identity_init",
        x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
        K=args.K, L=args.L, hidden=args.hidden,
        lambda_2=0.0, epochs=args.epochs, seed=args.seed,
    )
    scenarios.append(s_b)
    print(f"   test MSE: {s_b['mse_test_at_best']:.3e}")
    print(f"   L1 visited after: {s_b['coverage_after']['layer1']['visited_fraction_mean']:.2f}")
    print(f"   L2 visited after: {s_b['coverage_after']['layer2']['visited_fraction_mean']:.2f}")

    # --- Scenario C: random-init LUT-KAN2 ---
    print(f"\n[4/4] LUT-KAN2 (random-init, lambda_2=0.0)...")
    s_c = run_scenario(
        name="random_init, l2=0.0",
        init_strategy="random_init",
        x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
        K=args.K, L=args.L, hidden=args.hidden,
        lambda_2=0.0, epochs=args.epochs, seed=args.seed,
    )
    scenarios.append(s_c)
    print(f"   test MSE: {s_c['mse_test_at_best']:.3e}")
    print(f"   L1 visited after: {s_c['coverage_after']['layer1']['visited_fraction_mean']:.2f}")
    print(f"   L2 visited after: {s_c['coverage_after']['layer2']['visited_fraction_mean']:.2f}")

    # --- Save + plot ---
    out_json = {
        "config": {"K": args.K, "L": args.L, "hidden": args.hidden,
                   "epochs": args.epochs, "seed": args.seed,
                   "target": "feynman_2d",
                   "target_std2": target_std2},
        "poly_kan_baseline": {
            "test_mse": poly_res["mse_test_at_best"],
            "memory_bytes_float32": poly.memory_bytes_float32(),
        },
        "scenarios": scenarios,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved: {out_dir / 'summary.json'}")

    plot_comparison(scenarios, out_dir / "comparison.png", target_std2)
    print(f"Saved: {out_dir / 'comparison.png'}")

    # --- Headline table ---
    print("\n" + "=" * 72)
    print("HEADLINE:")
    print("=" * 72)
    print(f"{'scenario':<28} {'test MSE':>10}  {'L1 vis':>7} {'L2 vis':>7}  {'L2 eff_supp/KL':>14}")
    for s in scenarios:
        print(f"{s['name']:<28} {s['mse_test_at_best']:>10.2e}  "
              f"{s['coverage_after']['layer1']['visited_fraction_mean']:>7.2f} "
              f"{s['coverage_after']['layer2']['visited_fraction_mean']:>7.2f}  "
              f"{s['coverage_after']['layer2']['effective_support_mean'] / (args.K*args.L):>14.2f}")
    print(f"{'PolyKAN2':<28} {poly_res['mse_test_at_best']:>10.2e}  "
          f"{'—':>7} {'—':>7}  {'—':>14}")


if __name__ == "__main__":
    main()
