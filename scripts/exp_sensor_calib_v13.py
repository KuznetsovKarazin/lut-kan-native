"""
exp_sensor_calib_v13.py — Multi-sensor calibration study for TIM v13.

Three sensor types: NTC thermistor, MQ gas, pH electrode.
For each: 8 virtual units spanning datasheet variation.
Baseline: nominal factory polynomial (no calibration) vs direct LUT training.

Outputs (in results/sensor_calib/):
  summary.json         — all numbers, structured
  per_sensor_table.csv — Table III equivalent for all three sensors
  improvement_plot.png — bar chart: MAE before/after, per sensor type
  ncal_sweep.png       — MAE vs N_cal for all three sensors
  ncal_sweep.json      — raw data for ncal sweep
  coverage_rule.png    — K*L vs n_train: stable vs unstable regime

Run: python scripts/exp_sensor_calib_v13.py
Reproducible: all seeds fixed, no random global state.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lut_native.core import LUTEdge, lut_forward_numpy
from lut_native.training import train_lut_edge, TrainConfig
from lut_native.sensors import (
    make_ntc_units, ntc_calibration_data, ntc_factory_baseline,
    make_mq_units, mq_calibration_data,
    make_ph_units, ph_calibration_data,
    fit_poly_baseline,
)

OUT = ROOT / "results" / "sensor_calib"
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment config
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    # LUT config (matching NTC hardware setup, ~144 bytes uint8)
    K=4, L=32,
    lambda_2=1.0,
    lr=0.01,
    epochs=1000,
    batch_size=32,
    # Calibration
    N_CAL=50,
    N_VAL=25,
    N_TEST=200,
    N_UNITS=8,
    # Polynomial baseline degree (per sensor type)
    POLY_DEGREE_NTC=3,
    POLY_DEGREE_MQ=4,
    POLY_DEGREE_PH=3,
    # Seeds
    SENSOR_SEED=42,
    TRAIN_SEEDS=[0, 1, 2],
    # N_cal sweep
    N_CAL_VALUES=[5, 10, 20, 30, 50, 75, 100, 150, 200],
    # Coverage rule sweep
    KL_VALUES=[(1,8),(1,16),(1,32),(2,8),(2,16),(2,32),(4,8),(4,16),(4,32)],
    N_TRAIN_VALUES=[8, 16, 32, 64, 128, 256, 512],
)


# ─────────────────────────────────────────────────────────────────────────────
# Core calibration runner  (sensor-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(sensor_data: dict,
                    K: int, L: int,
                    lambda_2: float,
                    poly_degree: int,
                    train_seeds: list,
                    lr: float = 0.01,
                    epochs: int = 1000,
                    batch_size: int = 32) -> dict:
    """
    Train a K×L LUT on sensor calibration data; compare to polynomial baseline.
    Returns dict with mse and physical-unit MAE for both methods.
    """
    x_cal  = sensor_data["x_cal"]
    y_cal  = sensor_data["y_cal"]
    x_val  = sensor_data["x_val"]
    y_val  = sensor_data["y_val"]
    x_test = sensor_data["x_test"]
    y_test = sensor_data["y_test"]
    denorm_y = sensor_data["denorm_y"]

    # ── Polynomial baseline ──────────────────────────────────────────────────
    poly_res = fit_poly_baseline(x_cal, y_cal, poly_degree, x_test, y_test, denorm_y)

    # ── Direct LUT training (multiple seeds) ─────────────────────────────────
    lut_init = np.zeros((K, L), dtype=np.float32)  # zero init
    cfg_train = TrainConfig(
        lambda_2=lambda_2, lr=lr, epochs=epochs,
        batch_size=batch_size, eval_every_epochs=20,
    )

    lut_mse_vals = []
    lut_mae_vals = []
    for seed in train_seeds:
        cfg_train.seed = seed
        result = train_lut_edge(
            lut_init=lut_init.copy(),
            x_train=x_cal, y_train=y_cal,
            x_val=x_val,   y_val=y_val,
            x_test=x_test, y_test=y_test,
            x_min=-1.0, x_max=1.0,
            cfg=cfg_train,
        )
        y_lut = lut_forward_numpy(x_test, result.lut_best)
        lut_mse_vals.append(result.mse_test_at_best)
        mae_phys = float(np.mean(np.abs(denorm_y(y_lut) - denorm_y(y_test))))
        lut_mae_vals.append(mae_phys)

    return dict(
        poly_mse=poly_res["mse"],
        poly_mae_phys=poly_res["mae_phys"],
        lut_mse_mean=float(np.mean(lut_mse_vals)),
        lut_mse_std=float(np.std(lut_mse_vals)),
        lut_mae_mean=float(np.mean(lut_mae_vals)),
        lut_mae_std=float(np.std(lut_mae_vals)),
        lut_mse_values=lut_mse_vals,
        lut_mae_values=lut_mae_vals,
        ratio_mse=poly_res["mse"] / (float(np.mean(lut_mse_vals)) + 1e-30),
        ratio_mae=poly_res["mae_phys"] / (float(np.mean(lut_mae_vals)) + 1e-30),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Experiment A: Per-unit calibration across all sensor types
# ─────────────────────────────────────────────────────────────────────────────

def exp_A_per_sensor():
    """Main calibration experiment: 8 units × 3 sensor types."""
    print("\n=== Experiment A: Per-sensor calibration ===")
    K, L = CFG["K"], CFG["L"]
    results = {}

    # ── NTC ─────────────────────────────────────────────────────────────────
    print("  NTC thermistor...")
    ntc_units = make_ntc_units(CFG["N_UNITS"], seed=CFG["SENSOR_SEED"])
    ntc_rows = []
    for i, unit in enumerate(ntc_units):
        data = ntc_calibration_data(unit, CFG["N_CAL"], CFG["N_VAL"], CFG["N_TEST"],
                                     seed=CFG["SENSOR_SEED"] + i)
        # Factory baseline (no calibration)
        factory = ntc_factory_baseline(unit, data)
        res = run_calibration(data, K, L, CFG["lambda_2"],
                               CFG["POLY_DEGREE_NTC"], CFG["TRAIN_SEEDS"],
                               CFG["lr"], CFG["epochs"], CFG["batch_size"])
        row = dict(
            label=unit.label, B=round(unit.B, 1),
            factory_mae=round(factory["mae_phys"], 4),
            poly_mae=round(res["poly_mae_phys"], 4),
            lut_mae=round(res["lut_mae_mean"], 4),
            lut_mae_std=round(res["lut_mae_std"], 4),
            improvement_vs_factory=round(factory["mae_phys"] / (res["lut_mae_mean"] + 1e-10), 2),
            improvement_vs_poly=round(res["poly_mae_phys"] / (res["lut_mae_mean"] + 1e-10), 2),
        )
        ntc_rows.append(row)
        print(f"    {unit.label}: factory {factory['mae_phys']:.3f}°C → poly {res['poly_mae_phys']:.3f}°C → LUT {res['lut_mae_mean']:.3f}°C")
    results["ntc"] = ntc_rows

    # ── MQ Gas ───────────────────────────────────────────────────────────────
    print("  MQ gas sensor...")
    mq_units = make_mq_units(CFG["N_UNITS"], seed=CFG["SENSOR_SEED"])
    mq_rows = []
    for i, unit in enumerate(mq_units):
        data = mq_calibration_data(unit, CFG["N_CAL"], CFG["N_VAL"], CFG["N_TEST"],
                                    seed=CFG["SENSOR_SEED"] + i)
        res = run_calibration(data, K, L, CFG["lambda_2"],
                               CFG["POLY_DEGREE_MQ"], CFG["TRAIN_SEEDS"],
                               CFG["lr"], CFG["epochs"], CFG["batch_size"])
        row = dict(
            label=unit.label,
            A=round(unit.A, 3), B_exp=round(unit.B_exp, 3),
            poly_mae=round(res["poly_mae_phys"], 4),
            lut_mae=round(res["lut_mae_mean"], 4),
            lut_mae_std=round(res["lut_mae_std"], 4),
            improvement_vs_poly=round(res["poly_mae_phys"] / (res["lut_mae_mean"] + 1e-10), 2),
        )
        mq_rows.append(row)
        print(f"    {unit.label}: poly {res['poly_mae_phys']:.4f} → LUT {res['lut_mae_mean']:.4f} (log10 ppm)")
    results["mq"] = mq_rows

    # ── pH ───────────────────────────────────────────────────────────────────
    print("  pH electrode...")
    ph_units = make_ph_units(CFG["N_UNITS"], seed=CFG["SENSOR_SEED"])
    ph_rows = []
    for i, unit in enumerate(ph_units):
        data = ph_calibration_data(unit, CFG["N_CAL"], CFG["N_VAL"], CFG["N_TEST"],
                                    seed=CFG["SENSOR_SEED"] + i)
        res = run_calibration(data, K, L, CFG["lambda_2"],
                               CFG["POLY_DEGREE_PH"], CFG["TRAIN_SEEDS"],
                               CFG["lr"], CFG["epochs"], CFG["batch_size"])
        row = dict(
            label=unit.label,
            S=round(unit.S, 3), E0=round(unit.E0, 2),
            poly_mae=round(res["poly_mae_phys"], 4),
            lut_mae=round(res["lut_mae_mean"], 4),
            lut_mae_std=round(res["lut_mae_std"], 4),
            improvement_vs_poly=round(res["poly_mae_phys"] / (res["lut_mae_mean"] + 1e-10), 2),
        )
        ph_rows.append(row)
        print(f"    {unit.label}: poly {res['poly_mae_phys']:.4f} → LUT {res['lut_mae_mean']:.4f} pH units")
    results["ph"] = ph_rows

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment B: N_cal sweep
# ─────────────────────────────────────────────────────────────────────────────

def exp_B_ncal_sweep():
    """How many calibration points are needed? Sweep N_cal for all sensors."""
    print("\n=== Experiment B: N_cal sweep ===")
    K, L = CFG["K"], CFG["L"]
    sweep = {}

    ntc_unit = make_ntc_units(1, seed=CFG["SENSOR_SEED"])[0]
    mq_unit  = make_mq_units(1,  seed=CFG["SENSOR_SEED"])[0]
    ph_unit  = make_ph_units(1,  seed=CFG["SENSOR_SEED"])[0]

    for sensor_name, unit, data_fn, poly_deg in [
        ("ntc", ntc_unit, ntc_calibration_data, CFG["POLY_DEGREE_NTC"]),
        ("mq",  mq_unit,  mq_calibration_data,  CFG["POLY_DEGREE_MQ"]),
        ("ph",  ph_unit,  ph_calibration_data,   CFG["POLY_DEGREE_PH"]),
    ]:
        print(f"  {sensor_name}...")
        rows = []
        for n_cal in CFG["N_CAL_VALUES"]:
            if n_cal <= poly_deg + 1:
                # Can't fit polynomial with fewer points than parameters
                rows.append(dict(n_cal=n_cal, poly_mae=None, lut_mae=None, lut_std=None))
                continue
            n_val = max(5, n_cal // 4)
            data = data_fn(unit, n_cal=n_cal, n_val=n_val, n_test=200,
                           seed=CFG["SENSOR_SEED"])
            res = run_calibration(data, K, L, CFG["lambda_2"], poly_deg,
                                   CFG["TRAIN_SEEDS"], CFG["lr"], CFG["epochs"],
                                   CFG["batch_size"])
            rows.append(dict(
                n_cal=n_cal,
                poly_mae=round(res["poly_mae_phys"], 5),
                lut_mae=round(res["lut_mae_mean"], 5),
                lut_std=round(res["lut_mae_std"], 5),
            ))
            print(f"    n_cal={n_cal}: poly {res['poly_mae_phys']:.4f} → LUT {res['lut_mae_mean']:.4f}")
        sweep[sensor_name] = rows

    return sweep


# ─────────────────────────────────────────────────────────────────────────────
# Experiment C: Coverage rule K*L < n_train
# ─────────────────────────────────────────────────────────────────────────────

def exp_C_coverage_rule():
    """
    Validate K*L < n_train empirically.
    Use NTC unit; sweep K,L and n_train; report whether training is stable.
    Stable = LUT improves over zero-init baseline.
    """
    print("\n=== Experiment C: Coverage rule ===")
    ntc_unit = make_ntc_units(1, seed=0)[0]
    rows = []

    for (K, L) in CFG["KL_VALUES"]:
        kl = K * L
        for n_train in CFG["N_TRAIN_VALUES"]:
            if n_train < 5:
                continue
            n_val = max(5, n_train // 4)
            data = ntc_calibration_data(ntc_unit, n_cal=n_train, n_val=n_val,
                                         n_test=200, seed=99)
            try:
                res = run_calibration(data, K, L, 1.0, 3,
                                       [0, 1], 0.01, 500, min(32, n_train))
                stable = res["lut_mse_mean"] < res["poly_mse"]
                rows.append(dict(K=K, L=L, KL=kl, n_train=n_train,
                                  ratio=kl / n_train,
                                  lut_mse=res["lut_mse_mean"],
                                  poly_mse=res["poly_mse"],
                                  stable=stable))
                print(f"    K={K} L={L} KL={kl} n={n_train} ratio={kl/n_train:.2f} → {'stable' if stable else 'UNSTABLE'}")
            except Exception as e:
                print(f"    K={K} L={L} n={n_train}: error {e}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_improvement_bars(results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    sensors = [
        ("ntc", "NTC Thermistor", "MAE (°C)", "factory_mae"),
        ("mq",  "MQ Gas Sensor",  "MAE (log₁₀ ppm)", None),
        ("ph",  "pH Electrode",   "MAE (pH units)", None),
    ]
    for ax, (key, title, ylabel, factory_key) in zip(axes, sensors):
        rows = results[key]
        labels = [r["label"] for r in rows]
        poly_maes = [r["poly_mae"] for r in rows]
        lut_maes  = [r["lut_mae"] for r in rows]
        x = np.arange(len(labels))
        w = 0.35
        if factory_key and factory_key in rows[0]:
            factory_maes = [r[factory_key] for r in rows]
            ax.bar(x - w, factory_maes, w, label="Factory (no cal)", color="#B4B2A9", alpha=0.85)
            ax.bar(x,     poly_maes,    w, label=f"Poly baseline",   color="#AFA9EC", alpha=0.85)
            ax.bar(x + w, lut_maes,     w, label="Direct LUT",       color="#5DCAA5", alpha=0.85)
        else:
            ax.bar(x - w/2, poly_maes, w, label="Poly baseline", color="#AFA9EC", alpha=0.85)
            ax.bar(x + w/2, lut_maes,  w, label="Direct LUT",    color="#5DCAA5", alpha=0.85)

        mean_imp = np.mean([r["improvement_vs_poly"] for r in rows])
        ax.set_title(f"{title}\n(mean improvement {mean_imp:.1f}×)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / "improvement_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT / 'improvement_bars.png'}")


def plot_ncal_sweep(sweep: dict):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sensor_meta = {
        "ntc": ("NTC Thermistor", "MAE (°C)"),
        "mq":  ("MQ Gas Sensor",  "MAE (log₁₀ ppm)"),
        "ph":  ("pH Electrode",   "MAE (pH units)"),
    }
    for ax, (key, (title, ylabel)) in zip(axes, sensor_meta.items()):
        rows = [r for r in sweep[key] if r["poly_mae"] is not None]
        n_vals = [r["n_cal"] for r in rows]
        poly_maes = [r["poly_mae"] for r in rows]
        lut_maes  = [r["lut_mae"]  for r in rows]
        lut_stds  = [r["lut_std"]  for r in rows]
        ax.plot(n_vals, poly_maes, "o--", color="#534AB7", label="Polynomial")
        ax.errorbar(n_vals, lut_maes, yerr=lut_stds, fmt="s-",
                    color="#0F6E56", capsize=3, label="Direct LUT")
        ax.axvline(x=4*8, color="#E24B4A", linestyle=":", alpha=0.7, label="K×L = 32")
        ax.set_xlabel("N_cal (calibration points)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(OUT / "ncal_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT / 'ncal_sweep.png'}")


def plot_coverage_rule(coverage_rows: list):
    if not coverage_rows:
        return
    stable   = [r for r in coverage_rows if r["stable"]]
    unstable = [r for r in coverage_rows if not r["stable"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    if stable:
        ax.scatter([r["KL"] for r in stable],   [r["n_train"] for r in stable],
                   color="#1D9E75", s=80, label="Stable (LUT > poly)", zorder=3)
    if unstable:
        ax.scatter([r["KL"] for r in unstable], [r["n_train"] for r in unstable],
                   color="#E24B4A", s=80, marker="x", label="Unstable (LUT ≤ poly)", zorder=3)

    # Draw K*L = n_train diagonal
    kl_max = max(r["KL"] for r in coverage_rows) * 1.2
    xs = np.linspace(1, kl_max, 200)
    ax.plot(xs, xs, "k--", alpha=0.5, label="K×L = n_train")
    ax.fill_between(xs, xs, kl_max * 1.5, alpha=0.07, color="red")
    ax.fill_between(xs, 0, xs, alpha=0.07, color="green")

    ax.set_xlabel("K × L (LUT cells)", fontsize=11)
    ax.set_ylabel("n_train (calibration points)", fontsize=11)
    ax.set_title("Coverage rule: K×L < n_train", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, kl_max)

    plt.tight_layout()
    fig.savefig(OUT / "coverage_rule.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT / 'coverage_rule.png'}")


def save_csv_table(results: dict):
    """Save per-sensor summary as CSV for Table III equivalent."""
    import csv
    for key, rows in results.items():
        path = OUT / f"table_{key}.csv"
        if not rows:
            continue
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    print(f"  CSV tables saved in {OUT}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("LUT-KAN Sensor Calibration Study — v13")
    print(f"Config: K={CFG['K']}, L={CFG['L']}, N_CAL={CFG['N_CAL']}")

    # Experiment A
    results_A = exp_A_per_sensor()
    save_csv_table(results_A)
    plot_improvement_bars(results_A)

    # Experiment B
    sweep_B = exp_B_ncal_sweep()
    with open(OUT / "ncal_sweep.json", "w") as f:
        json.dump(sweep_B, f, indent=2)
    plot_ncal_sweep(sweep_B)

    # Experiment C
    coverage_C = exp_C_coverage_rule()
    with open(OUT / "coverage_rule.json", "w") as f:
        json.dump(coverage_C, f, indent=2)
    plot_coverage_rule(coverage_C)

    # Master summary
    summary = dict(config=CFG, sensor_results=results_A,
                   ncal_sweep=sweep_B, coverage_rule=coverage_C)
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll results saved to {OUT}")

    # Print summary table
    print("\n=== Summary: Mean improvement vs polynomial baseline ===")
    for sensor_key, label in [("ntc","NTC (°C)"), ("mq","MQ (log ppm)"), ("ph","pH units")]:
        rows = results_A[sensor_key]
        mean_lut = np.mean([r["lut_mae"] for r in rows])
        mean_poly = np.mean([r["poly_mae"] for r in rows])
        mean_imp = mean_poly / (mean_lut + 1e-10)
        print(f"  {label:18s}: poly {mean_poly:.4f} → LUT {mean_lut:.4f}  ({mean_imp:.1f}×)")
