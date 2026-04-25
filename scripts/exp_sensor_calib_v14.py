"""
exp_sensor_calib_v14.py
=======================
Multi-sensor calibration regime study for IEEE TIM v14.

Experiments
-----------
A  NTC extended:   6 methods x 8 units (synthetic B-values around 3950 K)
B  Multi-sensor:   6 sensor types, regime classification table
C  N_cal sweep:    NTC, Type K TC, LDR — accuracy vs. number of cal. points
D  L sweep:        NTC — accuracy vs. LUT table size L in {8, 16, 32, 64}

Key findings (confirmed by experiment)
---------------------------------------
LUT wins  (ratio > 1, p < 0.01): NTC 1.64x, LDR 2.37x, Humidity 1.58x
Poly pref (ratio < 1, ns)       : Type K TC 0.85x, MQ gas 0.82x, pH 0.40x

L sweep reveals that coverage rule (L < N_cal) is necessary but NOT sufficient:
  L=8  → 2.05 °C  (7.7x worse than poly!)
  L=16 → 0.63 °C  (2.4x worse than poly)
  L=32 → 0.18 °C  (LUT wins, 1.51x)    ← minimum recommended
  L=64 → 0.09 °C  (LUT wins, 3.12x)

Fixes from v13
--------------
- Type K: T_range=(0, 350), V_bias=0.27 V — ADC now strictly monotone
- SHH baseline: two modes (3 ideal noiseless pts vs 50 noisy pts)
- Single-point offset added as new NTC baseline
- L sweep added: quantifies minimum table size requirement
- PDF export: every figure saved as both .png and .pdf

Usage
-----
    python scripts/exp_sensor_calib_v14.py           # full run (~15-30 min)
    python scripts/exp_sensor_calib_v14.py --quick   # smoke test (~3 min)

Outputs
-------
    results/sensor_calib_v14/
        summary_v14.json            all numeric results
        per_sensor_table.csv        Table II (paper)
        ntc_extended_table.csv      Table I extended (paper)
        sensor_curves.{png,pdf}     Fig NEW-1: ADC response curves
        regime_map.{png,pdf}        Fig 2: improvement ratio + CI
        ntc_extended.{png,pdf}      Fig 1: 6-method NTC comparison
        ncal_sweep.{png,pdf}        Fig NEW-2: accuracy vs N_cal
        l_sweep_ntc.{png,pdf}       Fig NEW-3: accuracy vs L

Notes
-----
NTC numbers in this script use SYNTHETIC B-values (uniform around 3950 K).
Table I in the paper uses REAL manufacturer R/T data (Adafruit, Vector
Controls, Vishay, TDK, EPCOS, Semitec) with ratio 1.80x — see v13 results.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lut_native.core import lut_forward_numpy
from lut_native.training import TrainConfig, train_lut_edge
from lut_native.sensors import (
    # NTC thermistor
    make_ntc_units, ntc_calibration_data, ntc_factory_baseline,
    fit_shh_baseline, fit_onepoint_offset_baseline, fit_poly_baseline,
    # Type K thermocouple (fixed: T_range=0..350, V_bias)
    make_typek_units, typek_calibration_data, typek_factory_baseline,
    # LDR photoresistor (new in v14)
    make_ldr_units, ldr_calibration_data,
    # remaining parametric sensors
    make_mq_units, mq_calibration_data,
    make_ph_units, ph_calibration_data,
    make_humidity_units, humidity_calibration_data, humidity_factory_baseline,
    nist_k_emf,
)

OUT = ROOT / "results" / "sensor_calib_v14"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Color scheme (consistent with H1–H5 experiments)
# ---------------------------------------------------------------------------
C = dict(
    factory="#888780",
    offset="#EF9F27",
    poly="#E24B4A",
    shh_ideal="#7F77DD",
    shh_noisy="#3B8BD4",
    lut="#1D9E75",
    lut_win="#1D9E75",
    poly_pref="#E24B4A",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_cfg(quick: bool = False) -> dict:
    """Return experiment configuration. --quick reduces seeds/epochs for CI."""
    return dict(
        K=1, L=32,
        lambda_2=1.0, lr=0.01,
        epochs=600 if quick else 1000,
        batch_size=32,
        N_CAL=50, N_VAL=25, N_TEST=200, N_UNITS=8,
        POLY_DEGREES=dict(ntc=5, typek=5, ldr=4, humidity=5, mq=4, ph=1),
        SENSOR_SEED=42,
        TRAIN_SEEDS=[0, 1] if quick else [0, 1, 2, 3, 4],
        N_CAL_VALUES=[40, 50, 75, 100, 150] if quick else
                     [10, 15, 20, 30, 40, 50, 75, 100, 150],
        L_VALUES=[8, 16, 32, 64],
        quick=quick,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, stem: str) -> None:
    """Save figure as both PNG (150 dpi) and PDF in OUT."""
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    print(f"  Saved: {stem}.{{png,pdf}}")


def _train_lut(data: dict, cfg: dict, seeds: list) -> list[float]:
    """Train LUT for each seed; return list of physical-unit MAE values."""
    lut_init = np.zeros((cfg["K"], cfg["L"]), dtype=np.float32)
    tc = TrainConfig(
        lambda_2=cfg["lambda_2"], lr=cfg["lr"],
        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        eval_every_epochs=25,
    )
    maes = []
    for seed in seeds:
        tc.seed = seed
        res = train_lut_edge(
            lut_init=lut_init.copy(),
            x_train=data["x_cal"], y_train=data["y_cal"],
            x_val=data["x_val"],   y_val=data["y_val"],
            x_test=data["x_test"], y_test=data["y_test"],
            x_min=-1.0, x_max=1.0, cfg=tc,
        )
        y_lut = lut_forward_numpy(data["x_test"], res.lut_best)
        mae = float(np.mean(np.abs(
            data["denorm_y"](y_lut) - data["denorm_y"](data["y_test"])
        )))
        maes.append(mae)
    return maes


def _run_sensor(make_fn, calib_fn, cfg, poly_deg,
                factory_fn=None, seed_offset: int = 0) -> dict:
    """Run calibration study for one sensor type: all 8 units, poly + LUT."""
    units = make_fn(cfg["N_UNITS"], seed=cfg["SENSOR_SEED"] + seed_offset)
    poly_maes, lut_maes, lut_stds, fac_maes = [], [], [], []
    for i, unit in enumerate(units):
        data = calib_fn(unit, cfg["N_CAL"], cfg["N_VAL"], cfg["N_TEST"],
                        seed=cfg["SENSOR_SEED"] + i)
        poly = fit_poly_baseline(data["x_cal"], data["y_cal"], poly_deg,
                                 data["x_test"], data["y_test"], data["denorm_y"])
        maes = _train_lut(data, cfg, cfg["TRAIN_SEEDS"])
        poly_maes.append(poly["mae_phys"])
        lut_maes.append(float(np.mean(maes)))
        lut_stds.append(float(np.std(maes)))
        if factory_fn:
            fac_maes.append(factory_fn(unit, data)["mae_phys"])

    from scipy import stats as sp
    try:
        _, p = sp.wilcoxon(poly_maes, lut_maes, alternative="greater")
    except Exception:
        p = float("nan")

    poly_m = float(np.mean(poly_maes))
    lut_m  = float(np.mean(lut_maes))
    return dict(
        poly=poly_m, lut=lut_m,
        lut_std=float(np.mean(lut_stds)),
        factory=float(np.mean(fac_maes)) if fac_maes else None,
        ratio=poly_m / (lut_m + 1e-30),
        p=float(p),
        poly_values=poly_maes,
        lut_values=lut_maes,
    )


# ---------------------------------------------------------------------------
# Experiment A — NTC extended comparison (6 methods)
# ---------------------------------------------------------------------------

def exp_A_ntc_extended(cfg: dict) -> list[dict]:
    """
    NTC: 6 calibration methods x 8 units with synthetic B-values.

    Methods: factory | one-point offset | poly-5 |
             SHH (3 ideal) | SHH (50 noisy) | LUT (K=1, L=32)
    """
    print("\n=== Experiment A: NTC extended comparison (6 methods) ===")
    ntc_units = make_ntc_units(cfg["N_UNITS"], seed=cfg["SENSOR_SEED"])
    rows = []
    for i, unit in enumerate(ntc_units):
        data = ntc_calibration_data(
            unit, cfg["N_CAL"], cfg["N_VAL"], cfg["N_TEST"],
            seed=cfg["SENSOR_SEED"] + i)

        fac_res   = ntc_factory_baseline(unit, data)
        off_res   = fit_onepoint_offset_baseline(
            data["x_cal"], data["y_cal"],
            data["x_test"], data["y_test"], data["denorm_y"])
        poly_res  = fit_poly_baseline(
            data["x_cal"], data["y_cal"], 5,
            data["x_test"], data["y_test"], data["denorm_y"])
        shh_ideal = fit_shh_baseline(
            data["x_cal"], data["y_cal"],
            data["x_test"], data["y_test"],
            data["denorm_y"], data["denorm_x"], unit, use_noisy=False)
        shh_noisy = fit_shh_baseline(
            data["x_cal"], data["y_cal"],
            data["x_test"], data["y_test"],
            data["denorm_y"], data["denorm_x"], unit, use_noisy=True)
        lut_maes  = _train_lut(data, cfg, cfg["TRAIN_SEEDS"])

        lut_mean = float(np.mean(lut_maes))
        lut_std  = float(np.std(lut_maes))
        ratio    = poly_res["mae_phys"] / (lut_mean + 1e-30)

        row = dict(
            label=unit.label,
            B=round(unit.B, 1),
            factory_mae=round(fac_res["mae_phys"], 4),
            offset_mae=round(off_res["mae_phys"], 4),
            poly_mae=round(poly_res["mae_phys"], 4),
            shh_ideal_mae=round(shh_ideal["mae_phys"], 5),
            shh_noisy_mae=round(shh_noisy["mae_phys"], 4),
            lut_mae=round(lut_mean, 4),
            lut_mae_std=round(lut_std, 4),
        )
        rows.append(row)
        print(
            f"  {unit.label}  B={unit.B:.0f}: "
            f"fac={fac_res['mae_phys']:.3f}  "
            f"off={off_res['mae_phys']:.3f}  "
            f"poly={poly_res['mae_phys']:.3f}  "
            f"SHH_i={shh_ideal['mae_phys']:.5f}  "
            f"SHH_n={shh_noisy['mae_phys']:.4f}  "
            f"LUT={lut_mean:.4f} +/-{lut_std:.4f} C  "
            f"ratio={ratio:.2f}x"
        )
    return rows


# ---------------------------------------------------------------------------
# Experiment B — Multi-sensor regime (6 types)
# ---------------------------------------------------------------------------

def exp_B_multi_sensor(cfg: dict) -> dict:
    """
    Regime table for 6 sensor types.

    Confirmed results:
      LUT wins:    NTC 1.64x **, LDR 2.37x **, Humidity 1.58x **
      Poly pref:   Type K 0.85x ns, MQ 0.82x ns, pH 0.40x ns

    Note: Type K (T_range=0..350, NIST ITS-90 EMF) has smooth poly-like
    response in this range; poly-5 achieves ~0.04 C residual. LUT provides
    no benefit — correctly classified as poly preferred.
    """
    print("\n=== Experiment B: Multi-sensor regime (6 types) ===")
    results = {}

    print("  NTC thermistor (exponential R/T)...")
    results["ntc"] = _run_sensor(
        make_ntc_units, ntc_calibration_data, cfg,
        cfg["POLY_DEGREES"]["ntc"],
        factory_fn=ntc_factory_baseline)

    print("  Type K thermocouple (NIST ITS-90, T_range=0..350 C)...")
    results["typek"] = _run_sensor(
        make_typek_units, typek_calibration_data, cfg,
        cfg["POLY_DEGREES"]["typek"],
        factory_fn=typek_factory_baseline, seed_offset=10)

    print("  LDR photoresistor (power-law R/lux, GL55-series)...")
    results["ldr"] = _run_sensor(
        make_ldr_units, ldr_calibration_data, cfg,
        cfg["POLY_DEGREES"]["ldr"], seed_offset=20)

    print("  Resistive humidity sensor (exponential R/RH)...")
    results["humidity"] = _run_sensor(
        make_humidity_units, humidity_calibration_data, cfg,
        cfg["POLY_DEGREES"]["humidity"],
        factory_fn=humidity_factory_baseline, seed_offset=30)

    print("  MQ-type gas sensor (power-law Rs/R0 vs concentration)...")
    results["mq"] = _run_sensor(
        make_mq_units, mq_calibration_data, cfg,
        cfg["POLY_DEGREES"]["mq"], seed_offset=40)

    print("  pH electrode (Nernst, linear)...")
    results["ph"] = _run_sensor(
        make_ph_units, ph_calibration_data, cfg,
        cfg["POLY_DEGREES"]["ph"], seed_offset=50)

    print()
    for k, v in results.items():
        sig  = "** p<0.01" if v["p"] < 0.01 else \
               ("*  p<0.05" if v["p"] < 0.05 else f"ns p={v['p']:.3f}")
        regime = "LUT WINS" if (v["ratio"] > 1.0 and v["p"] < 0.05) \
                 else "poly preferred"
        fac_s = f"{v['factory']:.3f}" if v["factory"] else "  ---  "
        print(
            f"  {k:10s}  fac={fac_s}  "
            f"poly={v['poly']:.4f}  "
            f"lut={v['lut']:.4f}+/-{v['lut_std']:.4f}  "
            f"ratio={v['ratio']:.3f}x  [{sig}]  => {regime}"
        )
    return results


# ---------------------------------------------------------------------------
# Experiment C — N_cal sweep: accuracy vs. number of calibration points
# ---------------------------------------------------------------------------

def exp_C_ncal_sweep(cfg: dict) -> dict:
    """
    MAE vs. N_cal for NTC, Type K, and LDR at K=1, L=32.

    NTC and LDR consistently show LUT > poly from N_cal >= 40.
    Type K consistently shows poly > LUT regardless of N_cal,
    confirming it belongs in the 'poly preferred' regime.
    """
    print("\n=== Experiment C: N_cal sweep (NTC, TypeK, LDR) ===")
    K, L = cfg["K"], cfg["L"]
    sweep_seeds = cfg["TRAIN_SEEDS"][:3]

    def _sweep(unit, calib_fn, poly_deg, n_values: list) -> dict:
        rows = {}
        for n in n_values:
            if n < K * L:
                # Coverage rule violated — skip
                rows[n] = dict(poly=None, lut=None, lut_std=None)
                continue
            try:
                data = calib_fn(unit, n_cal=n, n_val=25, n_test=200, seed=99)
                poly = fit_poly_baseline(
                    data["x_cal"], data["y_cal"], poly_deg,
                    data["x_test"], data["y_test"], data["denorm_y"])
                maes = _train_lut(data, cfg, sweep_seeds)
                rows[n] = dict(
                    poly=float(poly["mae_phys"]),
                    lut=float(np.mean(maes)),
                    lut_std=float(np.std(maes)),
                )
            except Exception as exc:
                rows[n] = dict(poly=None, lut=None, lut_std=None,
                               error=str(exc))
        return rows

    ntc_unit   = make_ntc_units(1,   seed=cfg["SENSOR_SEED"])[0]
    typek_unit = make_typek_units(1, seed=cfg["SENSOR_SEED"] + 10)[0]
    ldr_unit   = make_ldr_units(1,   seed=cfg["SENSOR_SEED"] + 20)[0]

    sweep = dict(
        ntc=_sweep(ntc_unit,   ntc_calibration_data,   cfg["POLY_DEGREES"]["ntc"],
                   cfg["N_CAL_VALUES"]),
        typek=_sweep(typek_unit, typek_calibration_data, cfg["POLY_DEGREES"]["typek"],
                     cfg["N_CAL_VALUES"]),
        ldr=_sweep(ldr_unit,   ldr_calibration_data,   cfg["POLY_DEGREES"]["ldr"],
                   cfg["N_CAL_VALUES"]),
    )

    for sensor, data in sweep.items():
        print(f"  {sensor}:")
        for n, v in data.items():
            if v.get("lut") is not None:
                r = v["poly"] / (v["lut"] + 1e-30)
                winner = "LUT" if v["lut"] < v["poly"] else "poly"
                print(f"    N={n:3d}: poly={v['poly']:.4f}  "
                      f"lut={v['lut']:.4f}+/-{v.get('lut_std',0):.4f}  "
                      f"ratio={r:.2f}x  [{winner} wins]")
    return sweep


# ---------------------------------------------------------------------------
# Experiment D — L sweep: accuracy vs. table size L (NTC)
# ---------------------------------------------------------------------------

def exp_D_lsize_sweep(cfg: dict) -> dict:
    """
    MAE vs. L for NTC at K=1, N_cal=50.

    Result: L=8, L=16 perform WORSE than polynomial.
    Coverage rule K*L < N_cal is necessary but not sufficient:
    L must also be large enough to represent the target curve shape.
    Minimum recommended: L=32 for NTC at N_cal=50.
    """
    print("\n=== Experiment D: L sweep (NTC, K=1, N_cal=50) ===")
    unit = make_ntc_units(1, seed=cfg["SENSOR_SEED"])[0]
    data = ntc_calibration_data(
        unit, cfg["N_CAL"], cfg["N_VAL"], cfg["N_TEST"],
        seed=cfg["SENSOR_SEED"])
    poly = fit_poly_baseline(
        data["x_cal"], data["y_cal"], 5,
        data["x_test"], data["y_test"], data["denorm_y"])

    rows = {}
    for L in cfg["L_VALUES"]:
        cfg_L = {**cfg, "L": L}
        maes = _train_lut(data, cfg_L, cfg["TRAIN_SEEDS"][:3])
        lut_mean = float(np.mean(maes))
        lut_std  = float(np.std(maes))
        winner   = "LUT wins" if lut_mean < poly["mae_phys"] else \
                   f"poly wins (LUT {lut_mean/poly['mae_phys']:.1f}x worse)"
        rows[L] = dict(
            poly=poly["mae_phys"],
            lut=lut_mean,
            lut_std=lut_std,
            bytes_lut=L,
        )
        print(f"  L={L:2d} ({L:2d} bytes): "
              f"lut={lut_mean:.4f}+/-{lut_std:.4f}  "
              f"poly={poly['mae_phys']:.4f}  [{winner}]")
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_sensor_curves() -> None:
    """
    Figure NEW-1: ADC output vs. physical quantity for all 6 sensor types.
    Color encodes regime: green = LUT wins, red = poly preferred.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    sensors = [
        ("NTC thermistor",          "T (deg C)",       [0, 100],    C["lut_win"],
         lambda T: make_ntc_units(1, seed=42)[0].adc_from_T(np.array(T))),
        ("Type K thermocouple\n(NIST ITS-90, 0-350 C)", "T (deg C)", [0, 350], C["poly_pref"],
         lambda T: make_typek_units(1, seed=10)[0].adc_from_T(np.array(T))),
        ("LDR photoresistor\n(GL55-series)",  "log10(lux)", [0, 4],      C["lut_win"],
         lambda lE: make_ldr_units(1, seed=6)[0].adc_from_E(10.0**np.array(lE))),
        ("Humidity (resistive)",    "RH (%)",          [10, 90],    C["lut_win"],
         lambda RH: make_humidity_units(1, seed=5)[0].adc_from_RH(np.array(RH))),
        ("MQ gas sensor",           "log10(C) [ppm]",  [2, 4],      C["poly_pref"],
         lambda lC: make_mq_units(1, seed=2)[0].adc_from_C(10.0**np.array(lC))),
        ("pH electrode",            "pH",              [2, 12],     C["poly_pref"],
         lambda pH: make_ph_units(1, seed=3)[0].adc_from_pH(np.array(pH))),
    ]

    for ax, (name, xlabel, xlim, color, fn) in zip(axes, sensors):
        x = np.linspace(xlim[0], xlim[1], 300)
        y = fn(x.tolist())
        ax.plot(x, y, color=color, lw=2.2)
        # Shade deviation from linear to visualise nonlinearity
        y_lin = np.linspace(float(y[0]), float(y[-1]), len(y))
        ax.fill_between(x, y, y_lin, alpha=0.10, color=color)
        nl = float(np.max(np.abs(y - y_lin)))
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("ADC output", fontsize=10)
        ax.set_title(name, fontsize=10, fontweight="bold", color=color)
        ax.text(0.97, 0.05, f"max nonlinearity: {nl:.0f} counts",
                transform=ax.transAxes, ha="right", fontsize=8, color="gray")
        ax.grid(True, alpha=0.25, lw=0.5)
        # Regime label
        regime = "LUT wins" if color == C["lut_win"] else "Poly preferred"
        ax.text(0.03, 0.97, regime, transform=ax.transAxes,
                va="top", fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=color, alpha=0.8))

    fig.suptitle(
        "Sensor ADC response curves: ADC output = f(physical quantity)",
        fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig(fig, "sensor_curves")
    plt.close()


def fig_regime_map(multi: dict) -> None:
    """
    Figure 2: Improvement ratio poly_MAE / LUT_MAE with significance.
    Green bars: LUT wins. Red bars: poly preferred. Break-even at 1.0x.
    """
    order = [
        ("ldr",      "LDR photoresistor",         C["lut_win"]),
        ("ntc",      "NTC thermistor",             C["lut_win"]),
        ("humidity", "Humidity (resistive)",       C["lut_win"]),
        ("typek",    "Type K thermocouple",        C["poly_pref"]),
        ("mq",       "MQ gas sensor",              C["poly_pref"]),
        ("ph",       "pH electrode",               C["poly_pref"]),
    ]
    ratios = [multi[k]["ratio"] for k, _, _ in order]
    pvals  = [multi[k]["p"]     for k, _, _ in order]
    labels = [lbl                for _, lbl, _ in order]
    colors = [col                for _, _, col in order]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(len(order))
    ax.barh(y_pos, ratios, color=colors, alpha=0.82, height=0.6)
    ax.axvline(1.0, color="black", lw=1.5, ls="--",
               label="Break-even (1.0x)")

    for i, (r, p) in enumerate(zip(ratios, pvals)):
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        ax.text(r + 0.04, i, f"{r:.2f}x  {sig}",
                va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(
        "Improvement ratio  poly MAE / LUT MAE  "
        "(> 1.0 means LUT wins)", fontsize=10)
    ax.set_title(
        "Regime map: LUT vs. polynomial calibration "
        r"($K{=}1$, $L{=}32$, $N_\mathrm{cal}{=}50$)",
        fontsize=11)
    ax.set_xlim(0, max(ratios) * 1.30)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25, lw=0.5)

    # Regime labels on left
    for i, (k, _, color) in enumerate(order):
        txt = "LUT wins" if color == C["lut_win"] else "poly pref."
        ax.text(-0.04, i, txt, va="center", ha="right",
                fontsize=8, color=color,
                transform=ax.get_yaxis_transform())

    plt.tight_layout()
    _savefig(fig, "regime_map")
    plt.close()


def fig_ntc_extended(ntc_rows: list) -> None:
    """
    Figure 1: Six-method NTC calibration comparison (bar chart).
    8 synthetic NTC units ordered by B-coefficient.
    Dashed line marks poly-5 mean for visual reference.
    """
    labels  = [f"{r['label']}\nB={r['B']:.0f} K" for r in ntc_rows]
    methods = [
        ("factory_mae",   "Factory",           C["factory"]),
        ("offset_mae",    "One-point offset",  C["offset"]),
        ("poly_mae",      "Poly-5",            C["poly"]),
        ("shh_ideal_mae", "SHH (3 ideal pts)", C["shh_ideal"]),
        ("shh_noisy_mae", "SHH (50 noisy pts)",C["shh_noisy"]),
        ("lut_mae",       "LUT (K=1, L=32)",   C["lut"]),
    ]
    x     = np.arange(len(ntc_rows))
    width = 0.13
    fig, ax = plt.subplots(figsize=(13, 5))

    for j, (key, label, color) in enumerate(methods):
        vals   = [r[key] for r in ntc_rows]
        offset = j * width - 2.5 * width
        ax.bar(x + offset, vals, width, label=label,
               color=color, alpha=0.85, zorder=3)

    # Poly mean reference line
    poly_mean = float(np.mean([r["poly_mae"] for r in ntc_rows]))
    ax.axhline(poly_mean, color=C["poly"], ls="--", lw=1.0,
               alpha=0.5, zorder=2, label=f"Poly mean ({poly_mean:.3f} C)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MAE (deg C)", fontsize=11)
    ax.set_title(
        "NTC thermistor: all calibration methods\n"
        r"(50 noisy cal. points, $K{=}1$, $L{=}32$, 5 seeds)",
        fontsize=11)
    ax.legend(fontsize=8, ncols=7, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25, lw=0.5, zorder=1)
    ax.set_axisbelow(True)

    # Note about SHH ideal
    ax.text(0.01, 0.97,
            "SHH (3 ideal): near-zero for synthetic data (exact B recovery);\n"
            "real manufacturer data: 0.0042 deg C (see Table I)",
            transform=ax.transAxes, va="top", fontsize=7,
            color="gray", style="italic")

    plt.tight_layout()
    _savefig(fig, "ntc_extended")
    plt.close()


def fig_ncal_sweep(sweep: dict) -> None:
    """
    Figure NEW-2: MAE vs. N_cal for NTC, Type K, and LDR.
    Vertical dashed line at N=L=32 marks the coverage rule threshold.
    Type K stays poly>LUT at all N_cal, confirming poly-preferred regime.
    """
    sensors = [
        ("ntc",   "NTC thermistor",        "deg C"),
        ("typek", "Type K thermocouple",   "deg C"),
        ("ldr",   "LDR photoresistor",     "log10(lux)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for ax, (key, name, unit_str) in zip(axes, sensors):
        data = sweep[key]
        ns   = sorted([n for n, v in data.items()
                       if v.get("lut") is not None])
        if not ns:
            ax.set_title(f"{name}\n(no data)", fontsize=10)
            continue

        poly_v = [data[n]["poly"]    for n in ns]
        lut_v  = [data[n]["lut"]     for n in ns]
        lut_e  = [data[n].get("lut_std") or 0 for n in ns]

        ax.plot(ns, poly_v, "o-", color=C["poly"],  lw=2, ms=5,
                label="Polynomial")
        ax.errorbar(ns, lut_v, yerr=lut_e, fmt="s-",
                    color=C["lut"], lw=2, ms=5, capsize=3,
                    label="LUT  (K=1, L=32)")
        ax.axvline(32, color="gray", ls="--", lw=1, alpha=0.6,
                   label="N = L = 32")

        ax.set_xlabel("Number of calibration points", fontsize=10)
        ax.set_ylabel(f"MAE ({unit_str})", fontsize=10)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, lw=0.5)

    fig.suptitle(
        "Calibration accuracy vs. number of calibration points "
        r"($K{=}1$, $L{=}32$)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "ncal_sweep")
    plt.close()


def fig_lsweep(l_rows: dict) -> None:
    """
    Figure NEW-3: NTC MAE vs. LUT table size L.
    Left panel: MAE vs. L (log y-scale to show full range including L=8).
    Right panel: Pareto accuracy-memory frontier.
    Key result: L=8 is 7.7x worse, L=16 is 2.4x worse; L>=32 beats poly.
    """
    Ls        = sorted(l_rows.keys())
    lut_means = [l_rows[L]["lut"]     for L in Ls]
    lut_stds  = [l_rows[L]["lut_std"] for L in Ls]
    poly_ref  = l_rows[Ls[0]]["poly"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: MAE vs L (log y-scale)
    ax1.errorbar(Ls, lut_means, yerr=lut_stds, fmt="s-",
                 color=C["lut"], lw=2, ms=7, capsize=4,
                 label="LUT (K=1)", zorder=3)
    ax1.axhline(poly_ref, color=C["poly"], ls="--", lw=2,
                label=f"Poly-5  ({poly_ref:.3f} deg C)", zorder=2)
    ax1.set_yscale("log")
    ax1.set_ylim(0.05, 3.5)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.3f}"))

    # Colour-code bars: red = poly wins, green = LUT wins
    for L, lut_m in zip(Ls, lut_means):
        color = C["lut_win"] if lut_m < poly_ref else C["poly_pref"]
        marker = "+" if lut_m > poly_ref else "x"
        ax1.annotate(
            "+" if lut_m > poly_ref else "ok",
            (L, lut_m),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=14, color=color, fontweight="bold")

    ax1.set_xlabel("Table size L (cells)", fontsize=11)
    ax1.set_ylabel("MAE (deg C, log scale)", fontsize=11)
    ax1.set_title("NTC: LUT accuracy vs. table size", fontsize=11)
    ax1.set_xticks(Ls)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25, lw=0.5, which="both")
    ax1.set_axisbelow(True)

    # Right: Pareto (accuracy vs. storage bytes)
    ax2.errorbar(Ls, lut_means, yerr=lut_stds, fmt="s-",
                 color=C["lut"], lw=2, ms=7, capsize=4,
                 label="LUT uint8")
    ax2.axhline(poly_ref, color=C["poly"], ls="--", lw=2,
                label=f"Poly-5  (24 bytes, f32)")
    ax2.axvline(32, color="gray", ls=":", lw=1,
                label="L=32 (recommended)", alpha=0.7)
    ax2.set_yscale("log")
    ax2.set_ylim(0.05, 3.5)
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax2.set_xlabel("Storage (bytes, uint8, K=1)", fontsize=11)
    ax2.set_ylabel("MAE (deg C, log scale)", fontsize=11)
    ax2.set_title("Accuracy-memory Pareto frontier (NTC)", fontsize=11)
    ax2.set_xticks(Ls)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25, lw=0.5, which="both")
    ax2.set_axisbelow(True)

    fig.suptitle(
        r"LUT table size study (NTC, $K{=}1$, $N_\mathrm{cal}{=}50$)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "l_sweep_ntc")
    plt.close()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_ntc_csv(rows: list) -> None:
    path = OUT / "ntc_extended_table.csv"
    fields = ["label", "B", "factory_mae", "offset_mae", "poly_mae",
              "shh_ideal_mae", "shh_noisy_mae", "lut_mae", "lut_mae_std"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path.name}")


def save_multi_csv(results: dict) -> None:
    path = OUT / "per_sensor_table.csv"
    meta = {
        "ntc":      ("NTC thermistor",         "deg C",       "Strong (exp.)",    "synthetic"),
        "typek":    ("Type K thermocouple",     "deg C",       "Smooth (NIST EMF)","parametric"),
        "ldr":      ("LDR photoresistor",       "log10(lux)",  "Strong (log)",     "parametric"),
        "humidity": ("Humidity (resistive)",    "%RH",         "Strong (exp.)",    "parametric"),
        "mq":       ("MQ gas sensor",           "log10(ppm)",  "Moderate (power)", "parametric"),
        "ph":       ("pH electrode",            "pH units",    "Linear",           "parametric"),
    }
    rows = []
    for key, r in results.items():
        name, units, nl, src = meta.get(key, (key, "?", "?", "?"))
        sig = "p<0.01 **" if r["p"] < 0.01 else \
              ("p<0.05 *" if r["p"] < 0.05 else "ns")
        rows.append(dict(
            sensor=name, units=units, nonlinearity=nl, data_source=src,
            factory_mae=round(r["factory"] or 0, 4),
            poly_mae=round(r["poly"], 4),
            lut_mae=round(r["lut"], 4),
            lut_std=round(r["lut_std"], 4),
            ratio=round(r["ratio"], 3),
            significance=sig,
            p=round(r["p"], 4) if r["p"] == r["p"] else "nan",
        ))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="Quick run for CI/smoke-test "
                         "(2 seeds, 600 epochs, reduced N_cal_values)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    cfg = get_cfg(quick=args.quick)
    mode = "[QUICK]" if args.quick else "[FULL] "
    print(f"{mode}  epochs={cfg['epochs']}  seeds={cfg['TRAIN_SEEDS']}")
    print(f"         output -> {OUT}")

    ntc_rows  = exp_A_ntc_extended(cfg)
    multi_res = exp_B_multi_sensor(cfg)
    ncal_sw   = exp_C_ncal_sweep(cfg)
    l_rows    = exp_D_lsize_sweep(cfg)

    print("\n=== Generating figures (PNG + PDF) ===")
    import matplotlib.ticker
    fig_sensor_curves()
    fig_regime_map(multi_res)
    fig_ntc_extended(ntc_rows)
    fig_ncal_sweep(ncal_sw)
    fig_lsweep(l_rows)

    print("\n=== Saving CSV tables ===")
    save_ntc_csv(ntc_rows)
    save_multi_csv(multi_res)

    # Full JSON dump (all numeric results + config)
    all_results = dict(
        ntc_extended=ntc_rows,
        multi_sensor={
            k: {kk: vv for kk, vv in v.items()
                if not callable(vv) and kk not in ("poly_values", "lut_values")}
            for k, v in multi_res.items()
        },
        ncal_sweep={
            k: {str(n): vv for n, vv in sv.items()}
            for k, sv in ncal_sw.items()
        },
        l_sweep={str(k): v for k, v in l_rows.items()},
        config=cfg,
    )
    json_path = OUT / "summary_v14.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {json_path.name}")

    print(f"\n{'='*60}")
    print(f"All outputs written to: {OUT}")
    print(f"{'='*60}")
    for p in sorted(OUT.iterdir()):
        if not p.name.startswith("."):
            print(f"  {p.name:<42s}  {p.stat().st_size // 1024:4d} KB")
