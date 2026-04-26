"""
H8 experiment: N-layer LUTKANStack with adaptive normalisation vs
2-layer LUTKAN2Layer and poly-KAN baselines on the feynman_2d target.

H4 (v14) showed that poly-KAN Pareto-dominates 2-layer direct-LUT on
feynman_2d at every tested memory budget.  H8 asks: can deeper stacks
with per-layer adaptive normalisation close or reverse that gap?

Configurations tested (all at K=16, L=32, so 512 bytes per edge in uint8):

  Arch                           Edges    LUT params   Notes
  ──────────────────────────────────────────────────────────────
  LUTKANStack  [2 → 4 → 1]       10       5 120        Two-block; same as LUTKAN2Layer
  LUTKANStack  [2 → 4 → 4 → 1]   18       9 216 (+8)   Three-block (+2 norms)
  LUTKANStack  [2 → 8 → 8 → 1]   34      17 408 (+16)  Wider three-block
  LUTKANStack  [2 → 8 → 4 → 1]   28      14 336 (+12)  Funnel three-block
  LUTKAN2Layer [2 → 4 → 1]       10       5 120        H4 baseline (tanh squash)
  PolyKAN2     [2 → 4 → 1]  deg=20    11 × 84B = ~924B H4 poly baseline
  PolyKAN2     [2 → 8 → 1]  deg=20    19 × 84B = ~1596B Wider poly

5 seeds × 7 configs = 35 runs.  Paired bootstrap CI on log-MSE ratio
across seeds (same seed indexing as H4 for comparability).

Results written to results/H8_multilevel/:
  summary.json         — full numeric results
  README.md            — human-readable findings
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch

# Allow running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lut_native.kan_stack import LUTKANStack
from lut_native.kan2 import LUTKAN2Layer
from lut_native.poly_kan2 import PolyKAN2Layer, train_poly_kan2
from lut_native.training_kan2 import KAN2TrainConfig, train_kan2
from lut_native.training_stack import StackTrainConfig, train_lut_stack
from lut_native.targets import generate_data_2d
from lut_native.coverage_stack import compute_stack_coverage, stack_coverage_to_dict
from lut_native.metrics import paired_bootstrap_ci


# ─────────────────────────────────────────────────────────────────────────────
# Experiment constants
# ─────────────────────────────────────────────────────────────────────────────

K, L = 16, 32
N_TRAIN, N_VAL, N_TEST = 800, 300, 400
SEEDS = [0, 1, 2, 3, 4]
EPOCHS_LUT   = 2500
EPOCHS_POLY  = 3000
LR           = 1e-2
BATCH        = 64
LAMBDA_2     = 1e-5      # mild curvature prior, same as H4

OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "H8_multilevel"
)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture definitions
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (tag, kind, kwargs)
#   kind = "stack" | "kan2" | "poly"
ARCHITECTURES = [
    # Stack variants
    ("stack_2x4",    "stack", dict(dims=[2, 4, 1],    K=K, L=L)),
    ("stack_3x4",    "stack", dict(dims=[2, 4, 4, 1], K=K, L=L)),
    ("stack_3x8",    "stack", dict(dims=[2, 8, 8, 1], K=K, L=L)),
    ("stack_funnel", "stack", dict(dims=[2, 8, 4, 1], K=K, L=L)),
    # 2-layer LUT-KAN baseline (H4 regime, tanh squash)
    ("kan2_4",       "kan2",  dict(in_dim=2, hidden_dim=4, out_dim=1, K=K, L=L)),
    # Polynomial baselines
    ("poly_4_d20",   "poly",  dict(in_dim=2, hidden_dim=4,  out_dim=1, degree=20)),
    ("poly_8_d20",   "poly",  dict(in_dim=2, hidden_dim=8,  out_dim=1, degree=20)),
]


# ─────────────────────────────────────────────────────────────────────────────
# Memory budget helpers
# ─────────────────────────────────────────────────────────────────────────────

def stack_budget(dims, K, L):
    """Total uint8 bytes for a LUTKANStack (all LUT blocks)."""
    total = 0
    for i in range(len(dims) - 1):
        n_edges = dims[i] * dims[i + 1]
        total += n_edges * (K * L + 4 * K)   # q-table + scale + y_min at f16
    return total

def kan2_budget(in_dim, hidden_dim, out_dim, K, L):
    n_edges = in_dim * hidden_dim + hidden_dim * out_dim
    return n_edges * (K * L + 4 * K)

def poly_budget_bytes(in_dim, hidden_dim, out_dim, degree):
    n_edges = in_dim * hidden_dim + hidden_dim * out_dim
    return n_edges * (degree + 1) * 4   # float32


# ─────────────────────────────────────────────────────────────────────────────
# Single-seed run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(tag: str, kind: str, kwargs: dict, seed: int, data: tuple) -> dict:
    x_tr, y_tr, x_val, y_val, x_te, y_te = data

    t0 = time.time()

    if kind == "stack":
        model = LUTKANStack(**kwargs)
        cfg = StackTrainConfig(
            lambda_2=LAMBDA_2, lr=LR, epochs=EPOCHS_LUT,
            batch_size=BATCH, init_noise_std=0.05,
            seed=seed, eval_every_epochs=25,
            calibrate_at_start=True, norm_lr_scale=2.0,
            track_coverage=True,
        )
        res = train_lut_stack(model, x_tr, y_tr, x_val, y_val, x_te, y_te, cfg)

        # Coverage at end of training
        cov_report = compute_stack_coverage(model, x_tr)
        cov = stack_coverage_to_dict(cov_report)
        norm_uniformities = [
            n["uniformity_mean"] for n in cov.get("norms", [])
        ]

        budget = stack_budget(kwargs["dims"], kwargs["K"], kwargs["L"])
        return {
            "tag": tag, "kind": kind, "seed": seed,
            "mse_val": res.mse_val_at_best,
            "mse_test": res.mse_test_at_best,
            "best_epoch": res.best_epoch,
            "budget_bytes": budget,
            "n_lut_params": model.n_lut_params(),
            "n_norm_params": model.n_norm_params(),
            "norm_uniformities": norm_uniformities,
            "elapsed_s": round(time.time() - t0, 1),
        }

    elif kind == "kan2":
        model = LUTKAN2Layer(**kwargs)
        cfg = KAN2TrainConfig(
            lambda_2=LAMBDA_2, lr=LR, epochs=EPOCHS_LUT,
            batch_size=BATCH, init_noise_std_absolute=0.05,
            seed=seed, eval_every_epochs=25,
        )
        res = train_kan2(model, x_tr, y_tr, x_val, y_val, x_te, y_te, cfg)
        budget = kan2_budget(
            kwargs["in_dim"], kwargs["hidden_dim"], kwargs["out_dim"],
            kwargs["K"], kwargs["L"]
        )
        return {
            "tag": tag, "kind": kind, "seed": seed,
            "mse_val": res.mse_val_at_best,
            "mse_test": res.mse_test_at_best,
            "best_epoch": res.best_epoch,
            "budget_bytes": budget,
            "n_lut_params": model.lut_l1.numel() + model.lut_l2.numel(),
            "elapsed_s": round(time.time() - t0, 1),
        }

    elif kind == "poly":
        model = PolyKAN2Layer(**kwargs)
        res = train_poly_kan2(
            model, x_tr, y_tr, x_val, y_val, x_te, y_te,
            lr=LR, epochs=EPOCHS_POLY, batch_size=BATCH,
            seed=seed, eval_every_epochs=25,
        )
        budget = poly_budget_bytes(
            kwargs["in_dim"], kwargs["hidden_dim"], kwargs["out_dim"], kwargs["degree"]
        )
        return {
            "tag": tag, "kind": kind, "seed": seed,
            "mse_val": res["mse_val_best"],
            "mse_test": res["mse_test_best"],
            "best_epoch": res["best_epoch"],
            "budget_bytes": budget,
            "n_params": model.total_coeffs(),
            "elapsed_s": round(time.time() - t0, 1),
        }
    else:
        raise ValueError(kind)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical summary
# ─────────────────────────────────────────────────────────────────────────────

def summarise(results: List[dict]) -> dict:
    """Group by tag, compute mean/std MSE, paired CI vs kan2_4 baseline."""
    from collections import defaultdict
    by_tag: dict = defaultdict(list)
    for r in results:
        by_tag[r["tag"]].append(r)

    baseline_tag = "kan2_4"
    baseline_mse = np.array([r["mse_test"] for r in sorted(
        by_tag[baseline_tag], key=lambda r: r["seed"]
    )])

    summary = {}
    for tag, runs in by_tag.items():
        runs_sorted = sorted(runs, key=lambda r: r["seed"])
        mse_arr = np.array([r["mse_test"] for r in runs_sorted])
        entry = {
            "mse_test_mean": float(np.mean(mse_arr)),
            "mse_test_std":  float(np.std(mse_arr)),
            "mse_test_per_seed": mse_arr.tolist(),
            "budget_bytes": runs_sorted[0]["budget_bytes"],
            "best_epoch_mean": float(np.mean([r["best_epoch"] for r in runs_sorted])),
        }
        # Paired CI vs baseline (skip self-comparison)
        if tag != baseline_tag and len(mse_arr) == len(baseline_mse):
            ci = paired_bootstrap_ci(mse_arr, baseline_mse, n_boot=10_000)
            # ratio = mse_this / mse_baseline → > 1 means this is WORSE
            entry["vs_kan2_4"] = {
                "ratio_mean":    round(ci["ratio_mean"], 3),
                "ci_95_lower":   round(ci["ratio_ci_lower"], 3),
                "ci_95_upper":   round(ci["ratio_ci_upper"], 3),
                "interpretation": (
                    "better" if ci["ratio_ci_upper"] < 1.0 else
                    "worse"  if ci["ratio_ci_lower"] > 1.0 else
                    "parity"
                ),
            }
        summary[tag] = entry
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Results README
# ─────────────────────────────────────────────────────────────────────────────

def write_readme(summary: dict, out_dir: str):
    lines = [
        "# H8 — Multilevel LUT-KAN experiment results",
        "",
        "Target: `feynman_2d` — f(x,y) = sin(πx) + 0.5 cos(2π·xy)",
        f"K={K}, L={L}, {len(SEEDS)} seeds, paired bootstrap CI vs `kan2_4`.",
        "",
        "## Test MSE summary",
        "",
        "| Config | Budget (B) | MSE mean | MSE std | vs kan2_4 ratio | CI 95% | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]

    # Sort by mean MSE
    order = sorted(summary.items(), key=lambda kv: kv[1]["mse_test_mean"])
    for tag, d in order:
        budget = d["budget_bytes"]
        mean   = d["mse_test_mean"]
        std    = d["mse_test_std"]
        vs = d.get("vs_kan2_4", {})
        ratio  = f"{vs.get('ratio_mean', '—'):.3f}" if vs else "—"
        ci     = (f"[{vs['ci_95_lower']:.3f}, {vs['ci_95_upper']:.3f}]"
                  if vs else "—")
        verdict = vs.get("interpretation", "—") if vs else "baseline"
        lines.append(
            f"| `{tag}` | {budget} | {mean:.2e} | {std:.2e} | {ratio} | {ci} | {verdict} |"
        )

    lines += [
        "",
        "## Norm uniformity (stack configs only)",
        "",
        "Uniformity close to 1.0 = all K segments receive equal traffic = good coverage.",
        "",
    ]

    # Pull norm uniformity stats
    for tag, d in summary.items():
        if "norm_uniformities_mean" in d:
            u = d["norm_uniformities_mean"]
            lines.append(f"- `{tag}`: {[round(x,3) for x in u]}")

    lines += [
        "",
        "## Interpretation",
        "",
        "ratio < 1 → stack is **better** than 2-layer LUT-KAN (lower MSE)",
        "ratio > 1 → stack is **worse**",
        "CI includes 1 → parity (no significant difference)",
        "",
        "See `summary.json` for full numeric data.",
    ]

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Generate data once (same split for all runs)
    print("Generating feynman_2d data …")
    data = generate_data_2d(
        "feynman_2d",
        n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
        seed=42,
    )
    x_tr, y_tr, x_val, y_val, x_te, y_te = data
    y_tr  = y_tr.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_te  = y_te.reshape(-1, 1)
    data_2d = (x_tr, y_tr, x_val, y_val, x_te, y_te)

    total_runs = len(SEEDS) * len(ARCHITECTURES)
    print(f"Running {total_runs} experiments ({len(SEEDS)} seeds × {len(ARCHITECTURES)} configs)\n")

    all_results: List[dict] = []
    run_idx = 0

    for seed in SEEDS:
        for tag, kind, kwargs in ARCHITECTURES:
            run_idx += 1
            print(f"[{run_idx:2d}/{total_runs}] seed={seed}  {tag} ...", end=" ", flush=True)
            try:
                r = run_one(tag, kind, kwargs, seed, data_2d)
                all_results.append(r)
                print(f"test_mse={r['mse_test']:.4e}  ({r['elapsed_s']}s)")
            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({
                    "tag": tag, "kind": kind, "seed": seed,
                    "error": str(e), "mse_test": float("nan"),
                    "mse_val": float("nan"), "best_epoch": -1,
                    "budget_bytes": 0,
                })

    print("\nComputing summary statistics …")
    summary = summarise(all_results)

    # Per-tag: also add mean norm uniformity if available
    from collections import defaultdict
    uni_by_tag: dict = defaultdict(list)
    for r in all_results:
        if "norm_uniformities" in r and r["norm_uniformities"]:
            uni_by_tag[r["tag"]].append(r["norm_uniformities"])
    for tag, lists in uni_by_tag.items():
        arr = np.array(lists)   # (n_seeds, n_norms)
        summary[tag]["norm_uniformities_mean"] = arr.mean(axis=0).tolist()

    # Write outputs
    out_json = os.path.join(OUT_DIR, "summary.json")
    with open(out_json, "w") as f:
        json.dump(
            {"config": {"K": K, "L": L, "seeds": SEEDS,
                        "epochs_lut": EPOCHS_LUT, "epochs_poly": EPOCHS_POLY,
                        "lambda_2": LAMBDA_2, "n_train": N_TRAIN},
             "raw_results": all_results,
             "summary": summary},
            f, indent=2,
        )
    write_readme(summary, OUT_DIR)

    print(f"\nResults → {OUT_DIR}/")
    print(f"          summary.json")
    print(f"          README.md\n")

    # Print quick table
    print("Quick results (test MSE, mean over seeds):")
    print(f"{'Config':<20} {'Budget':>8}  {'MSE mean':>10}  {'vs kan2_4':>12}")
    print("-" * 58)
    for tag, d in sorted(summary.items(), key=lambda kv: kv[1]["mse_test_mean"]):
        vs_str = ""
        if "vs_kan2_4" in d:
            v = d["vs_kan2_4"]
            vs_str = f"{v['ratio_mean']:.3f} ({v['interpretation']})"
        print(f"{tag:<20} {d['budget_bytes']:>8}  {d['mse_test_mean']:>10.4e}  {vs_str:>12}")


if __name__ == "__main__":
    main()
