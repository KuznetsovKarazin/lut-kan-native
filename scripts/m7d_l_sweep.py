"""
M7d — L sweep: characterizing how L drives the direct-LUT advantage.

M7c probed discrete (K, L) pairs. The L-sweep probe revealed:

  K=8: ratio peaks at L≈48-64 then COLLAPSES at L=96+ (182× at L=96 vs
       4671× at L=64). Sharp crossover.
  K=4: ratio grows monotonically up to at least L=128. No crossover found.
  K=16 (M7a): peaked at L=32, collapsed at L=64.

This phase sweeps L ∈ {8, 16, 24, 32, 48, 64, 96, 128} for K ∈ {4, 8}
on all three targets (3 seeds each) to:

  1. Precisely locate the crossover for K=8.
  2. Determine whether K=4 ever crosses over within the tested range.
  3. Explain the crossover: data density per cell = n_train / (K × L).
     When K×L ≫ n_train, gradient coverage becomes sparse → collapse.
  4. Establish the rule: optimal L ≈ n_train / (K × coverage_factor).
"""

from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lut_native.baselines import (
    fit_chebyshev_ls, eval_chebyshev, sample_polynomial_to_lut,
    quantize_lut_uint8_asym, dequantize_lut,
)
from lut_native.core import lut_forward_numpy
from lut_native.targets import generate_data
from lut_native.training import train_lut_edge, TrainConfig

K_VALS  = [4, 8]
L_VALS  = [8, 16, 24, 32, 48, 64, 96, 128]
TARGETS = ["sine", "cusp", "saturating"]
SEEDS   = [0, 1]
EPOCHS  = 300
POLY_DEG = 16
N_TRAIN  = 500   # from generate_data default


def mem_bytes(K, L):
    return K * L + K * 4


def run_cell(K, L, lut_init, x_tr, y_tr, x_v, y_v, x_te, y_te):
    q, s, m = quantize_lut_uint8_asym(lut_init)
    post_mse = float(np.mean((lut_forward_numpy(x_te, dequantize_lut(q,s,m)) - y_te)**2))
    directs, best_eps = [], []
    for seed in SEEDS:
        cfg = TrainConfig(lambda_2=1.0, lr=1e-2, epochs=EPOCHS, seed=seed)
        res = train_lut_edge(lut_init, x_tr,y_tr,x_v,y_v,x_te,y_te,-1.,1.,cfg)
        directs.append(res.mse_test_at_best)
        best_eps.append(res.best_epoch)
    ratio = post_mse / float(np.mean(directs))
    density = N_TRAIN / (K * L)   # expected data points per LUT cell
    return {
        "post_mse":    post_mse,
        "direct_mean": float(np.mean(directs)),
        "direct_std":  float(np.std(directs)),
        "ratio":       round(ratio, 1),
        "best_ep_mean": round(float(np.mean(best_eps)), 0),
        "mem_bytes":   mem_bytes(K, L),
        "data_per_cell": round(density, 2),
        "total_cells": K * L,
    }


def run():
    out = Path(__file__).parent.parent / "results" / "M7d_l_sweep"
    out.mkdir(parents=True, exist_ok=True)
    all_res = {}

    for tname in TARGETS:
        print(f"\n{'='*60}\nTarget: {tname}\n{'='*60}")
        (out / tname).mkdir(exist_ok=True)

        x_tr,y_tr,x_v,y_v,x_te,y_te = generate_data(tname, seed=42)
        coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=POLY_DEG)
        poly_mse = float(np.mean((eval_chebyshev(x_te, coeffs) - y_te)**2))
        print(f"  poly MSE: {poly_mse:.3e}  (n_train={N_TRAIN})")

        tres = {"poly_mse": poly_mse, "K": {}}

        for K in K_VALS:
            print(f"\n  K={K}:")
            print(f"  L    cells  density  mem    ratio    best_ep")
            k_res = {}
            lut_init_base = None

            for L in L_VALS:
                lut_init = sample_polynomial_to_lut(coeffs, K=K, L=L)
                cell = run_cell(K, L, lut_init, x_tr,y_tr,x_v,y_v,x_te,y_te)
                k_res[L] = cell
                star = " ←PEAK" if (L > L_VALS[0] and
                                    k_res.get(L_VALS[L_VALS.index(L)-1], {}).get("ratio",0) < cell["ratio"]
                                    and (L == L_VALS[-1] or True)) else ""
                print(f"  L={L:3d} {K*L:4d}c  {cell['data_per_cell']:5.1f}/c  "
                      f"{cell['mem_bytes']:4d}B  {cell['ratio']:7.0f}×  "
                      f"ep={cell['best_ep_mean']:.0f}{star}")

            tres["K"][str(K)] = k_res

        all_res[tname] = tres

        # ── Plot ──────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"M7d L-sweep (K=4 vs K=8) — {tname}", fontsize=13)
        colors = {4: "#2196F3", 8: "#4CAF50"}

        # Left: ratio vs L
        ax = axes[0]
        for K in K_VALS:
            ls = L_VALS
            ratios = [tres["K"][str(K)][L]["ratio"] for L in ls]
            ax.plot(ls, ratios, "o-", color=colors[K], lw=2, ms=7, label=f"K={K}")
            for L, r in zip(ls, ratios):
                ax.annotate(f"{r:.0f}×", (L, r), textcoords="offset points",
                            xytext=(2, 5), fontsize=7, color=colors[K])
        ax.set_xlabel("L (cells per segment)"); ax.set_ylabel("Ratio (log)")
        ax.set_yscale("log"); ax.set_title("Ratio post/direct vs L")
        ax.legend(); ax.grid(True, alpha=0.3)

        # Middle: ratio vs memory
        ax2 = axes[1]
        for K in K_VALS:
            mems   = [tres["K"][str(K)][L]["mem_bytes"] for L in L_VALS]
            ratios = [tres["K"][str(K)][L]["ratio"]     for L in L_VALS]
            ax2.plot(mems, ratios, "o-", color=colors[K], lw=2, ms=7, label=f"K={K}")
            for m, r, L in zip(mems, ratios, L_VALS):
                ax2.annotate(f"L{L}", (m, r), textcoords="offset points",
                             xytext=(3, 4), fontsize=7, color=colors[K])
        ax2.set_xlabel("Memory (bytes)"); ax2.set_ylabel("Ratio (log)")
        ax2.set_yscale("log"); ax2.set_xscale("log")
        ax2.set_title("Ratio vs memory (log-log)"); ax2.legend(); ax2.grid(True, alpha=0.3)

        # Right: ratio vs data_per_cell (coverage density)
        ax3 = axes[2]
        for K in K_VALS:
            densities = [tres["K"][str(K)][L]["data_per_cell"] for L in L_VALS]
            ratios    = [tres["K"][str(K)][L]["ratio"]         for L in L_VALS]
            ax3.plot(densities, ratios, "o-", color=colors[K], lw=2, ms=7, label=f"K={K}")
            for d, r, L in zip(densities, ratios, L_VALS):
                ax3.annotate(f"L{L}", (d, r), textcoords="offset points",
                             xytext=(2, 4), fontsize=7, color=colors[K])
        ax3.axvline(1.0, color="red", ls="--", lw=1.5, label="1 pt/cell (coverage limit)")
        ax3.set_xlabel("Data points per LUT cell (n_train / K·L)")
        ax3.set_ylabel("Ratio (log)")
        ax3.set_yscale("log"); ax3.set_title("Ratio vs gradient coverage density")
        ax3.legend(); ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(out / tname / "plot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        with open(out / tname / "summary.json", "w") as f:
            json.dump(tres, f, indent=2)

    # ── Cross-target summary plot ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("M7d — Ratio vs L across all targets", fontsize=13)
    ls_style = {4: "-", 8: "--"}
    t_colors = {"sine": "#2196F3", "cusp": "#F44336", "saturating": "#4CAF50"}

    for ax, K in zip(axes[:2], K_VALS):
        ax.set_title(f"K={K}: ratio vs L")
        for tname in TARGETS:
            ratios = [all_res[tname]["K"][str(K)][L]["ratio"] for L in L_VALS]
            ax.plot(L_VALS, ratios, "o-", color=t_colors[tname], lw=2, ms=6, label=tname)
        ax.axvline(N_TRAIN / K, color="gray", ls=":", lw=1.5,
                   label=f"n_train/K = {N_TRAIN//K}")
        ax.set_xlabel("L"); ax.set_ylabel("Ratio (log)"); ax.set_yscale("log")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Third panel: best L per K per target
    ax3 = axes[2]
    x_pos = np.arange(len(TARGETS))
    w = 0.35
    for i, K in enumerate(K_VALS):
        best_ls = []
        for tname in TARGETS:
            k_data = all_res[tname]["K"][str(K)]
            best_L = max(L_VALS, key=lambda L: k_data[L]["ratio"])
            best_ls.append(best_L)
        ax3.bar(x_pos + i*w, best_ls, w, label=f"K={K}", color=colors[K], alpha=0.82)
    ax3.set_xticks(x_pos + w/2); ax3.set_xticklabels(TARGETS)
    ax3.set_ylabel("Best L (within sweep)"); ax3.set_title("Best L per target per K")
    ax3.legend()
    plt.tight_layout()
    fig.savefig(out / "summary_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Global JSON
    gsummary = {}
    for tname in TARGETS:
        for K in K_VALS:
            k_data = all_res[tname]["K"][str(K)]
            best_L = max(L_VALS, key=lambda L: k_data[L]["ratio"])
            key = f"{tname}_K{K}"
            gsummary[key] = {
                "best_L": best_L,
                "best_ratio": k_data[best_L]["ratio"],
                "best_mem": k_data[best_L]["mem_bytes"],
                "L32_ratio": k_data[32]["ratio"],
                "L64_ratio": k_data[64]["ratio"],
                "L96_ratio": k_data[96]["ratio"],
                "crossover_between": None,
            }
            # Detect crossover: first L where ratio drops vs previous
            prev_r = 0
            for L in L_VALS:
                r = k_data[L]["ratio"]
                if r < prev_r * 0.8:   # >20% drop = crossover
                    gsummary[key]["crossover_between"] = f"L={L_VALS[L_VALS.index(L)-1]}–L={L}"
                    break
                prev_r = r

    with open(out / "summary.json", "w") as f:
        json.dump(gsummary, f, indent=2)

    print(f"\n{'='*60}\nM7d COMPLETE\n{'='*60}")
    print(f"\n{'Target':12s} K   best_L  best_ratio  crossover")
    for tname in TARGETS:
        for K in K_VALS:
            s = gsummary[f"{tname}_K{K}"]
            print(f"{tname:12s} {K}   L={s['best_L']:3d}  {s['best_ratio']:8.0f}×  "
                  f"{s['crossover_between'] or 'none in range'}")


if __name__ == "__main__":
    run()
