"""
M7a — K × L sweep for single-edge direct-LUT.

H1 only tested K=16, L=32. A reviewer will ask: does the advantage hold
across different segment/resolution combinations?

We sweep K ∈ {8, 16, 32} × L ∈ {16, 32, 64} on all three H1 targets.
This gives a 3×3 grid per target, covering memory budgets from 160 B to
2176 B (uint8 + per-segment scale).

For each cell we report:
  - poly_mse (degree-16 Chebyshev, same model for all K)
  - post_lut_mse (quantize poly → LUT, no retraining)
  - direct_lut_mse (mean across 3 seeds, Adam, λ₂=1.0, 600 epochs)
  - ratio = post_lut_mse / direct_lut_mse (> 1 means direct-LUT wins)
  - 95% paired bootstrap CI on the ratio

Analysis:
  - Pareto frontier: best ratio at each memory budget
  - Does the H1 finding (K=16, L=32) lie on the Pareto front?
  - How does ratio scale with K and L separately?

Output:
  results/M7a_kl_sweep/{target}/heatmap.png
  results/M7a_kl_sweep/{target}/pareto.png
  results/M7a_kl_sweep/{target}/summary.json
  results/M7a_kl_sweep/summary.json
  docs/M7a_FINDINGS.md  (written after analysis)
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

K_VALS = [8, 16, 32]
L_VALS = [16, 32, 64]
TARGETS = ["sine", "cusp", "saturating"]
SEEDS = [0, 1, 2]
EPOCHS = 600
POLY_DEG = 16


def memory_bytes(K: int, L: int) -> int:
    """uint8 LUT + float16 scale + float16 y_min per segment."""
    return K * L + K * 4  # K*L bytes table + K * 4 bytes (scale f16 + ymin f16)


def boot_ratio_ci(post_mses, direct_mses, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    a, b = np.array(post_mses), np.array(direct_mses)
    n_ = len(a)
    rats = [np.mean(a[rng.randint(0,n_,n_)]) / np.mean(b[rng.randint(0,n_,n_)])
            for _ in range(n)]
    pt = float(np.mean(a) / np.mean(b))
    lo, hi = np.percentile(rats, [2.5, 97.5])
    return {"point": round(pt,1), "ci_lo": round(lo,1), "ci_hi": round(hi,1)}


def run_cell(K, L, lut_init, x_tr, y_tr, x_v, y_v, x_te, y_te):
    """Run direct-LUT training for one (K, L) cell, all seeds."""
    # Post-training LUT MSE (fixed, same across seeds)
    q, s, m = quantize_lut_uint8_asym(lut_init)
    post_mse = float(np.mean((lut_forward_numpy(x_te, dequantize_lut(q,s,m)) - y_te)**2))

    direct_mses = []
    for seed in SEEDS:
        cfg = TrainConfig(lambda_2=1.0, lr=1e-2, epochs=EPOCHS, seed=seed)
        res = train_lut_edge(lut_init, x_tr, y_tr, x_v, y_v, x_te, y_te,
                             -1.0, 1.0, cfg)
        direct_mses.append(res.mse_test_at_best)

    ci = boot_ratio_ci([post_mse]*len(direct_mses), direct_mses)
    return {
        "post_mse": post_mse,
        "direct_mse_mean": float(np.mean(direct_mses)),
        "direct_mse_std":  float(np.std(direct_mses)),
        "direct_mse_seeds": direct_mses,
        "ratio_mean": ci["point"],
        "ratio_ci_lo": ci["ci_lo"],
        "ratio_ci_hi": ci["ci_hi"],
        "memory_bytes": memory_bytes(K, L),
    }


def run():
    out = Path(__file__).parent.parent / "results" / "M7a_kl_sweep"
    out.mkdir(parents=True, exist_ok=True)
    all_res = {}

    for tname in TARGETS:
        print(f"\n{'='*60}\nTarget: {tname}\n{'='*60}")
        (out / tname).mkdir(exist_ok=True)

        x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data(tname, seed=42)
        coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=POLY_DEG)
        poly_mse = float(np.mean((eval_chebyshev(x_te, coeffs) - y_te)**2))
        print(f"  poly MSE: {poly_mse:.3e}")

        grid = {}
        ratio_grid = np.zeros((len(K_VALS), len(L_VALS)))
        direct_grid = np.zeros_like(ratio_grid)
        post_grid   = np.zeros_like(ratio_grid)

        for ki, K in enumerate(K_VALS):
            for li, L in enumerate(L_VALS):
                lut_init = sample_polynomial_to_lut(coeffs, K=K, L=L)
                cell = run_cell(K, L, lut_init, x_tr, y_tr, x_v, y_v, x_te, y_te)
                grid[f"K{K}_L{L}"] = cell
                ratio_grid[ki, li] = cell["ratio_mean"]
                direct_grid[ki, li] = cell["direct_mse_mean"]
                post_grid[ki, li]   = cell["post_mse"]
                mb = cell["memory_bytes"]
                print(f"  K={K:2d} L={L:2d} {mb:5d}B  "
                      f"post={cell['post_mse']:.2e}  "
                      f"direct={cell['direct_mse_mean']:.2e}  "
                      f"ratio={cell['ratio_mean']:.0f}× "
                      f"[{cell['ratio_ci_lo']:.0f},{cell['ratio_ci_hi']:.0f}]")

        all_res[tname] = {"poly_mse": poly_mse, "grid": grid}

        # ── Heatmap: ratio ────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"M7a K×L sweep — {tname}", fontsize=13)

        for ax, data, title, fmt in zip(
            axes,
            [np.log10(ratio_grid), np.log10(direct_grid), np.log10(post_grid)],
            ["log₁₀(ratio post/direct)", "log₁₀(direct-LUT MSE)", "log₁₀(post-LUT MSE)"],
            [".1f", ".1f", ".1f"],
        ):
            im = ax.imshow(data, cmap="RdYlGn" if "ratio" in title else "RdYlGn_r",
                           aspect="auto")
            ax.set_xticks(range(len(L_VALS))); ax.set_xticklabels([f"L={L}" for L in L_VALS])
            ax.set_yticks(range(len(K_VALS))); ax.set_yticklabels([f"K={K}" for K in K_VALS])
            ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.8)
            for ki in range(len(K_VALS)):
                for li in range(len(L_VALS)):
                    ax.text(li, ki, f"{data[ki,li]:.1f}", ha="center", va="center",
                            fontsize=10, color="black")

        plt.tight_layout()
        fig.savefig(out / tname / "heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── Pareto: ratio vs memory ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        colors_k = {8: "#2196F3", 16: "#4CAF50", 32: "#F44336"}
        markers_l = {16: "o", 32: "s", 64: "^"}
        for ki, K in enumerate(K_VALS):
            for li, L in enumerate(L_VALS):
                cell = grid[f"K{K}_L{L}"]
                mb = cell["memory_bytes"]
                r  = cell["ratio_mean"]
                lo = cell["ratio_ci_lo"]
                hi = cell["ratio_ci_hi"]
                ax.scatter(mb, r, color=colors_k[K], marker=markers_l[L],
                           s=120, zorder=3, label=f"K={K},L={L}")
                ax.errorbar(mb, r, yerr=[[r-lo],[hi-r]],
                            color=colors_k[K], capsize=4, lw=1, zorder=2)
                ax.text(mb+10, r*1.05, f"K{K}/L{L}", fontsize=7, color=colors_k[K])

        ax.axhline(1.0, color="gray", ls="--", lw=1, label="ratio=1 (parity)")
        ax.set_xlabel("Memory (bytes, uint8 + scale)")
        ax.set_ylabel("Ratio post/direct (higher = direct-LUT better)")
        ax.set_yscale("log")
        ax.set_title(f"Pareto: accuracy gain vs memory — {tname}")
        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen: seen[l] = h
        ax.legend(seen.values(), seen.keys(), fontsize=8, ncol=2)
        plt.tight_layout()
        fig.savefig(out / tname / "pareto.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        with open(out / tname / "summary.json", "w") as f:
            json.dump({"poly_mse": poly_mse, "grid": grid}, f, indent=2)

    # Global summary
    gsummary = {}
    for tname in TARGETS:
        tr = all_res[tname]
        best_key = max(tr["grid"], key=lambda k: tr["grid"][k]["ratio_mean"])
        best = tr["grid"][best_key]
        gsummary[tname] = {
            "poly_mse": tr["poly_mse"],
            "best_config": best_key,
            "best_ratio": best["ratio_mean"],
            "best_ratio_ci": [best["ratio_ci_lo"], best["ratio_ci_hi"]],
            "best_memory_bytes": best["memory_bytes"],
            "h1_config_ratio": tr["grid"]["K16_L32"]["ratio_mean"],
            "h1_config_ci":    [tr["grid"]["K16_L32"]["ratio_ci_lo"],
                                 tr["grid"]["K16_L32"]["ratio_ci_hi"]],
        }
    with open(out / "summary.json", "w") as f:
        json.dump(gsummary, f, indent=2)

    print(f"\n{'='*60}\nM7a COMPLETE\n{'='*60}")
    for tname, s in gsummary.items():
        print(f"\n  {tname}:")
        print(f"    H1 config (K16,L32): {s['h1_config_ratio']}× {s['h1_config_ci']}")
        print(f"    Best config: {s['best_config']}  {s['best_ratio']}× {s['best_ratio_ci']}")


if __name__ == "__main__":
    run()
