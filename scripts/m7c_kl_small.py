"""
M7c — Extended K × L sweep: smaller K and L values.

M7a covered K ∈ {8, 16, 32}. The monotonic trend (smaller K → better ratio)
suggests K=4 and K=2 might push the advantage further.

Sweep: K ∈ {1, 2, 4, 8} × L ∈ {8, 16, 32}
Targets: sine, cusp, saturating. 3 seeds. 600 epochs.

Primary question: where is the true Pareto frontier below 288 bytes?
Secondary: does the trend hold across all targets, or is it sine-specific?
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

K_VALS = [1, 2, 4, 8]
L_VALS = [8, 16, 32]
TARGETS = ["sine", "cusp", "saturating"]
SEEDS = [0, 1, 2]
EPOCHS = 400
POLY_DEG = 16


def mem_bytes(K, L):
    return K * L + K * 4   # uint8 table + f16 scale + f16 ymin


def boot_ratio_ci(post_mses, direct_mses, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    a, b = np.array(post_mses), np.array(direct_mses)
    n_ = len(a)
    rats = [np.mean(a[rng.randint(0,n_,n_)]) / np.mean(b[rng.randint(0,n_,n_)])
            for _ in range(n)]
    pt = float(np.mean(a) / np.mean(b))
    return {"point": round(pt, 1),
            "ci_lo": round(float(np.percentile(rats, 2.5)), 1),
            "ci_hi": round(float(np.percentile(rats, 97.5)), 1)}


def run_cell(K, L, lut_init, x_tr, y_tr, x_v, y_v, x_te, y_te):
    q, s, m = quantize_lut_uint8_asym(lut_init)
    post_mse = float(np.mean((lut_forward_numpy(x_te, dequantize_lut(q,s,m)) - y_te)**2))
    directs = []
    for seed in SEEDS:
        cfg = TrainConfig(lambda_2=1.0, lr=1e-2, epochs=EPOCHS, seed=seed)
        res = train_lut_edge(lut_init, x_tr, y_tr, x_v, y_v, x_te, y_te, -1., 1., cfg)
        directs.append(res.mse_test_at_best)
    ci = boot_ratio_ci([post_mse]*len(directs), directs)
    return {
        "post_mse": post_mse,
        "direct_mean": float(np.mean(directs)),
        "direct_std":  float(np.std(directs)),
        "direct_seeds": directs,
        "ratio":    ci["point"],
        "ci_lo":    ci["ci_lo"],
        "ci_hi":    ci["ci_hi"],
        "mem_bytes": mem_bytes(K, L),
    }


def run():
    out = Path(__file__).parent.parent / "results" / "M7c_kl_small"
    out.mkdir(parents=True, exist_ok=True)
    all_res = {}

    for tname in TARGETS:
        print(f"\n{'='*60}\nTarget: {tname}\n{'='*60}")
        (out / tname).mkdir(exist_ok=True)

        x_tr,y_tr,x_v,y_v,x_te,y_te = generate_data(tname, seed=42)
        coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=POLY_DEG)
        poly_mse = float(np.mean((eval_chebyshev(x_te, coeffs) - y_te)**2))
        print(f"  poly MSE: {poly_mse:.3e}")

        grid, ratio_grid = {}, np.zeros((len(K_VALS), len(L_VALS)))

        for ki, K in enumerate(K_VALS):
            for li, L in enumerate(L_VALS):
                lut_init = sample_polynomial_to_lut(coeffs, K=K, L=L)
                cell = run_cell(K, L, lut_init, x_tr,y_tr,x_v,y_v,x_te,y_te)
                grid[f"K{K}_L{L}"] = cell
                ratio_grid[ki, li] = cell["ratio"]
                print(f"  K={K:2d} L={L:2d} {cell['mem_bytes']:4d}B  "
                      f"post={cell['post_mse']:.2e}  direct={cell['direct_mean']:.2e}  "
                      f"ratio={cell['ratio']:.0f}× [{cell['ci_lo']:.0f},{cell['ci_hi']:.0f}]")

        all_res[tname] = {"poly_mse": poly_mse, "grid": grid}

        # ── Heatmap ──────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"M7c Small-K/L sweep — {tname}", fontsize=13)

        for ax, data, title in zip(
            axes,
            [np.log10(np.maximum(ratio_grid, 0.1)), np.log10(np.array([[grid[f'K{K}_L{L}']['direct_mean'] for L in L_VALS] for K in K_VALS]))],
            ["log₁₀(ratio post/direct)", "log₁₀(direct-LUT MSE)"],
        ):
            im = ax.imshow(data, cmap="RdYlGn" if "ratio" in title else "RdYlGn_r", aspect="auto")
            ax.set_xticks(range(len(L_VALS))); ax.set_xticklabels([f"L={L}" for L in L_VALS])
            ax.set_yticks(range(len(K_VALS))); ax.set_yticklabels([f"K={K}" for K in K_VALS])
            ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.8)
            for ki in range(len(K_VALS)):
                for li in range(len(L_VALS)):
                    ax.text(li, ki, f"{data[ki,li]:.1f}", ha="center", va="center",
                            fontsize=11, color="black", fontweight="bold")
        plt.tight_layout()
        fig.savefig(out / tname / "heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── Pareto vs memory ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        colors_k = {1:"#9E9E9E", 2:"#9C27B0", 4:"#2196F3", 8:"#4CAF50"}
        markers_l = {8:"o", 16:"s", 32:"^"}
        for K in K_VALS:
            for L in L_VALS:
                cell = grid[f"K{K}_L{L}"]
                mb, r = cell["mem_bytes"], cell["ratio"]
                lo, hi = cell["ci_lo"], cell["ci_hi"]
                ax.scatter(mb, r, color=colors_k[K], marker=markers_l[L], s=140, zorder=3)
                ax.errorbar(mb, r, yerr=[[max(r-lo,0)],[max(hi-r,0)]],
                            color=colors_k[K], capsize=4, lw=1, zorder=2)
                ax.annotate(f"K{K}/L{L}", (mb, r), textcoords="offset points",
                            xytext=(8,4), fontsize=7.5, color=colors_k[K])

        # Add M7a reference points for K=8
        ax.axhline(1.0, color="gray", ls="--", lw=1, label="ratio=1 (parity)")
        ax.set_xlabel("Memory (bytes)"); ax.set_ylabel("Ratio (log scale)")
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_title(f"Pareto: ratio vs memory (log-log) — {tname}")
        # Legend
        from matplotlib.lines import Line2D
        handles = ([Line2D([0],[0],color=colors_k[k],lw=0,marker='o',markersize=8,label=f"K={k}") for k in K_VALS] +
                   [Line2D([0],[0],color='black',lw=0,marker=markers_l[l],markersize=8,label=f"L={l}") for l in L_VALS])
        ax.legend(handles=handles, fontsize=8, ncol=2)
        plt.tight_layout()
        fig.savefig(out / tname / "pareto.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        with open(out / tname / "summary.json", "w") as f:
            json.dump({"poly_mse": poly_mse, "grid": grid}, f, indent=2)

    # ── Combined Pareto: all targets on one plot ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("M7c Extended sweep — Pareto frontier across targets", fontsize=13)
    colors_k = {1:"#9E9E9E", 2:"#9C27B0", 4:"#2196F3", 8:"#4CAF50"}
    markers_l = {8:"o", 16:"s", 32:"^"}
    for ax, tname in zip(axes, TARGETS):
        grid = all_res[tname]["grid"]
        for K in K_VALS:
            for L in L_VALS:
                c = grid[f"K{K}_L{L}"]
                ax.scatter(c["mem_bytes"], c["ratio"], color=colors_k[K],
                           marker=markers_l[L], s=120, zorder=3)
                ax.annotate(f"K{K}/L{L}", (c["mem_bytes"], c["ratio"]),
                            textcoords="offset points", xytext=(5,3), fontsize=7, color=colors_k[K])
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_xlabel("Memory (bytes)"); ax.set_ylabel("Ratio")
        ax.set_title(tname)
    plt.tight_layout()
    fig.savefig(out / "pareto_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Global summary
    gsummary = {}
    for tname in TARGETS:
        g = all_res[tname]["grid"]
        best = max(g, key=lambda k: g[k]["ratio"])
        gsummary[tname] = {
            "poly_mse": all_res[tname]["poly_mse"],
            "best_config": best,
            "best_ratio": g[best]["ratio"],
            "best_mem": g[best]["mem_bytes"],
            "K4_L32_ratio": g["K4_L32"]["ratio"],
            "K4_L32_mem": g["K4_L32"]["mem_bytes"],
            "K8_L32_ratio": g["K8_L32"]["ratio"],
            "K8_L32_mem": g["K8_L32"]["mem_bytes"],
        }
    with open(out / "summary.json", "w") as f:
        json.dump(gsummary, f, indent=2)

    print(f"\n{'='*60}\nM7c COMPLETE — Pareto summary\n{'='*60}")
    print(f"\n{'Config':10s} {'Mem':6s} {'sine':>10s} {'cusp':>10s} {'saturating':>12s}")
    print("-"*52)
    for K in K_VALS:
        for L in L_VALS:
            key = f"K{K}_L{L}"
            mem = all_res["sine"]["grid"][key]["mem_bytes"]
            vals = [f"{all_res[t]['grid'][key]['ratio']:>10.0f}×" for t in TARGETS]
            print(f"K={K},L={L:2d}   {mem:4d}B  {''.join(vals)}")


if __name__ == "__main__":
    run()
