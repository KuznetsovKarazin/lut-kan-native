"""
M7b — Random-init vs poly-init for single-edge direct-LUT.

Research question:
  Does direct-LUT training require polynomial pre-training (server step),
  or can it train from scratch on the device?

Three initialization strategies:
  poly   — sample polynomial coefficients into LUT (current H1 method)
  random — Gaussian noise scaled to match poly output range
  zero   — all LUT values = 0

K=8, L=32 (recommended config from M7a). Three targets. 5 seeds. 900 epochs.

This directly answers the reviewer question: "Your method requires a server
to compute polynomial coefficients first. What if you can't do that?"
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

K, L = 8, 32
TARGETS = ["sine", "cusp", "saturating"]
SEEDS = [0, 1, 2]
EPOCHS = 600
POLY_DEG = 16

INITS = {
    "poly":   "Sample Chebyshev polynomial into LUT (current method)",
    "random": "Gaussian noise, σ = poly_range / 2 (no server needed)",
    "zero":   "All LUT values = 0 (minimal prior)",
}


def make_lut_init(init_type: str, poly_lut: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed + 9999)
    if init_type == "poly":
        return poly_lut.copy()
    if init_type == "random":
        lut_range = max(float(poly_lut.max() - poly_lut.min()), 1e-6)
        return rng.randn(K, L).astype(np.float32) * (lut_range / 2)
    if init_type == "zero":
        return np.zeros((K, L), dtype=np.float32)
    raise ValueError(init_type)


def boot_ratio(a, b, n=1000, seed=0):
    rng = np.random.RandomState(seed); a, b = np.array(a), np.array(b)
    n_ = len(a)
    rats = [np.mean(a[rng.randint(0,n_,n_)]) / np.mean(b[rng.randint(0,n_,n_)])
            for _ in range(n)]
    return {"point": round(float(np.mean(a)/np.mean(b)), 3),
            "ci_lo": round(float(np.percentile(rats, 2.5)), 3),
            "ci_hi": round(float(np.percentile(rats, 97.5)), 3)}


def run():
    out = Path(__file__).parent.parent / "results" / "M7b_init"
    out.mkdir(parents=True, exist_ok=True)
    all_res = {}

    for tname in TARGETS:
        print(f"\n{'='*60}\nTarget: {tname}\n{'='*60}")
        (out / tname).mkdir(exist_ok=True)

        x_tr,y_tr,x_v,y_v,x_te,y_te = generate_data(tname, seed=42)
        coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=POLY_DEG)
        poly_mse = float(np.mean((eval_chebyshev(x_te, coeffs) - y_te)**2))
        poly_lut = sample_polynomial_to_lut(coeffs, K=K, L=L)

        # Post-LUT MSE (fixed reference — only meaningful for poly init)
        q,s,m = quantize_lut_uint8_asym(poly_lut)
        post_mse = float(np.mean((lut_forward_numpy(x_te, dequantize_lut(q,s,m)) - y_te)**2))
        print(f"  poly_mse={poly_mse:.3e}  post_lut_mse={post_mse:.3e}")

        tres = {"poly_mse": poly_mse, "post_mse": post_mse, "inits": {}}
        mse_by_init = {}
        trace_by_init = {}

        for iname in INITS:
            mses, traces = [], []
            for seed in SEEDS:
                lut_init = make_lut_init(iname, poly_lut, seed)
                cfg = TrainConfig(lambda_2=1.0, lr=1e-2, epochs=EPOCHS, seed=seed)
                res = train_lut_edge(lut_init, x_tr,y_tr,x_v,y_v,x_te,y_te,-1.,1.,cfg)
                mses.append(res.mse_test_at_best)
                traces.append({"ep": res.trace["epoch"], "val": res.trace["mse_val"]})

            mean_m = float(np.mean(mses)); std_m = float(np.std(mses))
            ratio_poly = poly_mse / mean_m
            ratio_post = post_mse / mean_m
            mse_by_init[iname] = mses
            trace_by_init[iname] = traces
            tres["inits"][iname] = {
                "seed_mses": mses, "mean_mse": mean_m, "std_mse": std_m,
                "ratio_vs_poly": round(ratio_poly, 1),
                "ratio_vs_post": round(ratio_post, 1),
                "traces": traces,
            }
            print(f"  {iname:8s}  mean={mean_m:.3e} ± {std_m:.0e}  "
                  f"{ratio_poly:.0f}× vs poly  {ratio_post:.0f}× vs post-LUT")

        # CIs: random vs poly, zero vs poly
        tres["ci_rand_vs_poly"] = boot_ratio(mse_by_init["random"], mse_by_init["poly"])
        tres["ci_zero_vs_poly"] = boot_ratio(mse_by_init["zero"],   mse_by_init["poly"])
        print(f"  random/poly ratio: {tres['ci_rand_vs_poly']}")
        print(f"  zero/poly ratio:   {tres['ci_zero_vs_poly']}")
        all_res[tname] = tres

        # ── Plot ────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"M7b Init strategy — K={K}, L={L} — {tname}", fontsize=13)
        colors = {"poly": "#2196F3", "random": "#FF9800", "zero": "#9E9E9E"}

        ax = axes[0]
        inames = list(INITS.keys())
        means = [tres["inits"][i]["mean_mse"] for i in inames]
        stds  = [tres["inits"][i]["std_mse"]  for i in inames]
        bars = ax.bar(inames, means, yerr=stds, capsize=6,
                      color=[colors[i] for i in inames], alpha=0.85, width=0.5)
        ax.axhline(poly_mse, color="green",  ls="--", lw=1.5, label=f"Poly MSE {poly_mse:.1e}")
        ax.axhline(post_mse, color="red",    ls=":",  lw=1.5, label=f"Post-LUT {post_mse:.1e}")
        ax.set_yscale("log"); ax.set_ylabel("Test MSE")
        ax.set_title("Test MSE by init strategy (lower = better)")
        ax.legend(fontsize=9)
        for bar, iname in zip(bars, inames):
            r = tres["inits"][iname]["ratio_vs_poly"]
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.4,
                    f"{r:.0f}×\nvs poly", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax2 = axes[1]
        for iname in inames:
            traces = trace_by_init[iname]
            eps = traces[0]["ep"]
            vm = np.array([t["val"] for t in traces])
            med = np.median(vm, axis=0)
            lo  = np.percentile(vm, 25, axis=0)
            hi  = np.percentile(vm, 75, axis=0)
            ax2.plot(eps, med, color=colors[iname], lw=1.8, label=iname)
            ax2.fill_between(eps, lo, hi, color=colors[iname], alpha=0.15)
        ax2.axhline(poly_mse, color="green", ls="--", lw=1, label="Poly MSE")
        ax2.axhline(post_mse, color="red",   ls=":",  lw=1, label="Post-LUT")
        ax2.set_xlabel("Epoch"); ax2.set_yscale("log"); ax2.set_ylabel("Val MSE")
        ax2.set_title("Convergence (median ± IQR, 5 seeds)"); ax2.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(out / tname / "plot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        clean = {k: {kk:vv for kk,vv in v.items() if kk != "traces"}
                 for k,v in tres["inits"].items()}
        with open(out/tname/"summary.json","w") as f:
            json.dump({**tres,"inits":clean}, f, indent=2)

    # Global
    gsummary = {}
    for tname in TARGETS:
        tr = all_res[tname]
        gsummary[tname] = {
            "poly_mse": tr["poly_mse"],
            "post_mse": tr["post_mse"],
            **{f"{i}_ratio_vs_poly": tr["inits"][i]["ratio_vs_poly"] for i in INITS},
            **{f"{i}_ratio_vs_post": tr["inits"][i]["ratio_vs_post"] for i in INITS},
            "ci_rand_vs_poly": tr["ci_rand_vs_poly"],
            "ci_zero_vs_poly": tr["ci_zero_vs_poly"],
        }
    with open(out/"summary.json","w") as f:
        json.dump(gsummary, f, indent=2)

    print(f"\n{'='*60}\nM7b COMPLETE\n{'='*60}")
    for tname, s in gsummary.items():
        print(f"\n  {tname}:")
        for iname in INITS:
            print(f"    {iname:8s}  {s[f'{iname}_ratio_vs_poly']:5.0f}× vs poly  "
                  f"{s[f'{iname}_ratio_vs_post']:5.0f}× vs post-LUT")
        print(f"    random/poly CI: {s['ci_rand_vs_poly']}")
        print(f"    zero/poly   CI: {s['ci_zero_vs_poly']}")


if __name__ == "__main__":
    run()
