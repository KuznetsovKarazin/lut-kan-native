"""
M6a — On-device training simulation for direct-LUT.

Research question:
  Does direct-LUT training remain viable when we emulate MCU constraints?
  Specifically: float16 gradient accumulation, SGD without momentum,
  and does regularization (lambda_2) transfer to the MCU regime?

Four regimes, three H1 targets (sine, cusp, saturating), K=16, L=32:

  A: float32 + Adam  lr=1e-2 + lambda_2=1.0   (H1 server best)
  B: float32 + SGD   lr=0.5  + lambda_2=1.0   (MCU optimizer, tuned lr)
  C: float16-grad + SGD lr=0.5 + lambda_2=1.0  (MCU sim — KEY test)
  D: float16-grad + SGD lr=0.5, no lambda      (MCU sim, no regularization)

Key questions:
  B vs C: does fp16 grad cast hurt? 
  A vs C: how much does SGD lose vs Adam?
  C vs D: how important is lambda_2 regularization on-device?

5 seeds. Paired bootstrap CI (1000 resamples) for A/C and B/C.

RAM budget accounting for each regime included.
"""

from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lut_native.core import LUTEdge
from lut_native.regularizers import combined_penalty
from lut_native.baselines import (
    fit_chebyshev_ls, eval_chebyshev, sample_polynomial_to_lut,
)
from lut_native.targets import generate_data

# ── Config ─────────────────────────────────────────────────────────────────
K, L = 16, 32
EPOCHS = 900
SEEDS = [0, 1, 2]
TARGET_NAMES = ["sine", "cusp", "saturating"]
EVAL_EVERY = 30
BATCH_SIZE = 64
POLY_DEGREE = 16

REGIMES = {
    "A_adam_f32": dict(
        optimizer="adam", lr=1e-2,  fp16_grad=False, lam2=1.0,
        label="float32+Adam lr=1e-2 λ₂=1 (server best)",
    ),
    "B_sgd_f32": dict(
        optimizer="sgd",  lr=0.5,   fp16_grad=False, lam2=1.0,
        label="float32+SGD  lr=0.5  λ₂=1 (MCU opt)",
    ),
    "C_sgd_fp16": dict(
        optimizer="sgd",  lr=0.5,   fp16_grad=True,  lam2=1.0,
        label="fp16-grad+SGD lr=0.5  λ₂=1 (MCU sim KEY)",
    ),
    "D_sgd_fp16_noreg": dict(
        optimizer="sgd",  lr=0.5,   fp16_grad=True,  lam2=0.0,
        label="fp16-grad+SGD lr=0.5  λ₂=0 (MCU no-reg)",
    ),
}

# ── RAM accounting ──────────────────────────────────────────────────────────
def ram_bytes(K, L, optimizer, fp16_grad, batch_size=64):
    return {
        "lut_param":  K*L*4,
        "grad_buf":   K*L*(2 if fp16_grad else 4),
        "adam_state": K*L*4*2 if optimizer=="adam" else 0,
        "batch_buf":  batch_size*4*3,
        "total":      K*L*4 + K*L*(2 if fp16_grad else 4)
                      + (K*L*4*2 if optimizer=="adam" else 0)
                      + batch_size*4*3,
    }

# ── Training ────────────────────────────────────────────────────────────────
def train_one(lut_init, x_tr, y_tr, x_v, y_v, x_te, y_te,
              optimizer, lr, fp16_grad, lam2, seed):
    torch.manual_seed(seed)
    np_rng = np.random.RandomState(seed)
    lut_r = float(lut_init.max() - lut_init.min())
    lut_noisy = lut_init.copy() + np_rng.randn(*lut_init.shape).astype(np.float32) * 0.01 * max(lut_r, 1e-6)

    edge = LUTEdge(K, L); edge.init_from_array(lut_noisy)
    opt = (torch.optim.Adam(edge.parameters(), lr=lr)
           if optimizer == "adam"
           else torch.optim.SGD(edge.parameters(), lr=lr, momentum=0.0))

    xt = torch.from_numpy(x_tr); yt = torch.from_numpy(y_tr)
    xv = torch.from_numpy(x_v);  yv = torch.from_numpy(y_v)
    xte= torch.from_numpy(x_te); yte= torch.from_numpy(y_te)

    best_val, best_ep, best_lut = float("inf"), 0, lut_noisy.copy()
    t_epochs, t_vals = [], []

    for ep in range(EPOCHS):
        edge.train()
        idx = torch.randperm(len(xt))
        for i in range(0, len(xt), BATCH_SIZE):
            bi = idx[i:i+BATCH_SIZE]; opt.zero_grad()
            pred = edge(xt[bi])
            loss = torch.mean((pred - yt[bi])**2)
            if lam2 > 0:
                loss = loss + lam2 * combined_penalty(edge.lut, 0, lam2, 0, 0)
            loss.backward()
            if fp16_grad:
                for p in edge.parameters():
                    if p.grad is not None:
                        p.grad.data = p.grad.data.half().float()
            opt.step()

        if (ep+1) % EVAL_EVERY == 0:
            edge.eval()
            with torch.no_grad():
                vm = float(torch.mean((edge(xv)-yv)**2))
            t_epochs.append(ep+1); t_vals.append(vm)
            if vm < best_val:
                best_val = vm; best_ep = ep+1
                best_lut = edge.lut.detach().numpy().copy()

    be = LUTEdge(K, L); be.init_from_array(best_lut); be.eval()
    with torch.no_grad():
        tm = float(torch.mean((be(xte)-yte)**2))
    return {"test_mse": tm, "best_val": best_val, "best_ep": best_ep,
            "t_epochs": t_epochs, "t_vals": t_vals}

# ── Bootstrap ───────────────────────────────────────────────────────────────
def boot_ratio(a, b, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    a, b = np.array(a), np.array(b)
    n_ = len(a)
    rats = [np.mean(a[rng.randint(0,n_,n_)]) / np.mean(b[rng.randint(0,n_,n_)]) for _ in range(n)]
    pt = float(np.mean(a) / np.mean(b))
    lo, hi = np.percentile(rats, [2.5, 97.5])
    return {"point": round(pt,3), "ci_lo": round(lo,3), "ci_hi": round(hi,3)}

# ── Main ────────────────────────────────────────────────────────────────────
def run():
    out = Path(__file__).parent.parent / "results" / "M6a_ondevice"
    out.mkdir(parents=True, exist_ok=True)
    all_res = {}

    for tname in TARGET_NAMES:
        print(f"\n{'='*60}\nTarget: {tname}\n{'='*60}")
        (out / tname).mkdir(exist_ok=True)

        x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data(tname, seed=42)
        coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=POLY_DEGREE)
        poly_mse = float(np.mean((eval_chebyshev(x_te, coeffs) - y_te)**2))
        lut_init = sample_polynomial_to_lut(coeffs, K=K, L=L)
        print(f"  PolyKAN MSE: {poly_mse:.3e}")

        tres = {"poly_mse": poly_mse, "regimes": {}, "ram": {}}
        mse_by_regime = {}

        for rkey, cfg in REGIMES.items():
            print(f"\n  [{rkey}] {cfg['label']}")
            seed_mses, seed_traces = [], []
            for seed in SEEDS:
                r = train_one(lut_init, x_tr, y_tr, x_v, y_v, x_te, y_te,
                              cfg["optimizer"], cfg["lr"], cfg["fp16_grad"], cfg["lam2"], seed)
                seed_mses.append(r["test_mse"])
                seed_traces.append({"ep": r["t_epochs"], "val": r["t_vals"]})
                print(f"    s={seed}  MSE={r['test_mse']:.3e}  best_ep={r['best_ep']}")

            mean_m = float(np.mean(seed_mses))
            std_m  = float(np.std(seed_mses))
            ratio  = poly_mse / mean_m
            mse_by_regime[rkey] = seed_mses
            ram = ram_bytes(K, L, cfg["optimizer"], cfg["fp16_grad"])
            tres["regimes"][rkey] = {
                "label": cfg["label"], "seed_mses": seed_mses,
                "mean_mse": mean_m, "std_mse": std_m,
                "ratio_vs_poly": round(ratio, 2),
                "traces": seed_traces,
            }
            tres["ram"][rkey] = {**ram, "total_kb": round(ram["total"]/1024, 2)}
            print(f"    mean={mean_m:.3e} ± {std_m:.0e}  ratio={ratio:.1f}×  RAM={ram['total']//1024+1} KB")

        tres["ci_B_vs_C"] = boot_ratio(mse_by_regime["B_sgd_f32"], mse_by_regime["C_sgd_fp16"])
        tres["ci_A_vs_C"] = boot_ratio(mse_by_regime["A_adam_f32"], mse_by_regime["C_sgd_fp16"])
        tres["ci_C_vs_D"] = boot_ratio(mse_by_regime["D_sgd_fp16_noreg"], mse_by_regime["C_sgd_fp16"])
        print(f"\n  B vs C (fp32 vs fp16 grad): {tres['ci_B_vs_C']}")
        print(f"  A vs C (Adam vs SGD-fp16):  {tres['ci_A_vs_C']}")
        print(f"  D vs C (no-reg vs reg):     {tres['ci_C_vs_D']}")
        all_res[tname] = tres

        # ── Plot ────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"M6a On-device training simulation — {tname}", fontsize=13)

        colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]
        ax = axes[0]
        keys = list(REGIMES.keys())
        short = ["A\nAdam f32\nserver", "B\nSGD f32\nMCU opt", "C\nSGD fp16\nMCU KEY", "D\nSGD fp16\nno-reg"]
        means = [tres["regimes"][k]["mean_mse"] for k in keys]
        stds  = [tres["regimes"][k]["std_mse"]  for k in keys]
        bars = ax.bar(short, means, yerr=stds, capsize=5, color=colors, alpha=0.82, width=0.6)
        ax.axhline(poly_mse, color="red", ls="--", lw=1.5, label=f"PolyKAN {poly_mse:.1e}")
        ax.set_yscale("log"); ax.set_ylabel("Test MSE"); ax.set_title("Test MSE by regime")
        ax.legend(fontsize=9)
        for bar, k in zip(bars, keys):
            r = tres["regimes"][k]["ratio_vs_poly"]
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.4,
                    f"{r:.0f}×" if abs(r) >= 1 else f"{r:.2f}×",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax2 = axes[1]
        for (rk, cfg), col in zip(REGIMES.items(), colors):
            traces = tres["regimes"][rk]["traces"]
            eps = traces[0]["ep"]
            vm = np.array([t["val"] for t in traces])
            med = np.median(vm, axis=0)
            lo = np.percentile(vm, 25, axis=0)
            hi = np.percentile(vm, 75, axis=0)
            ax2.plot(eps, med, color=col, lw=1.8, label=rk.replace("_"," "))
            ax2.fill_between(eps, lo, hi, color=col, alpha=0.15)
        ax2.axhline(poly_mse, color="red", ls="--", lw=1, label="PolyKAN")
        ax2.set_xlabel("Epoch"); ax2.set_yscale("log"); ax2.set_ylabel("Val MSE")
        ax2.set_title("Convergence (median ± IQR, 5 seeds)"); ax2.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(out / tname / "plot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save summary (no traces)
        clean_reg = {k: {kk: vv for kk, vv in v.items() if kk != "traces"}
                     for k, v in tres["regimes"].items()}
        with open(out / tname / "summary.json", "w") as f:
            json.dump({**tres, "regimes": clean_reg}, f, indent=2)

    # ── RAM plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    rkeys = list(REGIMES.keys())
    short4 = ["A: Adam f32\nserver", "B: SGD f32\nMCU opt", "C: SGD fp16\nMCU KEY", "D: SGD fp16\nno-reg"]
    rams_kb = [all_res["sine"]["ram"][k]["total_kb"] for k in rkeys]
    bars = ax.bar(short4, rams_kb, color=colors, alpha=0.82)
    for bar, kb in zip(bars, rams_kb):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f"{kb} KB", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(4,  color="gray", ls=":", lw=1, label="4 KB  (Cortex-M0+)")
    ax.axhline(16, color="gray", ls="--",lw=1, label="16 KB (Cortex-M4)")
    ax.axhline(64, color="gray", ls="-.",lw=1, label="64 KB (Cortex-M33)")
    ax.set_ylabel("RAM for training (KB)"); ax.set_ylim(0, max(rams_kb)*1.5)
    ax.set_title(f"RAM budget under on-device training — K={K}, L={L}")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    fig.savefig(out / "ram_budget.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Global summary ────────────────────────────────────────────────────────
    gsummary = {}
    for tname in TARGET_NAMES:
        tr = all_res[tname]
        gsummary[tname] = {
            "poly_mse": tr["poly_mse"],
            **{f"{rk}_mean_mse":   tr["regimes"][rk]["mean_mse"]   for rk in REGIMES},
            **{f"{rk}_ratio_poly": tr["regimes"][rk]["ratio_vs_poly"] for rk in REGIMES},
            "ci_B_vs_C": tr["ci_B_vs_C"],
            "ci_A_vs_C": tr["ci_A_vs_C"],
            "ci_C_vs_D": tr["ci_C_vs_D"],
            "ram_A_kb":  tr["ram"]["A_adam_f32"]["total_kb"],
            "ram_C_kb":  tr["ram"]["C_sgd_fp16"]["total_kb"],
        }
    with open(out / "summary.json", "w") as f:
        json.dump(gsummary, f, indent=2)

    print(f"\n{'='*60}\nM6a COMPLETE\n{'='*60}")
    for tname, s in gsummary.items():
        print(f"\n  {tname}  (poly={s['poly_mse']:.2e})")
        for rk in REGIMES:
            print(f"    {rk:20s}  MSE={s[f'{rk}_mean_mse']:.2e}  {s[f'{rk}_ratio_poly']:.1f}× vs poly")
        print(f"    B vs C (fp32 vs fp16): {s['ci_B_vs_C']}")
        print(f"    C vs D (reg vs noreg): {s['ci_C_vs_D']}")

if __name__ == "__main__":
    run()
