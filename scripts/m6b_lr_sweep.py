"""
M6b — SGD learning-rate sweep for on-device direct-LUT training.

M6a established that fp16-grad SGD has no accuracy penalty vs fp32-grad SGD.
It used lr=0.5 (chosen from a quick 4-point sweep). This phase does a
proper lr sweep to find the optimal SGD learning rate and check whether
decay schedules help once the right lr is identified.

Regimes (all: fp16-grad SGD, λ₂=1.0, 900 epochs, K=16, L=32):
  lr ∈ {0.1, 0.5, 1.0, 2.0}   (flat schedule)
  lr=1.0 + cosine decay 1.0→0.01
  lr=1.0 + step 300/300/300ep: 1.0 / 0.2 / 0.02

Targets: sine, cusp, saturating. 5 seeds.
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
from lut_native.baselines import fit_chebyshev_ls, eval_chebyshev, sample_polynomial_to_lut
from lut_native.targets import generate_data

K, L = 16, 32
EPOCHS = 900
SEEDS = [0, 1, 2]
TARGETS = ["sine", "cusp", "saturating"]
EVAL_EVERY = 30
BS = 64


def make_schedule(name: str, epochs: int):
    if name.startswith("flat_"):
        lr = float(name.split("_")[1])
        return [lr] * epochs
    if name == "cosine_1.0":
        return [0.01 + 1.0 * (1 + np.cos(np.pi * ep / epochs)) / 2 for ep in range(epochs)]
    if name == "step_1.0":
        # 300ep @ 1.0, 300ep @ 0.2, 300ep @ 0.02
        return [1.0] * 300 + [0.2] * 300 + [0.02] * 300
    raise ValueError(name)


SCHEDULES = {
    "flat_0.1":   make_schedule("flat_0.1",   EPOCHS),
    "flat_0.5":   make_schedule("flat_0.5",   EPOCHS),
    "flat_1.0":   make_schedule("flat_1.0",   EPOCHS),
    "flat_2.0":   make_schedule("flat_2.0",   EPOCHS),
    "cosine_1.0": make_schedule("cosine_1.0", EPOCHS),
    "step_1.0":   make_schedule("step_1.0",   EPOCHS),
}


def train_one(lut_init, x_tr, y_tr, x_v, y_v, x_te, y_te, schedule, seed):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    lut_r = max(float(lut_init.max() - lut_init.min()), 1e-6)
    lut_n = lut_init.copy() + rng.randn(*lut_init.shape).astype(np.float32) * 0.001
    edge = LUTEdge(K, L); edge.init_from_array(lut_n)
    opt = torch.optim.SGD(edge.parameters(), lr=schedule[0], momentum=0.0)
    xt = torch.from_numpy(x_tr); yt = torch.from_numpy(y_tr)
    xv = torch.from_numpy(x_v);  yv = torch.from_numpy(y_v)
    xte= torch.from_numpy(x_te); yte= torch.from_numpy(y_te)
    best_val, best_lut, best_ep = float("inf"), lut_n.copy(), 0
    tv_ep, tv_val = [], []
    for ep in range(EPOCHS):
        for g in opt.param_groups: g["lr"] = schedule[ep]
        edge.train()
        idx = torch.randperm(len(xt))
        for i in range(0, len(xt), BS):
            bi = idx[i:i+BS]; opt.zero_grad()
            loss = torch.mean((edge(xt[bi]) - yt[bi])**2) + combined_penalty(edge.lut, 0, 1.0, 0, 0)
            loss.backward()
            for p in edge.parameters():
                if p.grad is not None: p.grad.data = p.grad.data.half().float()
            opt.step()
        if (ep+1) % EVAL_EVERY == 0:
            edge.eval()
            with torch.no_grad(): vm = float(torch.mean((edge(xv)-yv)**2))
            tv_ep.append(ep+1); tv_val.append(vm)
            if vm < best_val:
                best_val = vm; best_lut = edge.lut.detach().numpy().copy(); best_ep = ep+1
    be = LUTEdge(K, L); be.init_from_array(best_lut); be.eval()
    with torch.no_grad(): tm = float(torch.mean((be(xte)-yte)**2))
    return {"test_mse": tm, "best_ep": best_ep, "tv_ep": tv_ep, "tv_val": tv_val}


def boot_ratio(a, b, n=1000, seed=0):
    rng = np.random.RandomState(seed); a, b = np.array(a), np.array(b)
    n_ = len(a)
    rats = [np.mean(a[rng.randint(0,n_,n_)]) / np.mean(b[rng.randint(0,n_,n_)]) for _ in range(n)]
    return {"point": round(float(np.mean(a)/np.mean(b)),3),
            "ci_lo": round(float(np.percentile(rats,2.5)),3),
            "ci_hi": round(float(np.percentile(rats,97.5)),3)}


def run():
    out = Path(__file__).parent.parent / "results" / "M6b_lr_sweep"
    out.mkdir(parents=True, exist_ok=True)
    all_res = {}

    for tname in TARGETS:
        print(f"\n{'='*60}\nTarget: {tname}\n{'='*60}")
        (out / tname).mkdir(exist_ok=True)
        x_tr,y_tr,x_v,y_v,x_te,y_te = generate_data(tname, seed=42)
        coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=16)
        poly_mse = float(np.mean((eval_chebyshev(x_te,coeffs)-y_te)**2))
        lut_init = sample_polynomial_to_lut(coeffs, K=K, L=L)
        print(f"  PolyKAN MSE: {poly_mse:.3e}")

        tres = {"poly_mse": poly_mse, "schedules": {}}
        mse_by_sched = {}

        for sname, sched in SCHEDULES.items():
            mses, traces = [], []
            for seed in SEEDS:
                r = train_one(lut_init,x_tr,y_tr,x_v,y_v,x_te,y_te,sched,seed)
                mses.append(r["test_mse"])
                traces.append({"ep": r["tv_ep"], "val": r["tv_val"]})
            mean_m = float(np.mean(mses)); std_m = float(np.std(mses))
            ratio = poly_mse / mean_m
            mse_by_sched[sname] = mses
            tres["schedules"][sname] = {
                "seed_mses": mses, "mean_mse": mean_m, "std_mse": std_m,
                "ratio_vs_poly": round(ratio, 2), "traces": traces,
            }
            print(f"  {sname:14s}  mean={mean_m:.3e} ± {std_m:.0e}  {ratio:.1f}× vs poly")

        # Key CIs: flat_1.0 vs flat_0.5, cosine vs flat_1.0, step vs flat_1.0
        tres["ci_10_vs_05"] = boot_ratio(mse_by_sched["flat_0.5"],  mse_by_sched["flat_1.0"])
        tres["ci_cos_vs_10"] = boot_ratio(mse_by_sched["cosine_1.0"], mse_by_sched["flat_1.0"])
        tres["ci_step_vs_10"]= boot_ratio(mse_by_sched["step_1.0"],  mse_by_sched["flat_1.0"])
        print(f"  flat_0.5 / flat_1.0 = {tres['ci_10_vs_05']}")
        print(f"  cosine / flat_1.0   = {tres['ci_cos_vs_10']}")
        print(f"  step   / flat_1.0   = {tres['ci_step_vs_10']}")
        all_res[tname] = tres

        # ── Plot ─────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"M6b LR sweep (SGD fp16-grad, λ₂=1) — {tname}", fontsize=13)

        snames = list(SCHEDULES.keys())
        colors = ["#9E9E9E","#4CAF50","#2196F3","#F44336","#FF9800","#9C27B0"]
        ax = axes[0]
        means = [tres["schedules"][s]["mean_mse"] for s in snames]
        stds  = [tres["schedules"][s]["std_mse"]  for s in snames]
        short = ["lr=0.1","lr=0.5\n(M6a)","lr=1.0","lr=2.0","cosine\n1.0→0","step\n1→0.2→0.02"]
        bars = ax.bar(short, means, yerr=stds, capsize=5, color=colors, alpha=0.82, width=0.6)
        ax.axhline(poly_mse, color="red", ls="--", lw=1.5, label=f"PolyKAN {poly_mse:.1e}")
        ax.set_yscale("log"); ax.set_ylabel("Test MSE"); ax.set_title("Test MSE by schedule")
        ax.legend(fontsize=9)
        for bar, s in zip(bars, snames):
            r = tres["schedules"][s]["ratio_vs_poly"]
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.4,
                    f"{r:.0f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax2 = axes[1]
        for sn, col in zip(snames, colors):
            traces = tres["schedules"][sn]["traces"]
            eps = traces[0]["ep"]
            vm = np.array([t["val"] for t in traces])
            med = np.median(vm, axis=0)
            lo = np.percentile(vm, 25, axis=0); hi = np.percentile(vm, 75, axis=0)
            ax2.plot(eps, med, color=col, lw=1.8, label=sn)
            ax2.fill_between(eps, lo, hi, color=col, alpha=0.12)
        ax2.axhline(poly_mse, color="red", ls="--", lw=1, label="PolyKAN")
        ax2.set_xlabel("Epoch"); ax2.set_yscale("log"); ax2.set_ylabel("Val MSE")
        ax2.set_title("Convergence (median ± IQR, 5 seeds)"); ax2.legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(out / tname / "plot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        clean = {k: {kk:vv for kk,vv in v.items() if kk!="traces"}
                 for k,v in tres["schedules"].items()}
        with open(out/tname/"summary.json","w") as f:
            json.dump({**tres,"schedules":clean}, f, indent=2)

    # Global summary
    gsummary = {}
    for tname in TARGETS:
        tr = all_res[tname]
        best_sched = min(tr["schedules"], key=lambda s: tr["schedules"][s]["mean_mse"])
        gsummary[tname] = {
            "poly_mse": tr["poly_mse"],
            "best_schedule": best_sched,
            "best_mean_mse": tr["schedules"][best_sched]["mean_mse"],
            "best_ratio_vs_poly": tr["schedules"][best_sched]["ratio_vs_poly"],
            "flat_05_ratio": tr["schedules"]["flat_0.5"]["ratio_vs_poly"],
            "flat_10_ratio": tr["schedules"]["flat_1.0"]["ratio_vs_poly"],
            "ci_10_vs_05":  tr["ci_10_vs_05"],
        }
    with open(out/"summary.json","w") as f:
        json.dump(gsummary, f, indent=2)

    print(f"\n{'='*60}\nM6b COMPLETE\n{'='*60}")
    for tname, s in gsummary.items():
        print(f"  {tname}: best={s['best_schedule']} {s['best_ratio_vs_poly']}× vs poly  "
              f"(M6a lr=0.5 was {s['flat_05_ratio']}×)")
        print(f"    CI flat_0.5/flat_1.0 = {s['ci_10_vs_05']}")


if __name__ == "__main__":
    run()
