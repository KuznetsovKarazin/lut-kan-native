#!/usr/bin/env python3
"""
M4d: Progressive unfreezing.

Motivation:
    In joint training, layer 1 and layer 2 weights move simultaneously.
    Layer 2's optimization target is a function of layer 1's output, so if
    layer 1 changes mid-training, layer 2 is chasing a moving target.
    Freezing one layer first should let the other stabilize.

Three scenarios, compared against best M4c config (L1_slower) as joint baseline:

    J) Joint (M4c best: L1_slower, lr_l1=1e-4, lr_l2=5e-4): 200 epochs
    A) L2-first → joint: freeze L1, train L2 for 100 ep; unfreeze, joint 100 ep
    B) L1-first → joint: freeze L2, train L1 for 100 ep; unfreeze, joint 100 ep
    C) L2-first → L1 fine-tune → joint: freeze L1, train L2 for 70 ep;
       freeze L2 + unfreeze L1, train L1 for 70 ep; joint 60 ep.
       (The "classical" staged schedule.)

Fixed: alpha=0.1, seeds 0-2, tasks composition_1d + feynman_2d.

Stop criterion check (per your threshold):
    If feynman_2d ceiling ≤ 1.25-1.30× AND no clear upward trend M4c→M4d,
    close the multi-edge topic.

Output: results/M4d_progressive/{composition_1d,feynman_2d}/{summary.json,plot.png}
Runtime: ~15 min CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lut_native import (  # noqa: E402
    ResidualLUTKAN2Layer,
    ResidualTrainConfig,
    generate_data_2d,
    sample_polynomial_to_lut,
    train_residual_kan2,
)
from exp_H4_kan2_2d import train_poly_kan2
from m2b_composition import generate_data_1d


def _init_from_poly(poly_c1, poly_c2, in_dim, hidden, K, L, alpha):
    model = ResidualLUTKAN2Layer(
        in_dim=in_dim, hidden_dim=hidden, out_dim=1, K=K, L=L, alpha=alpha,
    )
    init_l1 = np.zeros((in_dim, hidden, K, L), dtype=np.float32)
    for i in range(in_dim):
        for h in range(hidden):
            init_l1[i, h] = sample_polynomial_to_lut(poly_c1[i, h], K=K, L=L)
    init_l2 = np.zeros((hidden, 1, K, L), dtype=np.float32)
    for h in range(hidden):
        init_l2[h, 0] = sample_polynomial_to_lut(poly_c2[h, 0], K=K, L=L)
    model.init_layer1_from_arrays(init_l1)
    model.init_layer2_from_arrays(init_l2)
    return model


def _train_staged(model, stages, x_tr, y_tr, x_v, y_v, x_te, y_te, seed):
    """Train with multiple stages. Each stage is a dict with:
         'name', 'lr_l1', 'lr_l2', 'freeze_l1', 'freeze_l2', 'epochs'.

    Freezing is implemented by setting the corresponding lr to 0 (Adam
    with lr=0 effectively doesn't move the parameter, modulo second-moment
    statistics — see the note below).

    We explicitly zero the gradient of the frozen param before each step
    to ensure full freezing. This is more robust than lr=0 because Adam's
    momentum from previous stages can otherwise carry over.

    Returns merged trace and the best-val weights seen across all stages.
    """
    merged_trace = {"epoch": [], "mse_train": [], "mse_val": [],
                    "stage": [], "delta_l1_l2": [], "delta_l2_l2": []}

    # Initial eval
    with torch.no_grad():
        x_v_t = torch.from_numpy(x_v.astype(np.float32).reshape(-1, model.in_dim))
        y_v_t = torch.from_numpy(y_v.astype(np.float32).reshape(-1, model.out_dim))
        x_t_t = torch.from_numpy(x_tr.astype(np.float32).reshape(-1, model.in_dim))
        y_t_t = torch.from_numpy(y_tr.astype(np.float32).reshape(-1, model.out_dim))
        x_e_t = torch.from_numpy(x_te.astype(np.float32).reshape(-1, model.in_dim))
        y_e_t = torch.from_numpy(y_te.astype(np.float32).reshape(-1, model.out_dim))
        mse_val_init = ((model(x_v_t) - y_v_t) ** 2).mean().item()

    best_val = mse_val_init
    best_d1 = model.delta_l1.detach().cpu().numpy().copy()
    best_d2 = model.delta_l2.detach().cpu().numpy().copy()
    best_epoch_global = 0
    best_stage = "init"

    epoch_offset = 0
    merged_trace["epoch"].append(0)
    merged_trace["mse_train"].append(None)
    merged_trace["mse_val"].append(mse_val_init)
    merged_trace["stage"].append("init")
    merged_trace["delta_l1_l2"].append(0.0)
    merged_trace["delta_l2_l2"].append(0.0)

    for stage in stages:
        # Fresh optimizer per stage (matches what happens when we
        # re-wire which parameters are trainable).
        torch.manual_seed(seed + hash(stage["name"]) % 2**31)
        param_groups = []
        if not stage["freeze_l1"]:
            param_groups.append({"params": [model.delta_l1], "lr": stage["lr_l1"]})
        if not stage["freeze_l2"]:
            param_groups.append({"params": [model.delta_l2], "lr": stage["lr_l2"]})
        if not param_groups:
            raise ValueError(f"Stage {stage['name']} freezes both layers")
        optim = torch.optim.Adam(param_groups)

        N = x_t_t.shape[0]
        for ep in range(stage["epochs"]):
            perm = torch.randperm(N)
            for s in range(0, N, 128):
                idx = perm[s:s + 128]
                xb, yb = x_t_t[idx], y_t_t[idx]
                optim.zero_grad()
                y_pred = model(xb)
                loss = ((y_pred - yb) ** 2).mean()
                loss.backward()
                # Zero gradient of frozen params (belt-and-braces)
                if stage["freeze_l1"] and model.delta_l1.grad is not None:
                    model.delta_l1.grad.zero_()
                if stage["freeze_l2"] and model.delta_l2.grad is not None:
                    model.delta_l2.grad.zero_()
                optim.step()

            if (ep + 1) % 10 == 0 or (ep + 1) == stage["epochs"]:
                global_ep = epoch_offset + ep + 1
                with torch.no_grad():
                    mse_t = ((model(x_t_t) - y_t_t) ** 2).mean().item()
                    mse_v = ((model(x_v_t) - y_v_t) ** 2).mean().item()
                    dn = model.delta_norms()
                merged_trace["epoch"].append(global_ep)
                merged_trace["mse_train"].append(mse_t)
                merged_trace["mse_val"].append(mse_v)
                merged_trace["stage"].append(stage["name"])
                merged_trace["delta_l1_l2"].append(dn["delta_l1_l2"])
                merged_trace["delta_l2_l2"].append(dn["delta_l2_l2"])
                if mse_v < best_val:
                    best_val = mse_v
                    best_d1 = model.delta_l1.detach().cpu().numpy().copy()
                    best_d2 = model.delta_l2.detach().cpu().numpy().copy()
                    best_epoch_global = global_ep
                    best_stage = stage["name"]

        epoch_offset += stage["epochs"]

    # Restore best and eval
    with torch.no_grad():
        model.delta_l1.data = torch.from_numpy(best_d1)
        model.delta_l2.data = torch.from_numpy(best_d2)
        mse_test_at_best = ((model(x_e_t) - y_e_t) ** 2).mean().item()
        mse_val_final = ((model(x_v_t) - y_v_t) ** 2).mean().item()

    return {
        "mse_val_init": mse_val_init,
        "mse_val_at_best": best_val,
        "mse_test_at_best": mse_test_at_best,
        "mse_val_final": mse_val_final,
        "best_epoch": best_epoch_global,
        "best_stage": best_stage,
        "trace": merged_trace,
        "delta_l1_l2_best": float(np.linalg.norm(best_d1)),
        "delta_l2_l2_best": float(np.linalg.norm(best_d2)),
    }


def build_scenarios():
    """Each scenario is a list of training stages."""
    LR_L1 = 1e-4   # from M4c: L1 prefers slower
    LR_L2 = 5e-4
    # For freeze phases, non-zero dummy lr is fine since optimizer won't
    # include that parameter in param_groups.
    return {
        "J_joint": [
            {"name": "joint", "lr_l1": LR_L1, "lr_l2": LR_L2,
             "freeze_l1": False, "freeze_l2": False, "epochs": 200},
        ],
        "A_L2first_then_joint": [
            {"name": "L2only", "lr_l1": 0.0, "lr_l2": LR_L2,
             "freeze_l1": True, "freeze_l2": False, "epochs": 100},
            {"name": "joint", "lr_l1": LR_L1, "lr_l2": LR_L2,
             "freeze_l1": False, "freeze_l2": False, "epochs": 100},
        ],
        "B_L1first_then_joint": [
            {"name": "L1only", "lr_l1": LR_L1, "lr_l2": 0.0,
             "freeze_l1": False, "freeze_l2": True, "epochs": 100},
            {"name": "joint", "lr_l1": LR_L1, "lr_l2": LR_L2,
             "freeze_l1": False, "freeze_l2": False, "epochs": 100},
        ],
        "C_L2_L1_joint": [
            {"name": "L2only", "lr_l1": 0.0, "lr_l2": LR_L2,
             "freeze_l1": True, "freeze_l2": False, "epochs": 70},
            {"name": "L1only", "lr_l1": LR_L1, "lr_l2": 0.0,
             "freeze_l1": False, "freeze_l2": True, "epochs": 70},
            {"name": "joint", "lr_l1": LR_L1, "lr_l2": LR_L2,
             "freeze_l1": False, "freeze_l2": False, "epochs": 60},
        ],
    }


def run_scenario(name, stages, poly_coeffs_per_seed, seeds,
                 in_dim, hidden, K, L, alpha,
                 x_tr, y_tr, x_v, y_v, x_te, y_te):
    per_seed = []
    trace_seed0 = None
    for idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        c1, c2 = poly_coeffs_per_seed[idx]
        model = _init_from_poly(c1, c2, in_dim, hidden, K, L, alpha)
        out = _train_staged(model, stages, x_tr, y_tr, x_v, y_v, x_te, y_te, seed)
        per_seed.append({
            "seed": seed,
            "mse_val_init": out["mse_val_init"],
            "mse_val_at_best": out["mse_val_at_best"],
            "mse_test_at_best": out["mse_test_at_best"],
            "mse_val_final": out["mse_val_final"],
            "best_epoch": out["best_epoch"],
            "best_stage": out["best_stage"],
        })
        if idx == 0:
            trace_seed0 = out["trace"]
    arr = np.array([p["mse_test_at_best"] for p in per_seed])
    return {
        "name": name,
        "per_seed": per_seed,
        "test_mse_mean": float(arr.mean()),
        "test_mse_std": float(arr.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "trace_seed0": trace_seed0,
    }


def task_pipeline(task_name, x_tr, y_tr, x_v, y_v, x_te, y_te,
                  in_dim, hidden, K, L, degree, alpha, seeds,
                  poly_epochs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\n{task_name}\n{'='*72}")

    # Train polys
    poly_coeffs = []
    poly_mses = []
    for seed in seeds:
        pres = train_poly_kan2(
            in_dim=in_dim, hidden_dim=hidden, out_dim=1, degree=degree,
            x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
            lr=5e-3, epochs=poly_epochs, batch_size=128, seed=seed,
        )
        poly_coeffs.append((pres["coeffs_l1"], pres["coeffs_l2"]))
        poly_mses.append(pres["test_mse"])
        print(f"  PolyKAN2 seed={seed}: {pres['test_mse']:.3e}")
    poly_mean = float(np.mean(poly_mses))
    print(f"  PolyKAN2 mean: {poly_mean:.3e}")

    scenarios = build_scenarios()
    print(f"\n  Progressive unfreezing (alpha={alpha}, {len(seeds)} seeds):")
    print(f"  {'scenario':<24} {'test MSE':>18} {'best_ep':>8} "
          f"{'best_stage':>12} {'vs Poly':>8}")

    results = []
    for name, stages in scenarios.items():
        t0 = time.time()
        r = run_scenario(name, stages, poly_coeffs, seeds,
                         in_dim, hidden, K, L, alpha,
                         x_tr, y_tr, x_v, y_v, x_te, y_te)
        results.append(r)
        best_stages = [p["best_stage"] for p in r["per_seed"]]
        best_stage_s0 = r["per_seed"][0]["best_stage"]
        print(f"  {name:<24} {r['test_mse_mean']:>10.2e}±{r['test_mse_std']:>5.1e} "
              f"{r['per_seed'][0]['best_epoch']:>8d} "
              f"{best_stage_s0:>12} "
              f"{poly_mean/r['test_mse_mean']:>6.2f}x   (dt={time.time()-t0:.1f}s)")

    out = {
        "task": task_name,
        "config": {"in_dim": in_dim, "hidden": hidden, "K": K, "L": L,
                   "degree": degree, "alpha": alpha, "seeds": seeds,
                   "poly_epochs": poly_epochs},
        "poly_baseline": {"test_mse_per_seed": poly_mses, "test_mse_mean": poly_mean},
        "results": results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    names = [r["name"] for r in results]
    means = np.array([r["test_mse_mean"] for r in results])
    stds = np.array([r["test_mse_std"] for r in results])
    colors = plt.cm.tab10(np.arange(len(names)))
    ax1.bar(np.arange(len(names)), means, yerr=stds, capsize=4, color=colors,
            edgecolor="black")
    ax1.axhline(poly_mean, color="k", linestyle="--", linewidth=1.2,
                label=f"PolyKAN2 = {poly_mean:.2e}")
    ax1.set_xticks(np.arange(len(names)))
    ax1.set_xticklabels(names, rotation=15, fontsize=8)
    ax1.set_yscale("log")
    ax1.set_ylabel("Test MSE (best-val)")
    ax1.set_title(f"{task_name}: progressive unfreezing")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3, axis="y")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m * 1.1, f"{m:.2e}", ha="center", fontsize=7.5)

    for i, r in enumerate(results):
        tr = r["trace_seed0"]
        ax2.semilogy(tr["epoch"], tr["mse_val"], color=colors[i],
                     linewidth=1.3, label=r["name"])
    ax2.axhline(poly_mean, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.set_xlabel("Epoch (global)")
    ax2.set_ylabel("Val MSE")
    ax2.set_title(f"{task_name}: val-MSE trace (seed 0)")
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "plot.png", dpi=130)
    plt.close()
    print(f"  -> {out_dir / 'plot.png'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/M4d_progressive")
    ap.add_argument("--poly-epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_tr1, y_tr1, x_v1, y_v1, x_te1, y_te1 = generate_data_1d(1000, 400, 400, 42)
    r1 = task_pipeline("composition_1d", x_tr1, y_tr1, x_v1, y_v1, x_te1, y_te1,
                       in_dim=1, hidden=4, K=16, L=32, degree=12,
                       alpha=args.alpha, seeds=args.seeds,
                       poly_epochs=args.poly_epochs,
                       out_dir=out_dir / "composition_1d")

    x_tr2, y_tr2, x_v2, y_v2, x_te2, y_te2 = generate_data_2d(
        "feynman_2d", n_train=1000, n_val=400, n_test=400, seed=42)
    r2 = task_pipeline("feynman_2d", x_tr2, y_tr2, x_v2, y_v2, x_te2, y_te2,
                       in_dim=2, hidden=4, K=16, L=32, degree=8,
                       alpha=args.alpha, seeds=args.seeds,
                       poly_epochs=args.poly_epochs,
                       out_dir=out_dir / "feynman_2d")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"composition_1d": r1, "feynman_2d": r2}, f, indent=2)


if __name__ == "__main__":
    main()
