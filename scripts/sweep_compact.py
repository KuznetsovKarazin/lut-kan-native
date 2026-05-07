"""
sweep_compact.py — systematic search for stable compact LUT-KAN architectures.

Compares LUT-KAN against a MATCHED polynomial KAN:
  - Same dims (same layers and widths)
  - Same training algorithm (Adam, same lr/epochs/seed/batch_size)
  - Same parameter count per edge: LUT has K×L cells, poly has degree K×L-1

Measures both accuracy (MSE) and training speed (ms/epoch, seconds to target MSE).

Usage
-----
# Quick mode — 1D targets, small grid, 3 seeds:
  python scripts/sweep_compact.py --mode quick

# Standard mode — all 1D targets, more architectures:
  python scripts/sweep_compact.py --mode standard

# Custom architectures:
  python scripts/sweep_compact.py --dims 1,4,1 1,4,4,1 --kl 1,4 2,4 2,8 --targets sine cusp

# Single-edge mode (the original H1 regime: [1→1]):
  python scripts/sweep_compact.py --mode single_edge

All options:
  python scripts/sweep_compact.py --help
"""

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lut_native.targets import generate_data, TARGETS
from lut_native.kan_stack import LUTKANStack
from lut_native.training_stack import StackTrainConfig, train_lut_stack
from lut_native.poly_kan2 import PolyKAN2Layer, train_poly_kan2
from lut_native.baselines import fit_chebyshev_ls, eval_chebyshev, sample_polynomial_to_lut
from lut_native.training import TrainConfig, train_lut_edge


# ─────────────────────────────────────────────────────────────────────────────
# Coverage / sizing utilities
# ─────────────────────────────────────────────────────────────────────────────

def bottleneck_density(dims: List[int], K: int, L: int, n_train: int) -> float:
    densities = []
    for i in range(len(dims) - 1):
        cells = dims[i] * dims[i + 1] * K * L
        densities.append(n_train / max(cells, 1))
    return min(densities) if densities else 0.0


def coverage_flag(density: float) -> str:
    if density >= 20: return "✓ good"
    if density >= 8:  return "~ ok"
    if density >= 3:  return "! low"
    return "✗ bad"


def total_lut_bytes(dims: List[int], K: int, L: int) -> int:
    total = 0
    for i in range(len(dims) - 1):
        n_edges = dims[i] * dims[i + 1]
        total += n_edges * (K * L + 4 * K)
    return total


def _dims_str(dims: List[int]) -> str:
    return "[" + "→".join(str(d) for d in dims) + "]"


# ─────────────────────────────────────────────────────────────────────────────
# Single-edge (1→1) comparison
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EdgeResult:
    K: int
    L: int
    target: str
    seed: int
    lut_mse: float
    poly_mse: float
    lut_best_epoch: int
    lut_ms_per_epoch: float
    density: float
    mem_bytes: int


def run_single_edge(
    K: int, L: int, target: str, seed: int,
    n_train: int, epochs: int, lambda_2: float, lr: float,
) -> EdgeResult:
    d = generate_data(target, n_train=n_train, seed=seed * 17 + 3)
    x_tr, y_tr, x_val, y_val, x_te, y_te = d

    degree = K * L - 1
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=degree)
    lut_init = sample_polynomial_to_lut(coeffs, K=K, L=L)
    poly_mse = float(np.mean((eval_chebyshev(x_te, coeffs) - y_te) ** 2))

    cfg = TrainConfig(epochs=epochs, seed=seed, lambda_2=lambda_2, lr=lr)
    t0 = time.perf_counter()
    r = train_lut_edge(lut_init, x_tr, y_tr, x_val, y_val, x_te, y_te, -1.0, 1.0, cfg)
    elapsed = time.perf_counter() - t0

    return EdgeResult(
        K=K, L=L, target=target, seed=seed,
        lut_mse=float(r.mse_test_at_best),
        poly_mse=poly_mse,
        lut_best_epoch=r.best_epoch,
        lut_ms_per_epoch=elapsed / epochs * 1000,
        density=n_train / max(K * L, 1),
        mem_bytes=K * L + 4 * K,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-layer comparison: LUT stack vs poly KAN, matched dims + params
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StackResult:
    dims: List[int]
    K: int
    L: int
    target: str
    seed: int
    lut_mse: float
    poly_mse: float
    lut_ms_per_epoch: float
    poly_ms_per_epoch: float
    lut_best_epoch: int
    poly_best_epoch: int
    norm_uniformity: float
    density: float
    mem_bytes: int
    n_lut_params: int


def run_stack_vs_poly(
    dims: List[int], K: int, L: int,
    target: str, seed: int,
    n_train: int, epochs: int,
    lambda_2: float, lr: float, patience: int,
    recal_every: int, ema_alpha: float,
) -> StackResult:
    d = generate_data(target, n_train=n_train, seed=seed * 17 + 3)
    x_tr, y_tr, x_val, y_val, x_te, y_te = d
    degree = K * L - 1

    # Poly KAN — same dims, same training setup, degree = K*L-1 per edge
    poly_model = PolyKAN2Layer(dims[0], dims[1], dims[-1], degree=degree)
    t0 = time.perf_counter()
    rp = train_poly_kan2(
        poly_model, x_tr, y_tr, x_val, y_val, x_te, y_te,
        lr=lr, epochs=epochs, seed=seed,
    )
    t_poly = time.perf_counter() - t0

    # LUT KAN stack — same dims, K segments, L cells/edge
    lut_model = LUTKANStack(dims, K=K, L=L)
    cfg = StackTrainConfig(
        lambda_2=lambda_2, lr=lr, epochs=epochs, seed=seed,
        cheby_init=True, cheby_scale=1.5, cheby_noise=0.05,
        freeze_norms=True, smooth_norms=True,
        ema_alpha=ema_alpha, recalibrate_every_epochs=recal_every,
        calibrate_at_start=True,
        eval_every_epochs=max(10, epochs // 100),
    )
    t0 = time.perf_counter()
    rl = train_lut_stack(lut_model, x_tr, y_tr, x_val, y_val, x_te, y_te, cfg, patience=patience)
    t_lut = time.perf_counter() - t0

    # norm uniformity
    lut_model.load_snapshot(rl.best_luts, rl.best_norms)
    x_t = torch.tensor(x_tr.reshape(-1, dims[0]), dtype=torch.float32)
    if x_t.dim() == 1:
        x_t = x_t.unsqueeze(-1)
    uni_scores = []
    z = x_t
    for i, blk in enumerate(lut_model.blocks[:-1]):
        z_raw = blk(z)
        uni_scores.append(lut_model.norms[i].coverage_stats(z_raw, K)["uniformity"])
        z = lut_model.norms[i](z_raw)
    norm_uni = float(np.mean(uni_scores)) if uni_scores else -1.0

    return StackResult(
        dims=dims, K=K, L=L, target=target, seed=seed,
        lut_mse=float(rl.mse_test_at_best),
        poly_mse=float(rp["mse_test_at_best"]),
        lut_ms_per_epoch=t_lut / epochs * 1000,
        poly_ms_per_epoch=t_poly / epochs * 1000,
        lut_best_epoch=rl.best_epoch,
        poly_best_epoch=rp["best_epoch"],
        norm_uniformity=norm_uni,
        density=bottleneck_density(dims, K, L, n_train),
        mem_bytes=total_lut_bytes(dims, K, L),
        n_lut_params=lut_model.n_lut_params(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Summaries
# ─────────────────────────────────────────────────────────────────────────────

def summarise_edge(runs: List[EdgeResult]) -> dict:
    r0 = runs[0]
    luts = [r.lut_mse for r in runs]
    polys = [r.poly_mse for r in runs]
    return dict(
        K=r0.K, L=r0.L, target=r0.target, n_seeds=len(runs),
        lut_mean=float(np.mean(luts)), lut_std=float(np.std(luts)),
        poly_mean=float(np.mean(polys)),
        ratio_mean=float(np.mean(polys) / max(np.mean(luts), 1e-30)),
        lut_ms_per_ep=float(np.mean([r.lut_ms_per_epoch for r in runs])),
        density=r0.density, mem_bytes=r0.mem_bytes,
    )


def summarise_stack(runs: List[StackResult]) -> dict:
    r0 = runs[0]
    luts = [r.lut_mse for r in runs]
    polys = [r.poly_mse for r in runs]
    unis = [r.norm_uniformity for r in runs if r.norm_uniformity >= 0]
    lut_ms = [r.lut_ms_per_epoch for r in runs]
    poly_ms = [r.poly_ms_per_epoch for r in runs]
    return dict(
        dims=r0.dims, K=r0.K, L=r0.L, target=r0.target, n_seeds=len(runs),
        lut_mse_mean=float(np.mean(luts)), lut_mse_std=float(np.std(luts)),
        poly_mse_mean=float(np.mean(polys)), poly_mse_std=float(np.std(polys)),
        ratio_mean=float(np.mean(polys) / max(np.mean(luts), 1e-30)),
        lut_ms_per_ep=float(np.mean(lut_ms)),
        poly_ms_per_ep=float(np.mean(poly_ms)),
        speed_ratio=float(np.mean(poly_ms) / max(np.mean(lut_ms), 1e-9)),
        norm_uni=float(np.mean(unis)) if unis else -1.0,
        lut_best_ep=float(np.mean([r.lut_best_epoch for r in runs])),
        poly_best_ep=float(np.mean([r.poly_best_epoch for r in runs])),
        density=r0.density,
        mem_bytes=r0.mem_bytes,
        n_lut_params=r0.n_lut_params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_edge_table(summaries: List[dict], title: str = "") -> None:
    if title:
        print(f"\n{'─'*76}\n  {title}\n{'─'*76}")
    print(f"  {'K,L':8s} {'params':>6s}  {'LUT MSE':>10s} {'Poly LS':>10s}  "
          f"{'poly/LUT':>9s}  {'ms/ep':>6s}  {'bytes':>5s}  cov  cv")
    print("  " + "─" * 72)
    for s in sorted(summaries, key=lambda x: x["lut_mean"]):
        cv = s["lut_std"] / max(s["lut_mean"], 1e-30)
        ratio = s["ratio_mean"]
        sym = "✓" if ratio > 1.0 else "✗"
        print(f"  K={s['K']},L={s['L']:<2d} {s['K']*s['L']:>6d}  "
              f"{s['lut_mean']:>10.3e} {s['poly_mean']:>10.3e}  "
              f"{ratio:>8.1f}×{sym}  {s['lut_ms_per_ep']:>6.2f}  "
              f"{s['mem_bytes']:>5d}  {coverage_flag(s['density'])}  {cv:.2f}")


def print_stack_table(summaries: List[dict], title: str = "") -> None:
    if title:
        print(f"\n{'─'*96}\n  {title}\n{'─'*96}")
    print(f"  {'arch':18s} {'K':>2s} {'L':>3s}  {'LUT MSE':>10s}  {'Poly MSE':>10s}  "
          f"{'poly/LUT':>9s}  {'ms LUT':>7s}  {'ms poly':>7s}  {'spd×':>5s}  {'uni':>5s}  cov  cv")
    print("  " + "─" * 94)
    for s in sorted(summaries, key=lambda x: x["lut_mse_mean"]):
        ratio = s["ratio_mean"]
        sym = "✓" if ratio > 1.0 else "✗"
        cv = s["lut_mse_std"] / max(s["lut_mse_mean"], 1e-30)
        spd = s["speed_ratio"]
        uni = f"{s['norm_uni']:.2f}" if s["norm_uni"] >= 0 else " N/A"
        print(f"  {_dims_str(s['dims']):18s} {s['K']:>2d} {s['L']:>3d}  "
              f"{s['lut_mse_mean']:>10.3e}  {s['poly_mse_mean']:>10.3e}  "
              f"{ratio:>8.1f}×{sym}  {s['lut_ms_per_ep']:>7.2f}  {s['poly_ms_per_ep']:>7.2f}  "
              f"{spd:>5.2f}×  {uni:>5s}  {coverage_flag(s['density'])}  {cv:.2f}")


def print_speed_summary(sums: List[dict]) -> None:
    print(f"\n{'─'*72}\n  Training speed: ms/epoch — LUT vs matched poly KAN\n{'─'*72}")
    print(f"  {'arch':18s} {'K,L':8s}  {'target':12s}  {'LUT':>8s}  {'Poly':>8s}  {'ratio':>6s}")
    print("  " + "─" * 68)
    for s in sums:
        spd = s["speed_ratio"]
        tag = "LUT faster" if spd > 1.05 else ("poly faster" if spd < 0.95 else "≈ same")
        print(f"  {_dims_str(s['dims']):18s} K={s['K']},L={s['L']:<2d}  "
              f"{s['target']:12s}  {s['lut_ms_per_ep']:>8.2f}  {s['poly_ms_per_ep']:>8.2f}  "
              f"{spd:>5.2f}×  {tag}")


# ─────────────────────────────────────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────────────────────────────────────

PRESETS = {
    "quick": dict(
        mode="stack",
        dims_list=[[1,4,1],[1,8,1],[1,4,4,1],[1,4,4,4,1]],
        kl_list=[(1,4),(1,8),(2,4)],
        targets=["sine","saturating"],
        seeds=[0,1,2], n_train=500, epochs=1000, patience=200,
    ),
    "standard": dict(
        mode="stack",
        dims_list=[[1,4,1],[1,8,1],[1,16,1],[1,4,4,1],[1,8,4,1],[1,4,4,4,1]],
        kl_list=[(1,4),(1,8),(2,4),(2,8)],
        targets=["sine","cusp","saturating"],
        seeds=[0,1,2,3], n_train=500, epochs=1500, patience=300,
    ),
    "single_edge": dict(
        mode="single_edge",
        kl_list=[(4,32),(4,16),(2,16),(2,8),(1,8),(1,16)],
        targets=["sine","cusp","saturating"],
        seeds=[0,1,2], n_train=500, epochs=1500, patience=0,
    ),
    "deep": dict(
        mode="stack",
        dims_list=[[1,4,1],[1,4,4,1],[1,4,4,4,1]],
        kl_list=[(1,4),(1,8),(2,4)],
        targets=["sine","cusp","saturating"],
        seeds=[0,1,2,3,4], n_train=500, epochs=2000, patience=400,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep runner
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    mode, dims_list, kl_list, targets, seeds,
    n_train, epochs, patience, lambda_2, lr,
    recal_every, ema_alpha, min_density, output_path, verbose,
):
    print(f"\n{'━'*68}")
    print(f"  Mode: {mode}  |  Targets: {targets}")
    print(f"  n_train={n_train}  epochs={epochs}  patience={patience}")
    print(f"  λ₂={lambda_2}  lr={lr}  seeds={seeds}")
    print(f"  Comparison: LUT vs poly-KAN, matched params/edge (degree = K×L-1)")
    print(f"{'━'*68}\n")

    all_edge_runs: List[EdgeResult] = []
    all_stack_runs: List[StackResult] = []

    if mode == "single_edge":
        total = len(kl_list) * len(targets) * len(seeds)
        done = 0
        print(f"  Single-edge [1→1]: {total} runs total\n")
        for (K, L), target in itertools.product(kl_list, targets):
            density = n_train / max(K * L, 1)
            if density < min_density:
                continue
            print(f"▶  K={K},L={L} (degree={K*L-1})  {target:12s}  density={density:.1f}")
            seed_runs = []
            for seed in seeds:
                try:
                    r = run_single_edge(K, L, target, seed, n_train, epochs, lambda_2, lr)
                    seed_runs.append(r)
                    all_edge_runs.append(r)
                    done += 1
                    print(f"  [{done}/{total}] seed={seed}  LUT={r.lut_mse:.3e}  "
                          f"poly={r.poly_mse:.3e}  ratio={r.poly_mse/max(r.lut_mse,1e-30):.1f}×  "
                          f"{r.lut_ms_per_epoch:.1f}ms/ep")
                except Exception as e:
                    print(f"  ✗ seed={seed}: {e}")
                    done += 1
            if seed_runs:
                s = summarise_edge(seed_runs)
                cv = s["lut_std"] / max(s["lut_mean"], 1e-30)
                print(f"   → mean LUT={s['lut_mean']:.3e}  poly={s['poly_mean']:.3e}  "
                      f"ratio={s['ratio_mean']:.1f}×  cv={cv:.2f}\n")

        sums_by_target: Dict[str, List[dict]] = {}
        for (K, L), target in itertools.product(kl_list, targets):
            runs = [r for r in all_edge_runs if r.K==K and r.L==L and r.target==target]
            if runs:
                sums_by_target.setdefault(target, []).append(summarise_edge(runs))
        for target, sums in sums_by_target.items():
            print_edge_table(sums, title=f"Single-edge [1→1] — {target}")

    else:
        candidates = [(dims, K, L, target)
                      for dims, (K, L), target in itertools.product(dims_list, kl_list, targets)
                      if dims[0] == 1 and bottleneck_density(dims, K, L, n_train) >= min_density]
        total = len(candidates) * len(seeds)
        print(f"  {len(candidates)} configs × {len(seeds)} seeds = {total} runs")
        print(f"  NOTE: poly comparison only for 3-layer [in→H→out]; deeper = LUT-only\n")

        done = 0
        for dims, K, L, target in candidates:
            density = bottleneck_density(dims, K, L, n_train)
            poly_ok = len(dims) == 3
            print(f"▶  {_dims_str(dims):18s} K={K} L={L}  {target:12s}  "
                  f"density={density:.1f}  {total_lut_bytes(dims,K,L)/1024:.2f}kB  "
                  f"{'LUT+poly' if poly_ok else 'LUT-only'}  [{coverage_flag(density)}]")
            seed_runs = []
            for seed in seeds:
                try:
                    if poly_ok:
                        r = run_stack_vs_poly(
                            dims, K, L, target, seed, n_train, epochs,
                            lambda_2, lr, patience, recal_every, ema_alpha,
                        )
                    else:
                        d = generate_data(target, n_train=n_train, seed=seed*17+3)
                        x_tr, y_tr, x_val, y_val, x_te, y_te = d
                        lut_model = LUTKANStack(dims, K=K, L=L)
                        cfg = StackTrainConfig(
                            lambda_2=lambda_2, lr=lr, epochs=epochs, seed=seed,
                            cheby_init=True, freeze_norms=True, smooth_norms=True,
                            ema_alpha=ema_alpha, recalibrate_every_epochs=recal_every,
                        )
                        t0 = time.perf_counter()
                        rl = train_lut_stack(lut_model, x_tr, y_tr, x_val, y_val, x_te, y_te,
                                             cfg, patience=patience)
                        elapsed = time.perf_counter() - t0
                        r = StackResult(
                            dims=dims, K=K, L=L, target=target, seed=seed,
                            lut_mse=float(rl.mse_test_at_best), poly_mse=float("nan"),
                            lut_ms_per_epoch=elapsed/epochs*1000,
                            poly_ms_per_epoch=float("nan"),
                            lut_best_epoch=rl.best_epoch, poly_best_epoch=-1,
                            norm_uniformity=-1.0, density=density,
                            mem_bytes=total_lut_bytes(dims,K,L),
                            n_lut_params=lut_model.n_lut_params(),
                        )
                    seed_runs.append(r)
                    all_stack_runs.append(r)
                    done += 1
                    if not math.isnan(r.poly_mse):
                        print(f"  [{done}/{total}] seed={seed}  LUT={r.lut_mse:.3e}  "
                              f"poly={r.poly_mse:.3e}  ratio={r.poly_mse/max(r.lut_mse,1e-30):.1f}×  "
                              f"LUT {r.lut_ms_per_epoch:.1f}ms/ep  poly {r.poly_ms_per_epoch:.1f}ms/ep")
                    else:
                        print(f"  [{done}/{total}] seed={seed}  LUT={r.lut_mse:.3e}  "
                              f"{r.lut_ms_per_epoch:.1f}ms/ep  (no poly baseline)")
                except Exception as e:
                    print(f"  ✗ seed={seed}: {e}")
                    done += 1
            if seed_runs:
                s = summarise_stack(seed_runs)
                ratio_s = f"{s['ratio_mean']:.1f}×" if not math.isnan(s['ratio_mean']) else "N/A"
                cv = s['lut_mse_std'] / max(s['lut_mse_mean'], 1e-30)
                print(f"   → LUT={s['lut_mse_mean']:.3e}±{s['lut_mse_std']:.1e}  "
                      f"poly={s['poly_mse_mean']:.3e}  ratio={ratio_s}  "
                      f"speed={s['speed_ratio']:.2f}×  uni={s['norm_uni']:.2f}  cv={cv:.2f}\n")

        # tables
        sums_by_target: Dict[str, List[dict]] = {}
        for dims, K, L, target in candidates:
            runs = [r for r in all_stack_runs
                    if r.dims==dims and r.K==K and r.L==L and r.target==target]
            if runs:
                sums_by_target.setdefault(target, []).append(summarise_stack(runs))

        for target, sums in sums_by_target.items():
            valid = [s for s in sums if not math.isnan(s.get("poly_mse_mean", float("nan")))]
            print_stack_table(valid, title=f"LUT vs poly-KAN [in→H→out], matched params — {target}")

        all_valid = [s for sl in sums_by_target.values() for s in sl
                     if not math.isnan(s.get("poly_mse_mean", float("nan")))]
        if all_valid:
            print_speed_summary(all_valid)

        # recommendations
        print(f"\n{'─'*68}\n  Key observations\n{'─'*68}")
        for target in targets:
            sums = [s for s in sums_by_target.get(target, [])
                    if not math.isnan(s.get("poly_mse_mean", float("nan")))]
            if not sums: continue
            sums.sort(key=lambda s: s["lut_mse_mean"])
            best = sums[0]
            lut_wins = [s for s in sums if s["ratio_mean"] > 1.0]
            print(f"\n  [{target}]")
            print(f"    Best LUT   : {_dims_str(best['dims'])} K={best['K']} L={best['L']}  "
                  f"MSE={best['lut_mse_mean']:.3e}  ratio={best['ratio_mean']:.1f}×")
            if lut_wins:
                bw = max(lut_wins, key=lambda s: s["ratio_mean"])
                print(f"    Best ratio : {_dims_str(bw['dims'])} K={bw['K']} L={bw['L']}  "
                      f"poly/LUT={bw['ratio_mean']:.1f}×")
            fastest = min(sums, key=lambda s: s["lut_ms_per_ep"])
            print(f"    Fastest/ep : {_dims_str(fastest['dims'])} K={fastest['K']} L={fastest['L']}  "
                  f"{fastest['lut_ms_per_ep']:.2f}ms  vs poly {fastest['poly_ms_per_ep']:.2f}ms")

    # save JSON
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        def _ser(r):
            d = r.__dict__.copy()
            for k in list(d.keys()):
                if isinstance(d[k], float) and math.isnan(d[k]):
                    d[k] = None
            return d
        all_runs = all_edge_runs if mode == "single_edge" else all_stack_runs
        out = {
            "config": {"mode": mode, "n_train": n_train, "epochs": epochs,
                       "lambda_2": lambda_2, "lr": lr, "seeds": seeds},
            "runs": [_ser(r) for r in all_runs],
        }
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="LUT-KAN vs matched poly-KAN: accuracy + training speed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=list(PRESETS.keys()), default="quick")
    p.add_argument("--dims", nargs="+", type=lambda s: [int(x) for x in s.split(",")], metavar="IN,H,...,OUT")
    p.add_argument("--kl",   nargs="+", type=lambda s: tuple(int(x) for x in s.split(",")), metavar="K,L")
    p.add_argument("--targets", nargs="+", choices=list(TARGETS.keys()))
    p.add_argument("--seeds",   nargs="+", type=int)
    p.add_argument("--n_train", type=int)
    p.add_argument("--epochs",  type=int)
    p.add_argument("--patience",type=int)
    p.add_argument("--lambda2", type=float, default=1.0)
    p.add_argument("--lr",      type=float, default=1e-2)
    p.add_argument("--ema_alpha",   type=float, default=0.01)
    p.add_argument("--recal_every", type=int,   default=100)
    p.add_argument("--min_density", type=float, default=5.0)
    p.add_argument("--output",  type=Path)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    preset = PRESETS[args.mode]
    out = args.output or (
        Path(__file__).parent.parent / "results" / "sweep_compact"
        / f"{args.mode}_results.json"
    )
    run_sweep(
        mode=preset["mode"],
        dims_list=args.dims    or preset.get("dims_list", [[1,4,1]]),
        kl_list=args.kl        or preset["kl_list"],
        targets=args.targets   or preset["targets"],
        seeds=args.seeds       or preset["seeds"],
        n_train=args.n_train   or preset["n_train"],
        epochs=args.epochs     or preset["epochs"],
        patience=args.patience if args.patience is not None else preset.get("patience",0),
        lambda_2=args.lambda2,
        lr=args.lr,
        recal_every=args.recal_every,
        ema_alpha=args.ema_alpha,
        min_density=args.min_density,
        output_path=out,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
