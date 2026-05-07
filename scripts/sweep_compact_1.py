"""
sweep_compact.py — systematic search for stable compact LUT-KAN architectures.

Sweeps architectures obeying the coverage rule (K×L × edges_per_layer < n_train),
runs multiple seeds, and reports a ranked table of stable configs.

Usage examples
--------------
# Quick mode — preset small grid, 1D targets only:
  python scripts/sweep_compact.py --mode quick

# Standard — full 1D + 2D sweep, more seeds:
  python scripts/sweep_compact.py --mode standard

# Custom dims/K/L:
  python scripts/sweep_compact.py --dims 1,4,1 1,4,4,1 --kl 1,4 2,4 2,8 --targets sine cusp

# Deep architectures only:
  python scripts/sweep_compact.py --mode deep --targets sine saturating

# All options:
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
from typing import List, Optional, Tuple

import numpy as np
import torch

# ── make project importable when run from repo root ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lut_native.targets import generate_data, generate_data_2d, TARGETS
from lut_native.kan_stack import LUTKANStack
from lut_native.training_stack import StackTrainConfig, train_lut_stack
from lut_native.baselines import fit_chebyshev_ls, eval_chebyshev


# ─────────────────────────────────────────────────────────────────────────────
# Coverage utilities
# ─────────────────────────────────────────────────────────────────────────────

def bottleneck_density(dims: List[int], K: int, L: int, n_train: int) -> float:
    """
    Minimum gradient coverage density across all layers.

      density_l = n_train / (in_dim_l * out_dim_l * K * L)

    This is the expected number of training points that touch any
    given LUT cell in the worst layer.  Values < 5 are unstable.
    """
    densities = []
    for i in range(len(dims) - 1):
        edges = dims[i] * dims[i + 1]
        cells = edges * K * L
        densities.append(n_train / max(cells, 1))
    return min(densities) if densities else 0.0


def coverage_flag(density: float) -> str:
    if density >= 20:
        return "✓ good"
    if density >= 8:
        return "~ ok"
    if density >= 3:
        return "! low"
    return "✗ bad"


def total_lut_bytes(dims: List[int], K: int, L: int) -> int:
    """uint8-quantised deployment size (K*L + 4*K bytes per edge)."""
    total = 0
    for i in range(len(dims) - 1):
        n_edges = dims[i] * dims[i + 1]
        total += n_edges * (K * L + 4 * K)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Poly baseline (Chebyshev least-squares, 1D only)
# ─────────────────────────────────────────────────────────────────────────────

def poly_baseline_mse(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    degree: int = 8,
) -> float:
    coeffs = fit_chebyshev_ls(x_train.ravel(), y_train.ravel(), degree=degree)
    y_pred = eval_chebyshev(x_test.ravel(), coeffs)
    return float(np.mean((y_pred - y_test.ravel()) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Single-config runner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    dims: List[int]
    K: int
    L: int
    target: str
    seed: int
    mse: float
    best_epoch: int
    norm_uniformity: float    # mean across all norms; -1 if single-layer
    elapsed_s: float
    density: float
    n_lut_params: int
    mem_bytes: int


def run_one(
    dims: List[int],
    K: int,
    L: int,
    target: str,
    seed: int,
    n_train: int,
    epochs: int,
    lambda_2: float,
    lr: float,
    patience: int,
    recal_every: int,
    ema_alpha: float,
    verbose: bool = False,
) -> RunResult:
    """Train one config on one seed; return RunResult."""

    # ── data ─────────────────────────────────────────────────────────────────
    is_2d = (dims[0] == 2)
    if is_2d:
        n_val  = max(200, n_train // 4)
        n_test = max(200, n_train // 4)
        x_tr, y_tr, x_val, y_val, x_te, y_te = generate_data_2d(
            target, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed * 17 + 3
        )
    else:
        n_val  = max(100, n_train // 4)
        n_test = max(100, n_train // 4)
        x_tr, y_tr, x_val, y_val, x_te, y_te = generate_data(
            target, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed * 17 + 3
        )

    # ── model ─────────────────────────────────────────────────────────────────
    model = LUTKANStack(dims, K=K, L=L, smooth_norms=True)

    cfg = StackTrainConfig(
        lambda_2=lambda_2,
        lr=lr,
        epochs=epochs,
        seed=seed,
        cheby_init=True,
        cheby_scale=1.5,
        cheby_noise=0.05,
        freeze_norms=True,
        smooth_norms=True,
        ema_alpha=ema_alpha,
        recalibrate_every_epochs=recal_every,
        calibrate_at_start=True,
        eval_every_epochs=max(10, epochs // 100),
    )

    t0 = time.perf_counter()
    res = train_lut_stack(
        model,
        x_tr, y_tr, x_val, y_val, x_te, y_te,
        cfg,
        patience=patience,
    )
    elapsed = time.perf_counter() - t0

    # ── norm uniformity (best model weights) ─────────────────────────────────
    model.load_snapshot(res.best_luts, res.best_norms)
    x_t = torch.tensor(
        x_tr if is_2d else x_tr.reshape(-1, dims[0]), dtype=torch.float32
    )
    if x_t.dim() == 1:
        x_t = x_t.unsqueeze(-1)

    uni_scores = []
    z = x_t
    for i, blk in enumerate(model.blocks[:-1]):
        z_raw = blk(z)
        stats = model.norms[i].coverage_stats(z_raw, K)
        uni_scores.append(stats["uniformity"])
        z = model.norms[i](z_raw)
    norm_uni = float(np.mean(uni_scores)) if uni_scores else -1.0

    density = bottleneck_density(dims, K, L, n_train)

    if verbose:
        print(
            f"    seed={seed} mse={res.mse_test_at_best:.4e} "
            f"best_ep={res.best_epoch} uni={norm_uni:.2f} "
            f"t={elapsed:.1f}s"
        )

    return RunResult(
        dims=dims, K=K, L=L, target=target, seed=seed,
        mse=float(res.mse_test_at_best),
        best_epoch=res.best_epoch,
        norm_uniformity=norm_uni,
        elapsed_s=elapsed,
        density=density,
        n_lut_params=model.n_lut_params(),
        mem_bytes=model.memory_bytes_uint8(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate across seeds
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfigSummary:
    dims: List[int]
    K: int
    L: int
    target: str
    mse_mean: float
    mse_std: float
    mse_min: float
    best_epoch_mean: float
    norm_uni_mean: float
    density: float
    mem_bytes: int
    n_lut_params: int
    n_seeds: int
    poly_mse: float           # Chebyshev d=8 baseline; NaN for 2D
    ratio_vs_poly: float      # poly_mse / mse_mean (>1 = LUT wins)


def summarise(
    runs: List[RunResult],
    poly_mse: float,
) -> ConfigSummary:
    mses = [r.mse for r in runs]
    best_eps = [r.best_epoch for r in runs]
    unis = [r.norm_uniformity for r in runs if r.norm_uniformity >= 0]
    r0 = runs[0]
    mse_mean = float(np.mean(mses))
    return ConfigSummary(
        dims=r0.dims, K=r0.K, L=r0.L, target=r0.target,
        mse_mean=mse_mean,
        mse_std=float(np.std(mses)) if len(mses) > 1 else 0.0,
        mse_min=float(np.min(mses)),
        best_epoch_mean=float(np.mean(best_eps)),
        norm_uni_mean=float(np.mean(unis)) if unis else -1.0,
        density=r0.density,
        mem_bytes=r0.mem_bytes,
        n_lut_params=r0.n_lut_params,
        n_seeds=len(runs),
        poly_mse=poly_mse,
        ratio_vs_poly=poly_mse / mse_mean if mse_mean > 0 else float("nan"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Preset sweeps
# ─────────────────────────────────────────────────────────────────────────────

PRESETS = {
    "quick": dict(
        dims_list=[
            [1, 4, 1],
            [1, 8, 1],
            [1, 4, 4, 1],
            [1, 4, 4, 4, 1],
        ],
        kl_list=[(1, 4), (2, 4), (2, 8), (1, 8)],
        targets=["sine", "saturating"],
        seeds=[0, 1, 2],
        n_train=500,
        epochs=1500,
        patience=300,
    ),
    "standard": dict(
        dims_list=[
            [1, 4, 1],
            [1, 8, 1],
            [1, 16, 1],
            [1, 4, 4, 1],
            [1, 8, 4, 1],
            [1, 4, 4, 4, 1],
            [2, 4, 1],
            [2, 8, 1],
            [2, 4, 4, 1],
        ],
        kl_list=[(1, 4), (2, 4), (2, 8), (1, 8), (4, 4)],
        targets=["sine", "cusp", "saturating", "feynman_2d"],
        seeds=[0, 1, 2, 3],
        n_train=500,
        epochs=2000,
        patience=400,
    ),
    "deep": dict(
        dims_list=[
            [1, 4, 1],
            [1, 4, 4, 1],
            [1, 4, 4, 4, 1],
            [1, 4, 4, 4, 4, 1],
            [1, 8, 4, 1],
            [1, 8, 8, 1],
        ],
        kl_list=[(1, 4), (2, 4), (1, 8)],
        targets=["sine", "saturating", "cusp"],
        seeds=[0, 1, 2, 3, 4],
        n_train=500,
        epochs=3000,
        patience=500,
    ),
    "ultra_compact": dict(
        dims_list=[
            [1, 2, 1],
            [1, 4, 1],
            [1, 2, 2, 1],
            [1, 4, 4, 1],
        ],
        kl_list=[(1, 2), (1, 4), (2, 2), (2, 4)],
        targets=["sine", "cusp", "saturating"],
        seeds=[0, 1, 2, 3, 4],
        n_train=200,    # tight budget
        epochs=2000,
        patience=400,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dims_str(dims: List[int]) -> str:
    return "[" + "→".join(str(d) for d in dims) + "]"


def print_table(summaries: List[ConfigSummary], title: str = "") -> None:
    if not summaries:
        print("  (no results)")
        return

    if title:
        print(f"\n{'─'*72}")
        print(f"  {title}")
        print(f"{'─'*72}")

    hdr = (
        f"{'arch':<18} {'K':>2} {'L':>3}  "
        f"{'MSE mean':>10} {'±std':>8}  "
        f"{'poly/LUT':>8}  "
        f"{'uni':>5}  "
        f"{'ep':>5}  "
        f"{'den':>5}  "
        f"{'kB':>5}  "
        f"cov"
    )
    print(hdr)
    print("─" * 80)

    for s in summaries:
        ratio_str = f"{s.ratio_vs_poly:.1f}×" if not math.isnan(s.ratio_vs_poly) else "  N/A"
        uni_str   = f"{s.norm_uni_mean:.2f}" if s.norm_uni_mean >= 0 else " N/A"
        poly_str  = ratio_str.rjust(8)
        kB        = s.mem_bytes / 1024
        print(
            f"{_dims_str(s.dims):<18} {s.K:>2} {s.L:>3}  "
            f"{s.mse_mean:>10.4e} {s.mse_std:>8.1e}  "
            f"{poly_str}  "
            f"{uni_str:>5}  "
            f"{s.best_epoch_mean:>5.0f}  "
            f"{s.density:>5.1f}  "
            f"{kB:>5.2f}  "
            f"{coverage_flag(s.density)}"
        )


def print_stability_warning(summaries: List[ConfigSummary]) -> None:
    unstable = [s for s in summaries if s.mse_std > s.mse_mean * 0.5 and s.n_seeds >= 3]
    if unstable:
        print("\n⚠  High-variance configs (std > 50% of mean):")
        for s in unstable:
            print(f"   {_dims_str(s.dims)} K={s.K} L={s.L} target={s.target}  "
                  f"cv={s.mse_std/s.mse_mean:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    dims_list: List[List[int]],
    kl_list: List[Tuple[int, int]],
    targets: List[str],
    seeds: List[int],
    n_train: int,
    epochs: int,
    patience: int,
    lambda_2: float,
    lr: float,
    recal_every: int,
    ema_alpha: float,
    min_density: float,
    output_path: Optional[Path],
    verbose: bool,
) -> List[ConfigSummary]:

    # ── build candidate configs ───────────────────────────────────────────────
    candidates = []
    for dims, (K, L), target in itertools.product(dims_list, kl_list, targets):
        # skip 2D targets for 1D architectures and vice versa
        in_dim = dims[0]
        is_2d_target = target == "feynman_2d"
        if is_2d_target and in_dim != 2:
            continue
        if not is_2d_target and in_dim == 2:
            continue

        density = bottleneck_density(dims, K, L, n_train)
        if density < min_density:
            if verbose:
                print(
                    f"  skip {_dims_str(dims)} K={K} L={L} target={target}  "
                    f"density={density:.1f} < {min_density}"
                )
            continue
        candidates.append((dims, K, L, target))

    total = len(candidates) * len(seeds)
    print(f"\n{'━'*60}")
    print(f"  Sweep: {len(candidates)} configs × {len(seeds)} seeds = {total} runs")
    print(f"  Targets: {targets}")
    print(f"  Coverage threshold: density ≥ {min_density:.1f} pts/cell")
    print(f"  n_train={n_train}  epochs={epochs}  λ₂={lambda_2}  lr={lr}")
    print(f"  EMA α={ema_alpha}  recal_every={recal_every}")
    print(f"{'━'*60}\n")

    # ── poly baselines (1D targets, computed once per target) ─────────────────
    poly_baselines: dict = {}
    for target in targets:
        if target == "feynman_2d":
            poly_baselines[target] = float("nan")
            continue
        d = generate_data(target, n_train=n_train, seed=99)
        x_tr, y_tr, _, _, x_te, y_te = d
        pmse = poly_baseline_mse(x_tr, y_tr, x_te, y_te, degree=8)
        poly_baselines[target] = pmse
        print(f"  Poly d=8 baseline [{target}]: {pmse:.4e}")
    print()

    # ── main loop ─────────────────────────────────────────────────────────────
    all_runs: List[RunResult] = []
    done = 0
    for dims, K, L, target in candidates:
        density = bottleneck_density(dims, K, L, n_train)
        mem_kb  = total_lut_bytes(dims, K, L) / 1024
        print(
            f"▶  {_dims_str(dims):18s} K={K} L={L}  target={target:12s}  "
            f"density={density:5.1f}  {mem_kb:.2f}kB  "
            f"[{coverage_flag(density)}]"
        )
        seed_runs = []
        for seed in seeds:
            try:
                r = run_one(
                    dims=dims, K=K, L=L, target=target, seed=seed,
                    n_train=n_train, epochs=epochs,
                    lambda_2=lambda_2, lr=lr, patience=patience,
                    recal_every=recal_every, ema_alpha=ema_alpha,
                    verbose=verbose,
                )
                seed_runs.append(r)
                all_runs.append(r)
                done += 1
                if verbose:
                    pass  # already printed inside run_one
                else:
                    status = f"  [{done}/{total}] seed={seed} mse={r.mse:.3e} ep={r.best_epoch}"
                    print(status, flush=True)
            except Exception as exc:
                print(f"  ✗ seed={seed} ERROR: {exc}")
                done += 1

        if seed_runs:
            s = summarise(seed_runs, poly_baselines[target])
            mses = [r.mse for r in seed_runs]
            uni  = f"{s.norm_uni_mean:.2f}" if s.norm_uni_mean >= 0 else "N/A"
            ratio = f"{s.ratio_vs_poly:.1f}×" if not math.isnan(s.ratio_vs_poly) else "N/A"
            cv = s.mse_std / s.mse_mean if s.mse_mean > 0 else 0
            print(
                f"   → mean={s.mse_mean:.3e} ± {s.mse_std:.1e}  "
                f"poly/LUT={ratio}  uni={uni}  "
                f"cv={cv:.2f}  ep≈{s.best_epoch_mean:.0f}"
            )
        print()

    # ── aggregate ─────────────────────────────────────────────────────────────
    summaries: List[ConfigSummary] = []
    for (dims, K, L, target), group in itertools.groupby(
        sorted(all_runs, key=lambda r: (r.target, str(r.dims), r.K, r.L, r.seed)),
        key=lambda r: (tuple(r.dims), r.K, r.L, r.target),
    ):
        runs = list(group)
        if runs:
            summaries.append(summarise(runs, poly_baselines[target]))

    # ── print per-target tables ───────────────────────────────────────────────
    for target in targets:
        tsums = [s for s in summaries if s.target == target]
        tsums.sort(key=lambda s: s.mse_mean)
        print_table(tsums, title=f"Results — {target}")

    print_stability_warning(summaries)

    # ── Pareto front: best MSE vs memory ─────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  Pareto front: best config per memory budget (per target)")
    print(f"{'─'*72}")
    for target in targets:
        tsums = [s for s in summaries if s.target == target]
        if not tsums:
            continue
        # bin by memory: <0.25kB, 0.25-0.5, 0.5-1, 1-2, 2-4, 4+
        bins = [256, 512, 1024, 2048, 4096, 99999999]
        bin_labels = ["<256B", "<512B", "<1kB", "<2kB", "<4kB", "4kB+"]
        prev_best = None
        print(f"\n  {target}:")
        for limit, label in zip(bins, bin_labels):
            candidates_in_bin = [s for s in tsums if s.mem_bytes <= limit]
            if not candidates_in_bin:
                continue
            best = min(candidates_in_bin, key=lambda s: s.mse_mean)
            if prev_best is not None and best.mse_mean >= prev_best.mse_mean:
                continue  # dominated by smaller model
            uni = f"{best.norm_uni_mean:.2f}" if best.norm_uni_mean >= 0 else "N/A"
            ratio = f"{best.ratio_vs_poly:.1f}×" if not math.isnan(best.ratio_vs_poly) else "N/A"
            print(
                f"    {label:6s}  {_dims_str(best.dims):18s} K={best.K} L={best.L}  "
                f"MSE={best.mse_mean:.3e}  poly/LUT={ratio}  uni={uni}  "
                f"cv={best.mse_std/best.mse_mean:.2f}"
            )
            prev_best = best

    # ── recommendations ───────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  Key observations")
    print(f"{'─'*72}")

    for target in targets:
        tsums = [s for s in summaries if s.target == target]
        if not tsums:
            continue
        tsums.sort(key=lambda s: s.mse_mean)
        best = tsums[0]
        # most stable = lowest CV
        stable = min(tsums, key=lambda s: s.mse_std / max(s.mse_mean, 1e-12))
        lut_wins = [s for s in tsums if s.ratio_vs_poly > 1.0]
        print(f"\n  [{target}]")
        print(
            f"    Best accuracy : {_dims_str(best.dims)} K={best.K} L={best.L}  "
            f"MSE={best.mse_mean:.3e}  {coverage_flag(best.density)}"
        )
        if stable is not best:
            cv = stable.mse_std / max(stable.mse_mean, 1e-12)
            print(
                f"    Most stable   : {_dims_str(stable.dims)} K={stable.K} L={stable.L}  "
                f"cv={cv:.2f}  MSE={stable.mse_mean:.3e}"
            )
        if lut_wins:
            best_win = max(lut_wins, key=lambda s: s.ratio_vs_poly)
            print(
                f"    Beats poly    : {_dims_str(best_win.dims)} K={best_win.K} L={best_win.L}  "
                f"poly/LUT={best_win.ratio_vs_poly:.1f}×"
            )

    # ── save JSON ─────────────────────────────────────────────────────────────
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "config": {
                "n_train": n_train, "epochs": epochs, "lambda_2": lambda_2,
                "lr": lr, "seeds": seeds, "ema_alpha": ema_alpha,
                "recal_every": recal_every, "patience": patience,
            },
            "poly_baselines": {k: v for k, v in poly_baselines.items()
                               if not (isinstance(v, float) and math.isnan(v))},
            "summaries": [
                {
                    "dims": s.dims, "K": s.K, "L": s.L, "target": s.target,
                    "mse_mean": s.mse_mean, "mse_std": s.mse_std, "mse_min": s.mse_min,
                    "best_epoch_mean": s.best_epoch_mean,
                    "norm_uni_mean": s.norm_uni_mean,
                    "density": s.density,
                    "mem_bytes": s.mem_bytes,
                    "n_lut_params": s.n_lut_params,
                    "n_seeds": s.n_seeds,
                    "poly_mse": s.poly_mse if not math.isnan(s.poly_mse) else None,
                    "ratio_vs_poly": (s.ratio_vs_poly
                                      if not math.isnan(s.ratio_vs_poly) else None),
                }
                for s in summaries
            ],
            "all_runs": [
                {
                    "dims": r.dims, "K": r.K, "L": r.L, "target": r.target,
                    "seed": r.seed, "mse": r.mse, "best_epoch": r.best_epoch,
                    "norm_uniformity": r.norm_uniformity,
                    "elapsed_s": round(r.elapsed_s, 2),
                    "density": r.density,
                }
                for r in all_runs
            ],
        }
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved → {output_path}")

    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dims(s: str) -> List[int]:
    return [int(x) for x in s.split(",")]


def _parse_kl(s: str) -> Tuple[int, int]:
    parts = s.split(",")
    return int(parts[0]), int(parts[1])


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sweep compact LUT-KAN architectures following the coverage rule.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--mode",
        choices=list(PRESETS.keys()),
        default="quick",
        help="Preset sweep (default: quick). Overridden by explicit --dims/--kl.",
    )
    p.add_argument(
        "--dims",
        nargs="+",
        type=_parse_dims,
        metavar="IN,H1,...,OUT",
        help="Explicit architecture list, e.g. --dims 1,4,1 1,4,4,1",
    )
    p.add_argument(
        "--kl",
        nargs="+",
        type=_parse_kl,
        metavar="K,L",
        help="Explicit K,L pairs, e.g. --kl 1,4 2,4 2,8",
    )
    p.add_argument(
        "--targets",
        nargs="+",
        default=None,
        choices=list(TARGETS.keys()) + ["feynman_2d"],
        metavar="TARGET",
        help="Target functions to test (default: from preset)",
    )
    p.add_argument("--seeds",  nargs="+", type=int, default=None, help="Seeds (default: from preset)")
    p.add_argument("--n_train", type=int, default=None, help="Training set size")
    p.add_argument("--epochs",  type=int, default=None, help="Max training epochs")
    p.add_argument("--patience", type=int, default=None, help="Early stopping patience (0=off)")
    p.add_argument("--lambda2", type=float, default=1.0, help="λ₂ regularisation (default 1.0)")
    p.add_argument("--lr",      type=float, default=1e-2, help="Adam learning rate (default 1e-2)")
    p.add_argument("--ema_alpha", type=float, default=0.01,
                   help="EMA norm tracking alpha (default 0.01, 0=off)")
    p.add_argument("--recal_every", type=int, default=100,
                   help="Recalibrate norms every N epochs (0=off, default 100)")
    p.add_argument("--min_density", type=float, default=5.0,
                   help="Skip configs with coverage density < this (default 5.0)")
    p.add_argument("--output",  type=Path, default=None,
                   help="Path to save JSON results (default: results/sweep_compact/results.json)")
    p.add_argument("--verbose", action="store_true", help="Print per-seed details")

    args = p.parse_args()

    # ── resolve preset vs explicit ────────────────────────────────────────────
    preset = PRESETS[args.mode]
    dims_list  = args.dims    if args.dims    else preset["dims_list"]
    kl_list    = args.kl      if args.kl      else preset["kl_list"]
    targets    = args.targets if args.targets else preset["targets"]
    seeds      = args.seeds   if args.seeds   else preset["seeds"]
    n_train    = args.n_train if args.n_train else preset["n_train"]
    epochs     = args.epochs  if args.epochs  else preset["epochs"]
    patience   = args.patience if args.patience is not None else preset.get("patience", 0)

    output_path = args.output or (
        Path(__file__).parent.parent / "results" / "sweep_compact"
        / f"{args.mode}_results.json"
    )

    run_sweep(
        dims_list=dims_list,
        kl_list=kl_list,
        targets=targets,
        seeds=seeds,
        n_train=n_train,
        epochs=epochs,
        patience=patience,
        lambda_2=args.lambda2,
        lr=args.lr,
        recal_every=args.recal_every,
        ema_alpha=args.ema_alpha,
        min_density=args.min_density,
        output_path=output_path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
