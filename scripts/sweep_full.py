"""
sweep_full.py — comprehensive LUT-KAN vs poly-KAN benchmark.

Fixes vs sweep_compact.py:
  1. Proper timing: warmup run + median of N repeats, not single elapsed time.
  2. poly gets the same lambda_2 regularisation as LUT.
  3. PolyKANStack: a multi-layer poly KAN matching LUTKANStack dims exactly —
     so 3+ layer comparisons are honest.
  4. All 5 targets (sine, cusp, saturating, piecewise_smooth, local_sharp) + feynman_2d.
  5. Coverage density enforced per-layer (not total).
  6. Time-to-threshold metric: epochs × ms/ep until each model hits poly's final MSE.

Usage
-----
  # recommended first run (~30 min):
  python scripts/sweep_full.py --mode standard

  # quick sanity check (~5 min):
  python scripts/sweep_full.py --mode quick

  # deep architecture focus:
  python scripts/sweep_full.py --mode deep

  # single-edge H1 regime (original project baseline):
  python scripts/sweep_full.py --mode single_edge

  # custom:
  python scripts/sweep_full.py \\
      --dims 1,4,1 1,4,4,1 1,4,4,4,1 \\
      --kl 2,8 4,8 8,4 \\
      --targets cusp saturating sine \\
      --seeds 0 1 2 3 4

  python scripts/sweep_full.py --help
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lut_native.targets import generate_data, generate_data_2d, TARGETS
from lut_native.kan_stack import LUTKANStack
from lut_native.training_stack import StackTrainConfig, train_lut_stack
from lut_native.poly_kan2 import PolyKAN2Layer, _cheb_basis_torch  # type: ignore[attr-defined]
from lut_native.baselines import fit_chebyshev_ls, eval_chebyshev, sample_polynomial_to_lut
from lut_native.training import TrainConfig, train_lut_edge


# ─────────────────────────────────────────────────────────────────────────────
# PolyKANStack: multi-layer poly KAN matching LUTKANStack dims
# ─────────────────────────────────────────────────────────────────────────────

class PolyKANStack(nn.Module):
    """
    N-layer polynomial KAN mirroring LUTKANStack.

    Architecture is identical to LUTKANStack except activations are
    Chebyshev polynomial sums instead of LUT lookups.
    Inter-layer nonlinearity: tanh (same as LUTKANStack default norm).
    Parameter count per edge = degree + 1 = K*L.
    """

    def __init__(self, dims: List[int], degree: int):
        super().__init__()
        self.dims = list(dims)
        self.degree = int(degree)
        n_blocks = len(dims) - 1
        # Each block: weight tensor (in_d, out_d, degree+1)
        self.blocks = nn.ParameterList()
        scale = 1.0 / float(degree + 1)
        for i in range(n_blocks):
            in_d, out_d = dims[i], dims[i + 1]
            self.blocks.append(
                nn.Parameter(torch.randn(in_d, out_d, degree + 1) * scale)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        z = x.float()
        for i, w in enumerate(self.blocks):
            zc = torch.clamp(z, -1.0, 1.0)
            T = _cheb_basis_torch(zc, self.degree)   # (N, in_d, degree+1)
            z = torch.einsum("nid,iod->no", T, w)    # (N, out_d)
            if i < len(self.blocks) - 1:
                z = torch.tanh(z)
        return z

    def total_params(self) -> int:
        n = 0
        for i in range(len(self.dims) - 1):
            n += self.dims[i] * self.dims[i + 1] * (self.degree + 1)
        return n

    def memory_bytes_float32(self) -> int:
        return self.total_params() * 4


def train_poly_stack(
    model: PolyKANStack,
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    lr: float = 1e-2,
    epochs: int = 1500,
    batch_size: int = 64,
    seed: int = 0,
    lambda_2: float = 1.0,
    patience: int = 0,
    eval_every: int = 20,
) -> Dict:
    """Train PolyKANStack with val-best + early stopping."""
    torch.manual_seed(seed)

    def _t(a, width):
        a = np.asarray(a, dtype=np.float32)
        return a.reshape(-1, 1) if (a.ndim == 1 and width == 1) else (
            a.reshape(-1, width) if a.ndim == 1 else a)

    in_d, out_d = model.dims[0], model.dims[-1]
    xt = torch.from_numpy(_t(x_train, in_d))
    yt = torch.from_numpy(_t(y_train, out_d))
    xv = torch.from_numpy(_t(x_val,   in_d))
    yv = torch.from_numpy(_t(y_val,   out_d))
    xe = torch.from_numpy(_t(x_test,  in_d))
    ye = torch.from_numpy(_t(y_test,  out_d))
    N = xt.shape[0]

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    def _reg():
        s = torch.tensor(0.0)
        for w in model.blocks:
            s = s + (w ** 2).sum()
        return s

    with torch.no_grad():
        best_val = ((model(xv) - yv) ** 2).mean().item()
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    no_improve = 0
    val_trace: List[float] = [best_val]

    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            optim.zero_grad()
            loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
            if lambda_2 > 0:
                loss = loss + lambda_2 * _reg() / N
            loss.backward()
            optim.step()

        if (ep + 1) % eval_every == 0 or (ep + 1) == epochs:
            with torch.no_grad():
                v = ((model(xv) - yv) ** 2).mean().item()
            val_trace.append(v)
            if v < best_val:
                best_val = v
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
                best_epoch = ep + 1
                no_improve = 0
            else:
                no_improve += 1
            if patience > 0 and no_improve * eval_every >= patience:
                break

    model.load_state_dict(best_state)
    with torch.no_grad():
        mse_tr = ((model(xt) - yt) ** 2).mean().item()
        mse_v  = ((model(xv) - yv) ** 2).mean().item()
        mse_te = ((model(xe) - ye) ** 2).mean().item()

    return {
        "mse_train_at_best": float(mse_tr),
        "mse_val_at_best": float(mse_v),
        "mse_test_at_best": float(mse_te),
        "best_epoch": int(best_epoch),
        "val_trace": val_trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Timing: warmup + median of N repeats
# ─────────────────────────────────────────────────────────────────────────────

def measure_ms_per_epoch(
    dims: List[int],
    K: int,
    L: int,
    target: str,
    n_train: int,
    lr: float,
    lambda_2: float,
    timing_epochs: int = 60,
    n_repeats: int = 5,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Returns (lut_ms_per_epoch, poly_ms_per_epoch), both medians over n_repeats.
    First repeat is warmup and excluded from the median.
    """
    is_2d = dims[0] == 2
    if is_2d:
        d = generate_data_2d(target, n_train=n_train, n_val=100, n_test=100, seed=seed)
    else:
        d = generate_data(target, n_train=n_train, n_val=100, n_test=100, seed=seed)
    x_tr, y_tr, x_val, y_val, x_te, y_te = d

    degree = K * L - 1

    def _run_lut():
        m = LUTKANStack(dims, K=K, L=L)
        cfg = StackTrainConfig(
            epochs=timing_epochs, seed=seed, lambda_2=lambda_2, lr=lr,
            eval_every_epochs=timing_epochs + 1,  # no eval overhead
            calibrate_at_start=True, cheby_init=True,
            freeze_norms=True, smooth_norms=True,
        )
        t0 = time.perf_counter()
        train_lut_stack(m, x_tr, y_tr, x_val, y_val, x_te, y_te, cfg)
        return (time.perf_counter() - t0) / timing_epochs * 1000

    def _run_poly():
        m = PolyKANStack(dims, degree=degree)
        t0 = time.perf_counter()
        train_poly_stack(m, x_tr, y_tr, x_val, y_val, x_te, y_te,
                         lr=lr, epochs=timing_epochs, lambda_2=lambda_2, seed=seed,
                         eval_every=timing_epochs + 1)
        return (time.perf_counter() - t0) / timing_epochs * 1000

    lut_times, poly_times = [], []
    for i in range(n_repeats + 1):  # +1 for warmup
        lut_times.append(_run_lut())
        poly_times.append(_run_poly())

    # Drop warmup (index 0)
    lut_ms  = float(np.median(lut_times[1:]))
    poly_ms = float(np.median(poly_times[1:]))
    return lut_ms, poly_ms


# ─────────────────────────────────────────────────────────────────────────────
# Coverage helpers
# ─────────────────────────────────────────────────────────────────────────────

def bottleneck_density(dims: List[int], K: int, L: int, n_train: int) -> float:
    densities = []
    for i in range(len(dims) - 1):
        cells = dims[i] * dims[i + 1] * K * L
        densities.append(n_train / max(cells, 1))
    return min(densities) if densities else 0.0


def coverage_flag(d: float) -> str:
    if d >= 20: return "✓"
    if d >= 8:  return "~"
    if d >= 3:  return "!"
    return "✗"


def total_lut_bytes(dims: List[int], K: int, L: int) -> int:
    total = 0
    for i in range(len(dims) - 1):
        n_edges = dims[i] * dims[i + 1]
        total += n_edges * (K * L + 4 * K)
    return total


def _dims_str(dims: List[int]) -> str:
    return "[" + "→".join(str(d) for d in dims) + "]"


def _time_to_threshold(trace: List[float], threshold: float) -> Optional[int]:
    for i, v in enumerate(trace):
        if v <= threshold:
            return i
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(target: str, dims: List[int], n_train: int, seed: int):
    is_2d = dims[0] == 2
    n_val  = max(150, n_train // 4)
    n_test = max(150, n_train // 4)
    if is_2d:
        return generate_data_2d(target, n_train=n_train, n_val=n_val, n_test=n_test,
                                seed=seed * 17 + 3)
    else:
        return generate_data(target, n_train=n_train, n_val=n_val, n_test=n_test,
                             seed=seed * 17 + 3)


# ─────────────────────────────────────────────────────────────────────────────
# Run functions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StackRun:
    dims: List[int]
    K: int
    L: int
    target: str
    seed: int
    # accuracy
    lut_mse: float
    poly_mse: float
    lut_best_epoch: int
    poly_best_epoch: int
    # speed (None when not measured)
    lut_ms_per_epoch: Optional[float]
    poly_ms_per_epoch: Optional[float]
    # derived quality
    norm_uniformity: float   # mean across norms; -1 if none
    density: float
    mem_lut_bytes: int
    mem_poly_bytes: int
    n_lut_params: int
    n_poly_params: int
    # time-to-threshold (epochs for LUT to reach poly's val MSE)
    lut_epochs_to_poly_mse: Optional[int]


@dataclass
class EdgeRun:
    K: int
    L: int
    target: str
    seed: int
    lut_mse: float
    poly_ls_mse: float   # closed-form LS (no gradient) — original H1 regime
    poly_gd_mse: float   # gradient-trained poly, same budget — fair comparison
    lut_best_epoch: int
    lut_ms_per_epoch: Optional[float]
    poly_gd_ms_per_epoch: Optional[float]
    density: float
    mem_bytes: int


def run_stack(
    dims: List[int], K: int, L: int, target: str, seed: int,
    n_train: int, epochs: int, lr: float, lambda_2: float,
    patience: int, recal_every: int, ema_alpha: float,
    measure_timing: bool,
    timing_repeats: int,
) -> StackRun:

    d = load_data(target, dims, n_train, seed)
    x_tr, y_tr, x_val, y_val, x_te, y_te = d
    degree = K * L - 1

    # ── LUT ──────────────────────────────────────────────────────────────────
    lut_model = LUTKANStack(dims, K=K, L=L)
    cfg = StackTrainConfig(
        lambda_2=lambda_2, lr=lr, epochs=epochs, seed=seed,
        cheby_init=True, cheby_scale=1.5, cheby_noise=0.05,
        freeze_norms=True, smooth_norms=True,
        ema_alpha=ema_alpha, recalibrate_every_epochs=recal_every,
        calibrate_at_start=True,
        eval_every_epochs=max(10, epochs // 100),
    )
    rl = train_lut_stack(lut_model, x_tr, y_tr, x_val, y_val, x_te, y_te,
                         cfg, patience=patience)

    # ── Poly ─────────────────────────────────────────────────────────────────
    poly_model = PolyKANStack(dims, degree=degree)
    rp = train_poly_stack(
        poly_model, x_tr, y_tr, x_val, y_val, x_te, y_te,
        lr=lr, epochs=epochs, lambda_2=lambda_2, seed=seed,
        patience=patience, eval_every=max(10, epochs // 100),
    )

    # ── norm uniformity ───────────────────────────────────────────────────────
    lut_model.load_snapshot(rl.best_luts, rl.best_norms)
    x_t = torch.tensor(
        x_tr.reshape(-1, dims[0]) if x_tr.ndim == 1 else x_tr,
        dtype=torch.float32
    )
    uni_scores = []
    z = x_t
    for i, blk in enumerate(lut_model.blocks[:-1]):
        z_raw = blk(z)
        uni_scores.append(lut_model.norms[i].coverage_stats(z_raw, K)["uniformity"])
        z = lut_model.norms[i](z_raw)
    norm_uni = float(np.mean(uni_scores)) if uni_scores else -1.0

    # ── time-to-threshold ─────────────────────────────────────────────────────
    poly_target_val = rp["mse_val_at_best"]
    ep_to_target = _time_to_threshold(rl.trace.get("mse_val", []) if isinstance(rl.trace, dict)
                                      else [], poly_target_val)

    # ── timing ────────────────────────────────────────────────────────────────
    lut_ms, poly_ms = None, None
    if measure_timing:
        lut_ms, poly_ms = measure_ms_per_epoch(
            dims, K, L, target, n_train, lr, lambda_2,
            timing_epochs=60, n_repeats=timing_repeats, seed=seed,
        )

    density = bottleneck_density(dims, K, L, n_train)

    return StackRun(
        dims=dims, K=K, L=L, target=target, seed=seed,
        lut_mse=float(rl.mse_test_at_best),
        poly_mse=float(rp["mse_test_at_best"]),
        lut_best_epoch=rl.best_epoch,
        poly_best_epoch=rp["best_epoch"],
        lut_ms_per_epoch=lut_ms,
        poly_ms_per_epoch=poly_ms,
        norm_uniformity=norm_uni,
        density=density,
        mem_lut_bytes=total_lut_bytes(dims, K, L),
        mem_poly_bytes=poly_model.memory_bytes_float32(),
        n_lut_params=lut_model.n_lut_params(),
        n_poly_params=poly_model.total_params(),
        lut_epochs_to_poly_mse=ep_to_target,
    )


def run_edge(
    K: int, L: int, target: str, seed: int,
    n_train: int, epochs: int, lr: float, lambda_2: float,
    measure_timing: bool, timing_repeats: int,
) -> EdgeRun:
    """Single-edge [1→1]: LUT vs LS-poly vs gradient-trained poly."""
    d = generate_data(target, n_train=n_train,
                      n_val=max(100, n_train // 4), n_test=max(100, n_train // 4),
                      seed=seed * 17 + 3)
    x_tr, y_tr, x_val, y_val, x_te, y_te = d
    degree = K * L - 1

    # Chebyshev LS (original H1 regime — closed-form, no gradient)
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=degree)
    lut_init = sample_polynomial_to_lut(coeffs, K=K, L=L)
    poly_ls_mse = float(np.mean((eval_chebyshev(x_te, coeffs) - y_te) ** 2))

    # LUT with poly-init + gradient finetune
    cfg_e = TrainConfig(epochs=epochs, seed=seed, lambda_2=lambda_2, lr=lr)
    rl = train_lut_edge(lut_init, x_tr, y_tr, x_val, y_val, x_te, y_te, -1.0, 1.0, cfg_e)

    # Gradient-trained single-edge poly (fair comparison)
    poly_model = PolyKANStack([1, 1], degree=degree)
    rp = train_poly_stack(
        poly_model, x_tr, y_tr, x_val, y_val, x_te, y_te,
        lr=lr, epochs=epochs, lambda_2=lambda_2, seed=seed,
        patience=0, eval_every=max(10, epochs // 100),
    )

    # timing
    lut_ms, poly_ms = None, None
    if measure_timing:
        lut_ms, poly_ms = measure_ms_per_epoch(
            [1, 1], K, L, target, n_train, lr, lambda_2,
            timing_epochs=60, n_repeats=timing_repeats, seed=seed,
        )

    return EdgeRun(
        K=K, L=L, target=target, seed=seed,
        lut_mse=float(rl.mse_test_at_best),
        poly_ls_mse=poly_ls_mse,
        poly_gd_mse=float(rp["mse_test_at_best"]),
        lut_best_epoch=rl.best_epoch,
        lut_ms_per_epoch=lut_ms,
        poly_gd_ms_per_epoch=poly_ms,
        density=n_train / max(K * L, 1),
        mem_bytes=K * L + 4 * K,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _agg(vals: List[float]) -> Tuple[float, float]:
    return float(np.mean(vals)), float(np.std(vals))


def aggregate_stack(runs: List[StackRun]) -> dict:
    r0 = runs[0]
    luts   = [r.lut_mse  for r in runs]
    polys  = [r.poly_mse for r in runs]
    lut_m,  lut_s  = _agg(luts)
    poly_m, poly_s = _agg(polys)
    ratio  = poly_m / max(lut_m, 1e-30)
    lms    = [r.lut_ms_per_epoch  for r in runs if r.lut_ms_per_epoch  is not None]
    pms    = [r.poly_ms_per_epoch for r in runs if r.poly_ms_per_epoch is not None]
    unis   = [r.norm_uniformity   for r in runs if r.norm_uniformity   >= 0]
    etp    = [r.lut_epochs_to_poly_mse for r in runs if r.lut_epochs_to_poly_mse is not None]
    return dict(
        dims=r0.dims, K=r0.K, L=r0.L, target=r0.target, n_seeds=len(runs),
        lut_mse_mean=lut_m, lut_mse_std=lut_s,
        poly_mse_mean=poly_m, poly_mse_std=poly_s,
        ratio=ratio,
        cv=lut_s / max(lut_m, 1e-30),
        lut_ms=float(np.median(lms))  if lms  else None,
        poly_ms=float(np.median(pms)) if pms  else None,
        speed_ratio=float(np.median(pms)) / float(np.median(lms)) if lms and pms else None,
        norm_uni=float(np.mean(unis)) if unis else None,
        lut_best_ep=float(np.mean([r.lut_best_epoch  for r in runs])),
        poly_best_ep=float(np.mean([r.poly_best_epoch for r in runs])),
        ep_to_poly_mse=float(np.mean(etp)) if etp else None,
        density=r0.density,
        mem_lut_bytes=r0.mem_lut_bytes,
        mem_poly_bytes=r0.mem_poly_bytes,
        n_lut_params=r0.n_lut_params,
        n_poly_params=r0.n_poly_params,
    )


def aggregate_edge(runs: List[EdgeRun]) -> dict:
    r0 = runs[0]
    luts    = [r.lut_mse      for r in runs]
    ls_mse  = [r.poly_ls_mse  for r in runs]
    gd_mse  = [r.poly_gd_mse  for r in runs]
    lm, ls  = _agg(luts)
    plm, _  = _agg(ls_mse)
    pgm, _  = _agg(gd_mse)
    lms_t   = [r.lut_ms_per_epoch       for r in runs if r.lut_ms_per_epoch       is not None]
    pms_t   = [r.poly_gd_ms_per_epoch   for r in runs if r.poly_gd_ms_per_epoch   is not None]
    return dict(
        K=r0.K, L=r0.L, target=r0.target, n_seeds=len(runs),
        lut_mse_mean=lm, lut_mse_std=ls,
        poly_ls_mse=plm, poly_gd_mse=pgm,
        ratio_vs_ls=plm / max(lm, 1e-30),
        ratio_vs_gd=pgm / max(lm, 1e-30),
        cv=ls / max(lm, 1e-30),
        lut_ms=float(np.median(lms_t)) if lms_t else None,
        poly_gd_ms=float(np.median(pms_t)) if pms_t else None,
        speed_ratio=float(np.median(pms_t)) / float(np.median(lms_t)) if lms_t and pms_t else None,
        density=r0.density, mem_bytes=r0.mem_bytes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

W = 110

def _hdr(title: str):
    print(f"\n{'─'*W}")
    print(f"  {title}")
    print(f"{'─'*W}")


def print_stack_table(sums: List[dict], title: str = ""):
    if not sums: return
    sums = sorted(sums, key=lambda s: s["lut_mse_mean"])
    if title: _hdr(title)
    print(f"  {'arch':18s} {'K':>2s} {'L':>3s}  "
          f"{'LUT MSE':>10s} {'±':>7s}  {'Poly MSE':>10s}  "
          f"{'ratio':>7s}  {'ms LUT':>7s}  {'ms poly':>7s}  {'spd':>5s}  "
          f"{'uni':>5s}  {'cv':>5s}  {'den':>5s}  {'kB':>5s}")
    print("  " + "─" * (W - 2))
    for s in sums:
        ratio_s = f"{s['ratio']:.0f}×" if s['ratio'] >= 10 else f"{s['ratio']:.1f}×"
        lms_s   = f"{s['lut_ms']:.1f}"  if s["lut_ms"]  else "  —  "
        pms_s   = f"{s['poly_ms']:.1f}" if s["poly_ms"] else "  —  "
        spd_s   = f"{s['speed_ratio']:.2f}×" if s["speed_ratio"] else "  —  "
        uni_s   = f"{s['norm_uni']:.2f}" if s["norm_uni"] is not None else "  —  "
        win     = "✓" if s["ratio"] > 1.05 else ("✗" if s["ratio"] < 0.95 else "≈")
        print(
            f"  {_dims_str(s['dims']):18s} {s['K']:>2d} {s['L']:>3d}  "
            f"{s['lut_mse_mean']:>10.3e} {s['lut_mse_std']:>7.1e}  "
            f"{s['poly_mse_mean']:>10.3e}  "
            f"{ratio_s:>7s}{win}  {lms_s:>7s}  {pms_s:>7s}  {spd_s:>5s}  "
            f"{uni_s:>5s}  {s['cv']:>5.2f}  "
            f"{coverage_flag(s['density']):>5s}  {s['mem_lut_bytes']/1024:>5.2f}"
        )


def print_edge_table(sums: List[dict], title: str = ""):
    if not sums: return
    sums = sorted(sums, key=lambda s: s["lut_mse_mean"])
    if title: _hdr(title)
    print(f"  {'K,L':8s} {'params':>6s}  "
          f"{'LUT MSE':>10s} {'±':>7s}  "
          f"{'Poly LS':>10s}  {'Poly GD':>10s}  "
          f"{'vs LS':>7s}  {'vs GD':>7s}  "
          f"{'ms LUT':>7s}  {'ms poly':>7s}  {'cv':>5s}")
    print("  " + "─" * 95)
    for s in sums:
        lms_s  = f"{s['lut_ms']:.1f}"    if s["lut_ms"]     else "  —  "
        pgms_s = f"{s['poly_gd_ms']:.1f}" if s["poly_gd_ms"] else "  —  "
        r_ls   = s["ratio_vs_ls"]; r_gd = s["ratio_vs_gd"]
        rls_s  = f"{r_ls:.0f}×" if r_ls >= 10 else f"{r_ls:.2f}×"
        rgd_s  = f"{r_gd:.0f}×" if r_gd >= 10 else f"{r_gd:.2f}×"
        wls    = "✓" if r_ls > 1.05 else "✗"
        wgd    = "✓" if r_gd > 1.05 else "✗"
        print(
            f"  K={s['K']},L={s['L']:<2d} {s['K']*s['L']:>6d}  "
            f"{s['lut_mse_mean']:>10.3e} {s['lut_mse_std']:>7.1e}  "
            f"{s['poly_ls_mse']:>10.3e}  {s['poly_gd_mse']:>10.3e}  "
            f"{rls_s:>7s}{wls}  {rgd_s:>7s}{wgd}  "
            f"{lms_s:>7s}  {pgms_s:>7s}  {s['cv']:>5.2f}"
        )


def print_speed_table(sums: List[dict], title: str = ""):
    rows = [s for s in sums if s.get("speed_ratio") is not None]
    if not rows: return
    rows.sort(key=lambda s: s["speed_ratio"], reverse=True)
    if title: _hdr(title)
    print(f"  {'config':28s}  {'target':12s}  {'K×L':>5s}  "
          f"{'LUT ms':>8s}  {'Poly ms':>8s}  {'ratio':>7s}  {'accuracy':>9s}")
    print("  " + "─" * 90)
    for s in rows:
        cfg = f"{_dims_str(s.get('dims', [1,1]))}" if "dims" in s else f"K={s['K']},L={s['L']}"
        kl  = s.get("n_lut_params", s.get("K", 0) * s.get("L", 0))
        acc = f"{s.get('ratio', s.get('ratio_vs_gd', 0)):.1f}×"
        spd = s["speed_ratio"]
        tag = "LUT faster" if spd > 1.05 else ("poly faster" if spd < 0.95 else "≈ same")
        print(
            f"  {cfg:28s}  {s['target']:12s}  {kl:>5d}  "
            f"{s['lut_ms']:>8.2f}  {s.get('poly_ms', s.get('poly_gd_ms', 0)):>8.2f}  "
            f"{spd:>6.2f}×  {acc:>9s}  {tag}"
        )


def print_pareto(sums: List[dict], target: str):
    tsums = [s for s in sums if s["target"] == target]
    if not tsums: return
    tsums.sort(key=lambda s: s["lut_mse_mean"])
    bins = [128, 256, 512, 1024, 2048, 4096, 9999999]
    labs = ["<128B", "<256B", "<512B", "<1kB", "<2kB", "<4kB", "4kB+"]
    print(f"\n  Pareto ({target}):")
    prev_mse = float("inf")
    for lim, lab in zip(bins, labs):
        cands = [s for s in tsums if s["mem_lut_bytes"] <= lim]
        if not cands: continue
        best = min(cands, key=lambda s: s["lut_mse_mean"])
        if best["lut_mse_mean"] >= prev_mse: continue
        prev_mse = best["lut_mse_mean"]
        ratio_s = f"{best['ratio']:.0f}×" if best['ratio'] >= 10 else f"{best['ratio']:.1f}×"
        print(
            f"    {lab:6s}  {_dims_str(best['dims']):16s} K={best['K']} L={best['L']}  "
            f"MSE={best['lut_mse_mean']:.3e}  poly/LUT={ratio_s}  "
            f"cv={best['cv']:.2f}  {coverage_flag(best['density'])}"
        )


def print_key_findings(sums: List[dict]):
    _hdr("Key findings")
    targets_seen = sorted({s["target"] for s in sums})
    for target in targets_seen:
        tsums = [s for s in sums if s["target"] == target]
        valid = [s for s in tsums if not math.isnan(s.get("ratio", float("nan")))]
        if not valid: continue
        valid.sort(key=lambda s: s["lut_mse_mean"])
        best_acc = valid[0]
        wins = [s for s in valid if s["ratio"] > 1.0]
        best_win = max(wins, key=lambda s: s["ratio"]) if wins else None
        stable = min(valid, key=lambda s: s["cv"])
        fast = min((s for s in valid if s.get("speed_ratio")), key=lambda s: s.get("lut_ms", 99)) \
               if any(s.get("speed_ratio") for s in valid) else None
        print(f"\n  [{target}]")
        print(f"    best accuracy : {_dims_str(best_acc['dims'])} K={best_acc['K']} L={best_acc['L']}  "
              f"LUT={best_acc['lut_mse_mean']:.3e}  poly={best_acc['poly_mse_mean']:.3e}  "
              f"ratio={best_acc['ratio']:.1f}×  cv={best_acc['cv']:.2f}")
        if best_win and best_win is not best_acc:
            print(f"    best ratio    : {_dims_str(best_win['dims'])} K={best_win['K']} L={best_win['L']}  "
                  f"poly/LUT={best_win['ratio']:.0f}×")
        if stable["cv"] < 0.25:
            print(f"    most stable   : {_dims_str(stable['dims'])} K={stable['K']} L={stable['L']}  "
                  f"cv={stable['cv']:.2f}  LUT={stable['lut_mse_mean']:.3e}")
        if fast and fast.get("speed_ratio"):
            print(f"    speed         : {_dims_str(fast['dims'])} K={fast['K']} L={fast['L']}  "
                  f"{fast['lut_ms']:.1f}ms vs poly {fast['poly_ms']:.1f}ms  "
                  f"= {fast['speed_ratio']:.2f}× per epoch")


# ─────────────────────────────────────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: Dict[str, dict] = {
    "quick": dict(
        mode="stack",
        dims_list=[[1,4,1],[1,8,1],[1,4,4,1]],
        kl_list=[(1,8),(2,4),(4,8),(8,4)],
        targets=["sine","cusp","saturating"],
        seeds=[0,1,2],
        n_train=500, epochs=1000, patience=200,
        measure_timing=False,
    ),
    "standard": dict(
        mode="stack",
        dims_list=[[1,4,1],[1,8,1],[1,16,1],[1,4,4,1],[1,8,4,1],[1,4,4,4,1]],
        kl_list=[(1,4),(1,8),(2,4),(2,8),(4,4),(4,8),(8,4),(16,2)],
        targets=["sine","cusp","saturating","piecewise_smooth","local_sharp"],
        seeds=[0,1,2,3,4],
        n_train=500, epochs=2000, patience=400,
        measure_timing=True,
    ),
    "deep": dict(
        mode="stack",
        dims_list=[[1,4,1],[1,4,4,1],[1,4,4,4,1],[1,8,4,1],[1,4,8,1]],
        kl_list=[(1,4),(1,8),(2,4),(2,8),(4,4)],
        targets=["sine","cusp","saturating"],
        seeds=[0,1,2,3,4],
        n_train=500, epochs=2500, patience=500,
        measure_timing=True,
    ),
    "single_edge": dict(
        mode="single_edge",
        kl_list=[(1,8),(2,8),(4,8),(4,16),(4,32),(8,4),(16,2)],
        targets=["sine","cusp","saturating","piecewise_smooth"],
        seeds=[0,1,2,3,4],
        n_train=500, epochs=2000, patience=0,
        measure_timing=True,
    ),
    "wide": dict(
        mode="stack",
        dims_list=[[1,4,1],[1,8,1],[1,16,1],[1,32,1]],
        kl_list=[(1,4),(1,8),(2,4),(2,8),(4,4),(4,8),(8,4),(8,8),(16,2),(16,4)],
        targets=["cusp","saturating","local_sharp"],
        seeds=[0,1,2,3],
        n_train=500, epochs=2000, patience=400,
        measure_timing=True,
    ),
    "budget": dict(
        # ultra-compact: <= 512 bytes LUT storage
        mode="stack",
        dims_list=[[1,2,1],[1,4,1],[1,2,2,1],[1,4,4,1]],
        kl_list=[(1,2),(1,4),(1,8),(2,2),(2,4)],
        targets=["sine","cusp","saturating"],
        seeds=[0,1,2,3,4],
        n_train=300, epochs=2000, patience=400,
        measure_timing=False,
    ),
    "2d": dict(
        mode="stack",
        dims_list=[[2,4,1],[2,8,1],[2,4,4,1]],
        kl_list=[(1,4),(1,8),(2,4),(2,8)],
        targets=["feynman_2d"],
        seeds=[0,1,2],
        n_train=800, epochs=3000, patience=600,
        measure_timing=False,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    mode: str,
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
    measure_timing: bool,
    timing_repeats: int,
    output_path: Optional[Path],
    verbose: bool,
) -> None:

    print(f"\n{'━'*W}")
    print(f"  Mode: {mode}  |  Targets: {targets}")
    print(f"  n_train={n_train}  epochs={epochs}  patience={patience}  λ₂={lambda_2}  lr={lr}")
    print(f"  seeds={seeds}  timing={'on (warmup+median)' if measure_timing else 'off'}")
    print(f"  Poly: PolyKANStack (gradient-trained, same dims, degree=K×L-1, same λ₂)")
    print(f"{'━'*W}\n")

    all_stack_runs: List[StackRun] = []
    all_edge_runs:  List[EdgeRun]  = []

    # ── Single-edge mode ──────────────────────────────────────────────────────
    if mode == "single_edge":
        candidates = [
            (K, L, target)
            for (K, L), target in itertools.product(kl_list, targets)
            if (n_train / max(K * L, 1)) >= min_density
        ]
        total = len(candidates) * len(seeds)
        print(f"  {len(candidates)} configs × {len(seeds)} seeds = {total} runs (single-edge [1→1])\n")
        done = 0
        for (K, L), target in itertools.product(kl_list, targets):
            density = n_train / max(K * L, 1)
            if density < min_density:
                if verbose: print(f"  skip K={K},L={L} target={target} density={density:.1f}")
                continue
            print(f"▶  K={K},L={L:<2d} (degree={K*L-1:3d})  {target:14s}  "
                  f"density={density:.1f}  {K*L+4*K}B")
            seed_runs = []
            for seed in seeds:
                try:
                    r = run_edge(K, L, target, seed, n_train, epochs, lr, lambda_2,
                                 measure_timing, timing_repeats)
                    seed_runs.append(r)
                    all_edge_runs.append(r)
                    done += 1
                    print(f"  [{done:3d}/{total}] seed={seed}  "
                          f"LUT={r.lut_mse:.3e}  LS={r.poly_ls_mse:.3e}  "
                          f"GD={r.poly_gd_mse:.3e}  "
                          f"ep={r.lut_best_epoch}"
                          + (f"  {r.lut_ms_per_epoch:.1f}ms" if r.lut_ms_per_epoch else ""))
                except Exception as e:
                    print(f"  [{done:3d}/{total}] seed={seed}  ERROR: {e}")
                    done += 1
            if seed_runs:
                agg = aggregate_edge(seed_runs)
                print(f"   → LUT={agg['lut_mse_mean']:.3e}±{agg['lut_mse_std']:.1e}  "
                      f"LS={agg['poly_ls_mse']:.3e}  GD={agg['poly_gd_mse']:.3e}  "
                      f"vs_ls={agg['ratio_vs_ls']:.1f}×  vs_gd={agg['ratio_vs_gd']:.1f}×  "
                      f"cv={agg['cv']:.2f}\n")

        # per-target tables
        for target in targets:
            runs_t = [r for r in all_edge_runs if r.target == target]
            if not runs_t: continue
            agg_by_kl: Dict[Tuple[int,int], List[EdgeRun]] = defaultdict(list)
            for r in runs_t:
                agg_by_kl[(r.K, r.L)].append(r)
            sums = [aggregate_edge(v) for v in agg_by_kl.values()]
            print_edge_table(sums, title=f"Single-edge [1→1] — {target}")

        # global speed table
        all_aggs = []
        for runs_t in [all_edge_runs]:
            agg_by = defaultdict(list)
            for r in runs_t:
                agg_by[(r.K, r.L, r.target)].append(r)
            for v in agg_by.values():
                all_aggs.append(aggregate_edge(v))
        print_speed_table(
            [{"K": a["K"], "L": a["L"], "target": a["target"], "dims": [1,1],
              "lut_ms": a["lut_ms"], "poly_ms": a["poly_gd_ms"],
              "speed_ratio": a["speed_ratio"],
              "ratio_vs_gd": a["ratio_vs_gd"],
              "ratio": a["ratio_vs_gd"],
              "n_lut_params": a["K"]*a["L"]}
             for a in all_aggs if a.get("speed_ratio")],
            title="Training speed: LUT vs gradient-trained poly"
        )

    # ── Stack mode ────────────────────────────────────────────────────────────
    else:
        candidates = []
        for dims, (K, L), target in itertools.product(dims_list, kl_list, targets):
            is_2d = dims[0] == 2
            is_2d_tgt = target == "feynman_2d"
            if is_2d != is_2d_tgt:
                continue
            density = bottleneck_density(dims, K, L, n_train)
            if density < min_density:
                if verbose:
                    print(f"  skip {_dims_str(dims)} K={K} L={L} {target} density={density:.1f}")
                continue
            candidates.append((dims, K, L, target))

        total = len(candidates) * len(seeds)
        print(f"  {len(candidates)} configs × {len(seeds)} seeds = {total} runs")
        print(f"  Poly: PolyKANStack (gradient-trained, matched dims + params/edge)\n")

        done = 0
        for dims, K, L, target in candidates:
            density = bottleneck_density(dims, K, L, n_train)
            print(
                f"▶  {_dims_str(dims):16s} K={K} L={L:<2d}  {target:14s}  "
                f"density={density:.1f}  {total_lut_bytes(dims,K,L)/1024:.2f}kB  "
                f"{coverage_flag(density)}"
            )
            seed_runs = []
            for seed in seeds:
                try:
                    r = run_stack(
                        dims, K, L, target, seed,
                        n_train, epochs, lr, lambda_2, patience,
                        recal_every, ema_alpha,
                        measure_timing=(measure_timing and seed == seeds[0]),
                        timing_repeats=timing_repeats,
                    )
                    seed_runs.append(r)
                    all_stack_runs.append(r)
                    done += 1
                    ratio = r.poly_mse / max(r.lut_mse, 1e-30)
                    print(f"  [{done:3d}/{total}] seed={seed}  "
                          f"LUT={r.lut_mse:.3e}  poly={r.poly_mse:.3e}  "
                          f"ratio={ratio:.1f}×  ep={r.lut_best_epoch}"
                          + (f"  {r.lut_ms_per_epoch:.1f}vs{r.poly_ms_per_epoch:.1f}ms"
                             if r.lut_ms_per_epoch else ""))
                except Exception as e:
                    print(f"  [{done:3d}/{total}] seed={seed}  ERROR: {e}")
                    done += 1

            if seed_runs:
                a = aggregate_stack(seed_runs)
                spd_s = f"  spd={a['speed_ratio']:.2f}×" if a["speed_ratio"] else ""
                print(f"   → LUT={a['lut_mse_mean']:.3e}±{a['lut_mse_std']:.1e}  "
                      f"poly={a['poly_mse_mean']:.3e}  ratio={a['ratio']:.1f}×  "
                      f"cv={a['cv']:.2f}{spd_s}\n")

        # Per-target tables
        agg_by: Dict[Tuple, List[StackRun]] = defaultdict(list)
        for r in all_stack_runs:
            agg_by[(tuple(r.dims), r.K, r.L, r.target)].append(r)
        all_sums = [aggregate_stack(v) for v in agg_by.values()]

        for target in targets:
            tsums = [s for s in all_sums if s["target"] == target]
            print_stack_table(tsums, title=f"LUT vs poly-KAN (matched dims+params) — {target}")

        # Speed table
        speed_rows = [s for s in all_sums if s.get("speed_ratio") is not None]
        if speed_rows:
            print_speed_table(speed_rows, title="Training speed: LUT vs poly-KAN (ms/epoch, warmup+median)")

        # Pareto per target
        _hdr("Pareto front: best LUT MSE per memory budget")
        for target in targets:
            tsums = [s for s in all_sums if s["target"] == target]
            print_pareto(tsums, target)

        print_key_findings(all_sums)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _clean(d: dict) -> dict:
            out = {}
            for k, v in d.items():
                if isinstance(v, float) and math.isnan(v):
                    out[k] = None
                elif isinstance(v, list):
                    out[k] = v
                else:
                    out[k] = v
            return out

        if mode == "single_edge":
            runs_out = [asdict(r) for r in all_edge_runs]
        else:
            runs_out = [asdict(r) for r in all_stack_runs]

        out = {
            "config": {
                "mode": mode, "n_train": n_train, "epochs": epochs,
                "patience": patience, "lambda_2": lambda_2, "lr": lr,
                "seeds": seeds, "min_density": min_density,
                "measure_timing": measure_timing,
            },
            "runs": [_clean(r) for r in runs_out],
        }
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="LUT-KAN vs poly-KAN: comprehensive accuracy + speed benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=list(PRESETS.keys()), default="quick",
                   help=f"Preset: {list(PRESETS.keys())} (default: quick)")
    p.add_argument("--dims",    nargs="+", type=lambda s: [int(x) for x in s.split(",")],
                   metavar="IN,H,...,OUT", help="Architecture list, e.g. 1,4,1 1,4,4,1")
    p.add_argument("--kl",      nargs="+", type=lambda s: tuple(int(x) for x in s.split(",")),
                   metavar="K,L",          help="K,L pairs, e.g. 2,8 4,8")
    p.add_argument("--targets", nargs="+", choices=list(TARGETS.keys()) + ["feynman_2d"],
                   metavar="TARGET")
    p.add_argument("--seeds",   nargs="+", type=int)
    p.add_argument("--n_train", type=int)
    p.add_argument("--epochs",  type=int)
    p.add_argument("--patience",type=int)
    p.add_argument("--lambda2", type=float, default=1.0,   help="λ₂ for both LUT and poly (default 1.0)")
    p.add_argument("--lr",      type=float, default=1e-2,  help="Adam lr (default 1e-2)")
    p.add_argument("--ema",     type=float, default=0.01,  help="EMA alpha for norm tracking (default 0.01)")
    p.add_argument("--recal",   type=int,   default=100,   help="Recalibrate norms every N epochs (default 100)")
    p.add_argument("--min_density", type=float, default=3.0, help="Min pts/cell to include config (default 3.0)")
    p.add_argument("--timing",  action="store_true",       help="Force timing measurement (warmup + median)")
    p.add_argument("--no_timing", action="store_true",     help="Disable timing measurement")
    p.add_argument("--timing_repeats", type=int, default=4, help="Repeats for timing (default 4)")
    p.add_argument("--output",  type=Path)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    preset = PRESETS[args.mode]
    measure_timing = preset.get("measure_timing", False)
    if args.timing:    measure_timing = True
    if args.no_timing: measure_timing = False

    out = args.output or (
        Path(__file__).parent.parent / "results" / "sweep_full"
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
        patience=args.patience if args.patience is not None else preset.get("patience", 0),
        lambda_2=args.lambda2,
        lr=args.lr,
        recal_every=args.recal,
        ema_alpha=args.ema,
        min_density=args.min_density,
        measure_timing=measure_timing,
        timing_repeats=args.timing_repeats,
        output_path=out,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
