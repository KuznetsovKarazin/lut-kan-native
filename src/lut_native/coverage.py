"""
Coverage diagnostics for multi-edge LUT-KAN.

Phase M2 of the multi-edge plan. Completely read-only: instruments an already-
trained LUTKAN2Layer's forward pass to measure which LUT cells actually receive
inputs. Does not modify the model or training.

Three metrics per edge (shape = (K*L,) cells):

  1. visited_fraction
       fraction of cells that receive at least one sample's interpolation weight
       above a threshold (default 1/N, i.e. "non-trivial hit")

  2. effective_support  (entropy-based)
       exp(H) where H is Shannon entropy over normalized cell-hit counts.
       This is a smooth version of "how many cells actually matter".
       For uniformly-hit K*L cells, effective_support = K*L.
       For all mass in one cell, effective_support = 1.

  3. range_utilization
       fraction of the domain [x_min, x_max) that the input distribution actually
       covers, measured as (observed_x_range) / (x_max - x_min).

Also reports:

  - per-segment hit counts (where did the distribution land?)
  - gini coefficient on cell hits (concentration measure)

The expensive thing is weight accounting — for each sample-edge pair we need to
know which two LUT cells were hit and with what weight. We do this via a
non-reducing variant of the edge forward, which returns (r0, r1, w) as well as
the output. This reuses the same indexing math as LUTKAN2Layer._edge_forward_bulk
but records indices instead of gathering values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Per-edge cell-hit accumulator
# ─────────────────────────────────────────────────────────────────────────────

def _per_sample_indices(
    x: torch.Tensor,            # (N, n_src)
    K: int,
    L: int,
    x_min: float,
    x_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (k, r0, r1, w) for each (sample, src) pair, matching
    LUTKAN2Layer._edge_forward_bulk's math exactly.

    Returns all tensors of shape (N, n_src).
    """
    seg_width = (x_max - x_min) / K
    hi = x_max - 1e-7
    x_clip = torch.clamp(x, x_min, hi)
    t = (x_clip - x_min) / seg_width
    k = torch.clamp(torch.floor(t).long(), 0, K - 1)
    u = torch.clamp(t - k.to(t.dtype), 0.0, 1.0)
    pos = u * (L - 1)
    r0 = torch.clamp(torch.floor(pos).long(), 0, L - 1)
    r1 = torch.clamp(r0 + 1, 0, L - 1)
    w = pos - r0.to(pos.dtype)
    return k, r0, r1, w


def _accumulate_cell_hits_for_layer(
    x_in: np.ndarray,        # (N, n_src)
    n_dst: int,
    K: int,
    L: int,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    """Sum of interpolation weights per LUT cell, per (src, dst) edge.

    Returns array of shape (n_src, n_dst, K, L), where entry [i, j, k, l] is the
    total weight that cell (k, l) of edge (i->j) received across all N samples.

    Because edges at layer boundaries are (n_src, n_dst)-replicated per sample
    (same x_in[:, i] hits every edge (i, :)), the per-edge hit counts only
    depend on the src index, not dst. We still return the full (n_src, n_dst,
    K, L) tensor so downstream code is uniform; it's a view-like broadcast.
    """
    x = torch.from_numpy(x_in.astype(np.float32))
    N, n_src = x.shape
    k, r0, r1, w = _per_sample_indices(x, K, L, x_min, x_max)  # all (N, n_src)

    # The per-src weight pattern is identical across dst. We compute one
    # (n_src, K, L) hit-map, then broadcast to (n_src, n_dst, K, L).
    hits_src = np.zeros((n_src, K, L), dtype=np.float64)
    k_np = k.cpu().numpy()
    r0_np = r0.cpu().numpy()
    r1_np = r1.cpu().numpy()
    w_np = w.cpu().numpy()
    for i in range(n_src):
        # vectorized scatter-add on (K, L)
        np.add.at(hits_src[i], (k_np[:, i], r0_np[:, i]), 1.0 - w_np[:, i])
        np.add.at(hits_src[i], (k_np[:, i], r1_np[:, i]), w_np[:, i])

    # Broadcast
    return np.broadcast_to(hits_src[:, None, :, :], (n_src, n_dst, K, L)).copy()


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _visited_fraction(hits: np.ndarray, threshold_counts: float = 1.0) -> float:
    """Fraction of cells in `hits` (any shape) with count >= threshold_counts."""
    total = hits.size
    visited = int((hits >= threshold_counts).sum())
    return visited / total if total else 0.0


def _effective_support(hits: np.ndarray, eps: float = 1e-12) -> float:
    """exp(Shannon entropy) of normalized hit distribution. Higher = more spread."""
    flat = hits.ravel().astype(np.float64)
    total = flat.sum()
    if total <= eps:
        return 0.0
    p = flat / total
    nz = p[p > eps]
    H = -float((nz * np.log(nz)).sum())
    return float(np.exp(H))


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array. 0 = uniform, 1 = fully concentrated."""
    x = np.sort(values.ravel().astype(np.float64))
    n = x.size
    s = x.sum()
    if s <= 0 or n == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() - (n + 1) * s) / (n * s))


def _range_utilization(x: np.ndarray, x_min: float, x_max: float) -> float:
    """Observed input range / domain range. Clamped to [0, 1]."""
    lo = float(x.min())
    hi = float(x.max())
    obs = max(0.0, min(hi, x_max) - max(lo, x_min))
    dom = x_max - x_min
    return obs / dom if dom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerCoverageReport:
    name: str
    n_src: int
    n_dst: int
    K: int
    L: int
    x_min: float
    x_max: float

    # Per-src summary (same across dst due to architecture)
    visited_fraction_per_src: List[float]      # length n_src
    effective_support_per_src: List[float]     # length n_src
    range_utilization_per_src: List[float]     # length n_src
    gini_per_src: List[float]                  # length n_src
    segment_hit_counts_per_src: List[List[float]]  # length n_src, each list of K floats

    # Summary across the whole layer
    visited_fraction_mean: float
    effective_support_mean: float
    range_utilization_mean: float
    x_in_stats_per_src: List[Dict[str, float]]  # {mean, std, min, max}

    # Raw hits for further inspection (can be large - optional)
    hits_src: Optional[np.ndarray] = field(default=None, repr=False)  # (n_src, K, L)


@dataclass
class KAN2CoverageReport:
    layer1: LayerCoverageReport
    layer2: LayerCoverageReport
    hidden_activation_stats: Dict[str, float]   # distribution of tanh(z) values


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point: produce a coverage report for a trained LUTKAN2Layer
# ─────────────────────────────────────────────────────────────────────────────

def compute_kan2_coverage(
    model,                     # a LUTKAN2Layer (imported lazily to avoid cycles)
    x_in: np.ndarray,          # (N, in_dim) input data to measure on
    include_raw_hits: bool = False,
) -> KAN2CoverageReport:
    """Run forward + collect coverage metrics per layer.

    Measurements are taken on the provided x_in distribution (typically the
    training set or a held-out sample). For layer 2, the inputs are tanh(z_1)
    as produced by the model.

    The model is not modified, but its forward is called once with grad disabled
    to produce the hidden activations.
    """
    # Arrange x_in
    x_in = np.atleast_2d(np.asarray(x_in, dtype=np.float32))
    if x_in.shape[1] != model.in_dim:
        # allow (N,) for in_dim=1
        if model.in_dim == 1 and x_in.ndim == 2 and x_in.shape[0] == 1:
            x_in = x_in.T
        elif x_in.ndim == 1 and model.in_dim == 1:
            x_in = x_in.reshape(-1, 1)
        else:
            raise ValueError(
                f"x_in has {x_in.shape[1]} cols but model.in_dim={model.in_dim}"
            )

    # --- Layer 1 coverage ---
    L1_hits = _accumulate_cell_hits_for_layer(
        x_in=x_in, n_dst=model.hidden_dim, K=model.K, L=model.L,
        x_min=model.x_min, x_max=model.x_max,
    )  # (in_dim, hidden_dim, K, L) — broadcast across hidden_dim

    l1_report = _build_layer_report(
        name="layer1",
        x_in=x_in, hits=L1_hits,
        n_src=model.in_dim, n_dst=model.hidden_dim,
        K=model.K, L=model.L,
        x_min=model.x_min, x_max=model.x_max,
        include_raw_hits=include_raw_hits,
    )

    # --- Compute hidden activations through layer 1 to feed layer 2 ---
    with torch.no_grad():
        xt = torch.from_numpy(x_in)
        # Run first layer to get z = sum over edges.
        z1 = model._edge_forward_bulk(
            xt, model.lut_l1, model._x_min_l1, model._x_max_l1, model._seg_width_l1,
        )  # (N, hidden_dim)
        # Match model's activation mode exactly so coverage measures what
        # training actually sees.
        activation = getattr(model, "activation", "tanh")
        if activation == "tanh":
            a1_t = torch.tanh(z1)
        elif activation == "zscore":
            a1_t = (z1 - model._z_mean) / model._z_std
        else:
            raise ValueError(activation)
        a1 = a1_t.cpu().numpy().astype(np.float32)
        z1_np = z1.cpu().numpy()

    hidden_stats = {
        "activation": activation,
        "z_mean": float(z1_np.mean()),
        "z_std": float(z1_np.std()),
        "z_min": float(z1_np.min()),
        "z_max": float(z1_np.max()),
        # Historical names kept for backward-compat; for zscore, these are the
        # post-normalization stats, so 'tanh_z_*' is a legacy alias for 'a_*'.
        "tanh_z_mean": float(a1.mean()),
        "tanh_z_std": float(a1.std()),
        "tanh_z_min": float(a1.min()),
        "tanh_z_max": float(a1.max()),
        "frac_abs_gt_0_99": float((np.abs(a1) > 0.99).mean()),
        "frac_abs_lt_0_1":  float((np.abs(a1) < 0.1).mean()),
    }

    # --- Layer 2 coverage (inputs are a1; domain depends on activation) ---
    l2_x_min = getattr(model, "x_min_l2", -1.0)
    l2_x_max = getattr(model, "x_max_l2", 1.0)
    L2_hits = _accumulate_cell_hits_for_layer(
        x_in=a1, n_dst=model.out_dim, K=model.K, L=model.L,
        x_min=l2_x_min, x_max=l2_x_max,
    )  # (hidden_dim, out_dim, K, L)

    l2_report = _build_layer_report(
        name="layer2",
        x_in=a1, hits=L2_hits,
        n_src=model.hidden_dim, n_dst=model.out_dim,
        K=model.K, L=model.L,
        x_min=l2_x_min, x_max=l2_x_max,
        include_raw_hits=include_raw_hits,
    )

    return KAN2CoverageReport(
        layer1=l1_report,
        layer2=l2_report,
        hidden_activation_stats=hidden_stats,
    )


def _build_layer_report(
    name: str,
    x_in: np.ndarray,
    hits: np.ndarray,           # (n_src, n_dst, K, L)
    n_src: int, n_dst: int,
    K: int, L: int,
    x_min: float, x_max: float,
    include_raw_hits: bool,
) -> LayerCoverageReport:
    # Collapse dst axis since hits are broadcast-identical across dst
    hits_src = hits[:, 0, :, :]  # (n_src, K, L)

    visited, eff, rang, gi, seg_counts, xstats = [], [], [], [], [], []
    for i in range(n_src):
        h = hits_src[i]                  # (K, L)
        visited.append(_visited_fraction(h, threshold_counts=1.0))
        eff.append(_effective_support(h))
        rang.append(_range_utilization(x_in[:, i], x_min, x_max))
        gi.append(_gini(h))
        seg_counts.append(h.sum(axis=1).tolist())
        xi = x_in[:, i]
        xstats.append({
            "mean": float(xi.mean()),
            "std": float(xi.std()),
            "min": float(xi.min()),
            "max": float(xi.max()),
        })

    return LayerCoverageReport(
        name=name,
        n_src=n_src, n_dst=n_dst, K=K, L=L,
        x_min=x_min, x_max=x_max,
        visited_fraction_per_src=visited,
        effective_support_per_src=eff,
        range_utilization_per_src=rang,
        gini_per_src=gi,
        segment_hit_counts_per_src=seg_counts,
        visited_fraction_mean=float(np.mean(visited)) if visited else 0.0,
        effective_support_mean=float(np.mean(eff)) if eff else 0.0,
        range_utilization_mean=float(np.mean(rang)) if rang else 0.0,
        x_in_stats_per_src=xstats,
        hits_src=(hits_src if include_raw_hits else None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Serialization helper
# ─────────────────────────────────────────────────────────────────────────────

def coverage_report_to_dict(report: KAN2CoverageReport) -> Dict:
    """Convert to plain dict for JSON serialization. Drops raw hits arrays."""
    def layer_to_dict(L):
        return {
            "name": L.name,
            "n_src": L.n_src, "n_dst": L.n_dst, "K": L.K, "L": L.L,
            "x_min": L.x_min, "x_max": L.x_max,
            "visited_fraction_per_src": L.visited_fraction_per_src,
            "effective_support_per_src": L.effective_support_per_src,
            "range_utilization_per_src": L.range_utilization_per_src,
            "gini_per_src": L.gini_per_src,
            "segment_hit_counts_per_src": L.segment_hit_counts_per_src,
            "visited_fraction_mean": L.visited_fraction_mean,
            "effective_support_mean": L.effective_support_mean,
            "range_utilization_mean": L.range_utilization_mean,
            "x_in_stats_per_src": L.x_in_stats_per_src,
        }
    return {
        "layer1": layer_to_dict(report.layer1),
        "layer2": layer_to_dict(report.layer2),
        "hidden_activation_stats": report.hidden_activation_stats,
    }
