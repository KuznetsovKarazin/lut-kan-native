"""
Coverage diagnostics for LUTKANStack (N-layer stacks with adaptive norms).

Extends the two-layer coverage in coverage.py to arbitrary depth.

For each LUTBlock in the stack we measure the same three metrics as the
two-layer version:

  1. visited_fraction  — fraction of (K*L) cells that receive non-trivial weight
  2. effective_support — exp(H) of normalised hit counts (entropy-based)
  3. range_utilization — observed input range / domain width

Additionally, for each LUTInterLayerNorm we record:

  4. norm_uniformity   — how uniformly the normalised activations cover K segments
  5. active_segment_fraction — fraction of K segments with > 0 samples

This lets you see at a glance whether the norms are working: a well-calibrated
norm should have uniformity close to 1.0 and active_segment_fraction = 1.0.

Usage
-----
    from lut_native.coverage_stack import compute_stack_coverage
    report = compute_stack_coverage(model, x_train)
    print(report)               # short summary
    d = stack_coverage_to_dict(report)   # for JSON serialisation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .coverage import (
    _accumulate_cell_hits_for_layer,
    _visited_fraction,
    _effective_support,
    _range_utilization,
    _gini,
    _build_layer_report,
    LayerCoverageReport,
)
from .kan_stack import LUTKANStack


# ─────────────────────────────────────────────────────────────────────────────
# Norm statistics dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NormCoverageReport:
    """Coverage metrics for one LUTInterLayerNorm."""
    norm_index: int
    dim: int
    x_min: float
    x_max: float
    K: int                               # K of the NEXT block

    # Activation stats before normalisation (z, unbounded)
    z_mean_per_channel:  List[float]     # (dim,)
    z_std_per_channel:   List[float]     # (dim,)
    z_min_per_channel:   List[float]
    z_max_per_channel:   List[float]

    # After normalisation (a = normalised z, ∈ [x_min, x_max))
    # How uniformly does `a` cover the K segments of the next block?
    seg_counts_per_channel: List[List[float]]  # (dim, K)
    uniformity_per_channel: List[float]        # 1 = perfectly uniform, 0 = all in one seg
    active_segment_fraction_per_channel: List[float]   # fraction of K segments hit

    uniformity_mean: float
    active_segment_fraction_mean: float

    # Norm parameters snapshot
    shift_snapshot: List[float]          # (dim,)
    scale_snapshot: List[float]          # (dim,) — exp(log_scale)

    # Fraction of inputs that were clamped to boundary by the norm
    frac_clamped_lo: float
    frac_clamped_hi: float


# ─────────────────────────────────────────────────────────────────────────────
# Stack coverage report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StackCoverageReport:
    """Full coverage report for a LUTKANStack."""
    dims: List[int]
    K: int
    L: int
    n_samples: int

    block_reports: List[LayerCoverageReport]   # one per LUTBlock
    norm_reports:  List[NormCoverageReport]    # one per LUTInterLayerNorm

    def summary(self) -> str:
        lines = [f"LUTKANStack coverage — dims={self.dims}, K={self.K}, L={self.L}, N={self.n_samples}"]
        for i, blk in enumerate(self.block_reports):
            lines.append(
                f"  block[{i}] ({blk.n_src}->{blk.n_dst}): "
                f"visited={blk.visited_fraction_mean:.2f}  "
                f"eff_support={blk.effective_support_mean:.1f}/{self.K * self.L}  "
                f"range_util={blk.range_utilization_mean:.2f}"
            )
            if i < len(self.norm_reports):
                nm = self.norm_reports[i]
                lines.append(
                    f"  norm[{i}]   uniformity={nm.uniformity_mean:.2f}  "
                    f"active_segs={nm.active_segment_fraction_mean:.2f}  "
                    f"clamped_lo={nm.frac_clamped_lo:.2f} "
                    f"clamped_hi={nm.frac_clamped_hi:.2f}"
                )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_stack_coverage(
    model: LUTKANStack,
    x_in: np.ndarray,
    include_raw_hits: bool = False,
) -> StackCoverageReport:
    """
    Instrument a trained LUTKANStack to measure cell coverage per block and
    normalisation uniformity per inter-layer norm.

    Parameters
    ----------
    model : LUTKANStack
    x_in : (N, dims[0]) input data (typically x_train).
    include_raw_hits : bool
        If True, LayerCoverageReport.hits_src arrays are populated.

    Returns
    -------
    StackCoverageReport
    """
    x_in = np.atleast_2d(np.asarray(x_in, dtype=np.float32))
    if x_in.ndim == 1 and model.dims[0] == 1:
        x_in = x_in.reshape(-1, 1)
    N = x_in.shape[0]

    block_reports: List[LayerCoverageReport]  = []
    norm_reports:  List[NormCoverageReport]   = []

    # Propagate activations through the stack, measuring at each boundary.
    z_np = x_in.copy()

    for i, blk in enumerate(model.blocks):
        in_d  = blk.in_dim
        out_d = blk.out_dim
        x_min = blk.x_min
        x_max = blk.x_max

        # --- block coverage (input side) ---
        hits = _accumulate_cell_hits_for_layer(
            x_in=z_np, n_dst=out_d, K=blk.K, L=blk.L,
            x_min=x_min, x_max=x_max,
        )
        rep = _build_layer_report(
            name=f"block_{i}",
            x_in=z_np, hits=hits,
            n_src=in_d, n_dst=out_d,
            K=blk.K, L=blk.L,
            x_min=x_min, x_max=x_max,
            include_raw_hits=include_raw_hits,
        )
        block_reports.append(rep)

        # --- run block forward in torch to get z (pre-norm) ---
        with torch.no_grad():
            z_t = blk(torch.from_numpy(z_np))   # (N, out_d)
        z_np = z_t.cpu().numpy()

        # --- norm coverage (if this is not the last block) ---
        if i < len(model.norms):
            nm = model.norms[i]
            with torch.no_grad():
                a_t = nm(z_t)   # (N, dim)
            a_np = a_t.cpu().numpy()

            # Norm params snapshot
            shift_np = nm.shift.detach().cpu().numpy()
            scale_np = nm.scale.detach().cpu().numpy()

            # Segment counts per channel
            K_next = model.blocks[i + 1].K
            x_min_n = nm.x_min
            x_max_n = nm.x_max
            seg_counts = np.zeros((nm.dim, K_next), dtype=np.float64)
            for d in range(nm.dim):
                t = (np.clip(a_np[:, d], x_min_n, x_max_n - 1e-9) - x_min_n) \
                    / (x_max_n - x_min_n) * K_next
                k_idx = np.clip(t.astype(int), 0, K_next - 1)
                for ki in range(K_next):
                    seg_counts[d, ki] = (k_idx == ki).sum()

            # Uniformity per channel: 1 - normalised MAD from expected count
            expected = N / K_next
            uniformity = [
                float(max(0.0, 1.0 - abs(seg_counts[d] - expected).mean() / (expected + 1e-9)))
                for d in range(nm.dim)
            ]
            active_frac = [
                float((seg_counts[d] > 0).mean())
                for d in range(nm.dim)
            ]
            # Clamping fractions
            frac_lo = float((a_np < x_min_n + 1e-6).mean())
            frac_hi = float((a_np >= x_max_n - 1e-4).mean())

            norm_rep = NormCoverageReport(
                norm_index=i,
                dim=nm.dim,
                x_min=x_min_n,
                x_max=x_max_n,
                K=K_next,
                z_mean_per_channel=z_np.mean(axis=0).tolist(),
                z_std_per_channel= z_np.std(axis=0).tolist(),
                z_min_per_channel= z_np.min(axis=0).tolist(),
                z_max_per_channel= z_np.max(axis=0).tolist(),
                seg_counts_per_channel=[seg_counts[d].tolist() for d in range(nm.dim)],
                uniformity_per_channel=uniformity,
                active_segment_fraction_per_channel=active_frac,
                uniformity_mean=float(np.mean(uniformity)),
                active_segment_fraction_mean=float(np.mean(active_frac)),
                shift_snapshot=shift_np.tolist(),
                scale_snapshot=scale_np.tolist(),
                frac_clamped_lo=frac_lo,
                frac_clamped_hi=frac_hi,
            )
            norm_reports.append(norm_rep)
            z_np = a_np   # feed normalised activations to next block

    return StackCoverageReport(
        dims=model.dims,
        K=model.K,
        L=model.L,
        n_samples=N,
        block_reports=block_reports,
        norm_reports=norm_reports,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────

def stack_coverage_to_dict(report: StackCoverageReport) -> Dict:
    """Convert StackCoverageReport to a plain dict (JSON-serialisable)."""
    from .coverage import coverage_report_to_dict as _layer_to_d

    def blk_to_d(r: LayerCoverageReport) -> Dict:
        return {
            "name": r.name,
            "n_src": r.n_src, "n_dst": r.n_dst, "K": r.K, "L": r.L,
            "x_min": r.x_min, "x_max": r.x_max,
            "visited_fraction_mean": r.visited_fraction_mean,
            "effective_support_mean": r.effective_support_mean,
            "range_utilization_mean": r.range_utilization_mean,
            "gini_per_src": r.gini_per_src,
            "x_in_stats_per_src": r.x_in_stats_per_src,
        }

    def norm_to_d(r: NormCoverageReport) -> Dict:
        return {
            "norm_index": r.norm_index,
            "dim": r.dim,
            "x_min": r.x_min, "x_max": r.x_max, "K": r.K,
            "uniformity_mean": r.uniformity_mean,
            "active_segment_fraction_mean": r.active_segment_fraction_mean,
            "frac_clamped_lo": r.frac_clamped_lo,
            "frac_clamped_hi": r.frac_clamped_hi,
            "shift_snapshot": r.shift_snapshot,
            "scale_snapshot": r.scale_snapshot,
            "z_mean_per_channel": r.z_mean_per_channel,
            "z_std_per_channel": r.z_std_per_channel,
        }

    return {
        "dims": report.dims,
        "K": report.K, "L": report.L, "n_samples": report.n_samples,
        "blocks": [blk_to_d(b) for b in report.block_reports],
        "norms":  [norm_to_d(n) for n in report.norm_reports],
    }
