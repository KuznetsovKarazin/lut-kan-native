"""
Diagnostics for LUT analysis.

1. High-frequency energy: is the learned LUT smooth or jagged?
2. Cell liveness: how many LUT cells actually moved during training?
3. Effective rank: how many independent DOF is the LUT really using?
   This matters because if the LUT collapses to ~deg+1 independent modes,
   the advantage over polynomial fit is structural, not representational.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MSE with paired-seed bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def paired_bootstrap_ci(
    mse_a: np.ndarray,
    mse_b: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap CI on log-MSE ratio (A/B).

    Both arrays must be same length (seeds paired across regimes).
    Returns mean log-ratio and [lower, upper] CI bounds.
    """
    if mse_a.shape != mse_b.shape:
        raise ValueError("mse_a and mse_b must have identical shape (paired seeds)")
    rng = np.random.RandomState(seed)
    N = len(mse_a)
    log_ratio = np.log(mse_a) - np.log(mse_b)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, N, size=N)
        boot_means[i] = log_ratio[idx].mean()
    lo = np.quantile(boot_means, (1 - ci) / 2)
    hi = np.quantile(boot_means, 1 - (1 - ci) / 2)
    return {
        "log_ratio_mean": float(log_ratio.mean()),
        "log_ratio_ci_lower": float(lo),
        "log_ratio_ci_upper": float(hi),
        "ratio_mean": float(np.exp(log_ratio.mean())),
        "ratio_ci_lower": float(np.exp(lo)),
        "ratio_ci_upper": float(np.exp(hi)),
        "n_seeds": int(N),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LUT shape diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def high_freq_energy(lut: np.ndarray, frac_keep_low: float = 0.25) -> float:
    """Fraction of FFT power in the HIGH portion of the spectrum, averaged
    over segments. Higher = jaggier.

    Args:
        lut: (K, L)
        frac_keep_low: fraction of low-frequency bins considered "low".
                       Default 0.25 means bins [0.25*nbins, nbins) count as "high".
    """
    K, L = lut.shape
    centered = lut - lut.mean(axis=1, keepdims=True)  # remove DC per segment
    F = np.fft.rfft(centered, axis=1)
    power = (np.abs(F) ** 2).mean(axis=0)  # avg over segments
    total = power.sum() + 1e-12
    n = power.shape[0]
    cutoff = int(np.ceil(n * frac_keep_low))
    return float(power[cutoff:].sum() / total)


def cell_liveness(
    cell_change: np.ndarray,
    lut_range: float,
    threshold_rel: float = 0.01,
) -> Dict[str, float]:
    """How many LUT cells moved more than `threshold_rel` * value_range?

    Args:
        cell_change: (K, L), elementwise |lut_final - lut_init|
        lut_range: max - min of the init LUT (for relative threshold)
        threshold_rel: fraction of range defining "moved"

    Returns:
        dict with 'live_fraction', 'mean_abs_change', 'max_abs_change'.
    """
    thr = threshold_rel * lut_range
    return {
        "live_fraction": float((cell_change > thr).mean()),
        "mean_abs_change": float(cell_change.mean()),
        "max_abs_change": float(cell_change.max()),
        "threshold_absolute": float(thr),
    }


def effective_rank(lut: np.ndarray, energy_frac: float = 0.99) -> int:
    """Number of singular values needed to explain `energy_frac` of total energy.

    This is the "effective number of independent curve shapes" across segments.
    For a polynomial of degree d sampled on K segments, this should be ~min(d+1, K).
    """
    # Remove per-segment mean before SVD: we want shape complexity, not offsets
    centered = lut - lut.mean(axis=1, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    total = (s ** 2).sum() + 1e-12
    cum = np.cumsum(s ** 2) / total
    return int(np.searchsorted(cum, energy_frac) + 1)


def singular_value_spectrum(lut: np.ndarray) -> np.ndarray:
    """Return singular values of (lut - per-segment-mean), sorted descending."""
    centered = lut - lut.mean(axis=1, keepdims=True)
    return np.linalg.svd(centered, compute_uv=False)
