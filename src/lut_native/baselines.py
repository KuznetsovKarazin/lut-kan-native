"""
Baselines: the two methods direct-LUT training competes against.

1. Chebyshev polynomial fit via least-squares.
2. Post-training LUT: sample the fitted polynomial on the LUT grid,
   then per-segment uint8 quantization. This is the v2.1 pipeline from
   the main lut-kan repository.

Both methods are NumPy-only and do not depend on PyTorch.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Chebyshev polynomial baseline
# ─────────────────────────────────────────────────────────────────────────────

def chebyshev_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """Chebyshev T_0..T_degree via recurrence. Returns (N, degree+1)."""
    x = np.asarray(x, dtype=np.float32).ravel()
    N = x.shape[0]
    T = np.zeros((N, degree + 1), dtype=np.float32)
    T[:, 0] = 1.0
    if degree >= 1:
        T[:, 1] = x
    for n in range(2, degree + 1):
        T[:, n] = 2.0 * x * T[:, n - 1] - T[:, n - 2]
    return T


def fit_chebyshev_ls(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    ridge: float = 1e-6,
    x_min: float = -1.0,
    x_max: float = 1.0,
) -> np.ndarray:
    """Least-squares fit of Chebyshev polynomial of given degree.

    Normalizes x to [-1, 1] internally (Chebyshev's natural domain).
    Returns coefficients of shape (degree+1,).
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    xn = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    xn = np.clip(xn, -1.0, 1.0)
    T = chebyshev_basis(xn, degree)
    A = T.T @ T + ridge * np.eye(degree + 1, dtype=np.float32)
    b = T.T @ y
    return np.linalg.solve(A, b).astype(np.float32)


def eval_chebyshev(
    x: np.ndarray,
    coeffs: np.ndarray,
    x_min: float = -1.0,
    x_max: float = 1.0,
) -> np.ndarray:
    """Evaluate Chebyshev polynomial with given coefficients."""
    xn = 2.0 * (np.asarray(x, dtype=np.float32) - x_min) / (x_max - x_min) - 1.0
    xn = np.clip(xn, -1.0, 1.0)
    T = chebyshev_basis(xn, len(coeffs) - 1)
    return (T @ coeffs.astype(np.float32)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Post-training LUT construction (v2.1 pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def sample_polynomial_to_lut(
    coeffs: np.ndarray,
    K: int,
    L: int,
    x_min: float = -1.0,
    x_max: float = 1.0,
) -> np.ndarray:
    """
    Build a float LUT of shape (K, L) by sampling the Chebyshev polynomial
    on the standard half-open segment grid.

    Matches src/quant/lut_builder.py::build_segment_grid in the main repo:
        grid[k, i] = x_min + k*seg_width + i*(seg_width/L),  i = 0..L-1
    """
    lut = np.empty((K, L), dtype=np.float32)
    seg_width = (x_max - x_min) / K
    for k in range(K):
        a = x_min + k * seg_width
        step = seg_width / L
        xs = (a + step * np.arange(L)).astype(np.float32)
        lut[k, :] = eval_chebyshev(xs, coeffs, x_min=x_min, x_max=x_max)
    return lut


def quantize_lut_uint8_asym(
    lut_float: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-segment asymmetric uint8 quantization.
      scale_k = (max_k - min_k) / 255
      q_ki    = round((lut_ki - min_k) / scale_k),  clipped to [0, 255]

    Returns:
        q_table: uint8 (K, L)
        scale:   float32 (K,)
        y_min:   float32 (K,)
    """
    K, L = lut_float.shape
    y_min = lut_float.min(axis=1).astype(np.float32)
    y_max = lut_float.max(axis=1).astype(np.float32)
    span = np.maximum(y_max - y_min, np.float32(1e-10))
    scale = (span / np.float32(255.0)).astype(np.float32)
    q = np.rint((lut_float - y_min[:, None]) / scale[:, None]).astype(np.int32)
    q = np.clip(q, 0, 255).astype(np.uint8)
    return q, scale, y_min


def dequantize_lut(
    q: np.ndarray,
    scale: np.ndarray,
    y_min: np.ndarray,
) -> np.ndarray:
    """Inverse of quantize_lut_uint8_asym."""
    return (y_min[:, None] + scale[:, None] * q.astype(np.float32)).astype(np.float32)
