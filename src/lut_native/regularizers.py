"""
Regularizers on LUT values.

All take a tensor of shape (K, L) and return a scalar loss term.

Scales / units note:
  - first_diff, second_diff, boundary return `.mean()` across all relevant
    pairs/triplets, so their magnitude is comparable across L.
  - Applying them at the same lambda across different L values is reasonable
    but not exact; see docs/METHODOLOGY.md for a discussion.

On the "boundary continuity" penalty:
  The main lut-kan repo samples LUT at half-open grid points inside each
  segment: lut[k, i] = f(x_min + k*seg_width + i*(seg_width/L)), i = 0..L-1.
  Therefore lut[k, L-1] corresponds to x = knot_{k+1} - (seg_width/L), not
  to the knot itself. lut[k+1, 0] corresponds to x = knot_{k+1} exactly.
  The two adjacent samples are separated by (seg_width/L).

  A naive penalty ((lut[:-1, -1] - lut[1:, 0])**2).mean() would therefore
  push the LUT toward a function that is constant across segment boundaries,
  which for any non-flat target is wrong by construction.

  The correct extrapolation-based boundary penalty uses the LAST-SEGMENT
  slope to predict where the function should be at the boundary, then
  compares to lut[k+1, 0]:

      slope_k_right = lut[k, L-1] - lut[k, L-2]        # per-sample slope
      predicted_at_knot = lut[k, L-1] + slope_k_right  # extrapolate by one step
      penalty_k = (predicted_at_knot - lut[k+1, 0])**2

  This accounts for the half-open sampling and penalizes genuine discontinuity.
"""

from __future__ import annotations

import torch


def first_diff_penalty(lut: torch.Tensor) -> torch.Tensor:
    """Mean squared first difference within each segment, along L axis.

    Args:
        lut: (K, L) tensor.
    """
    d1 = lut[:, 1:] - lut[:, :-1]
    return (d1 ** 2).mean()


def second_diff_penalty(lut: torch.Tensor) -> torch.Tensor:
    """Mean squared second difference (curvature proxy) along L axis.

    Args:
        lut: (K, L) tensor.
    """
    d2 = lut[:, 2:] - 2.0 * lut[:, 1:-1] + lut[:, :-2]
    return (d2 ** 2).mean()


def boundary_continuity_penalty(lut: torch.Tensor) -> torch.Tensor:
    """Penalize discontinuity at segment boundaries, correctly accounting for
    the half-open sampling convention.

    Extrapolates lut[k, L-1] forward by one L-step using the last in-segment
    slope, and compares to lut[k+1, 0]:

        predicted_at_knot = lut[k, L-1] + (lut[k, L-1] - lut[k, L-2])
        penalty = (predicted_at_knot - lut[k+1, 0])**2

    Args:
        lut: (K, L) tensor. Requires K >= 2 and L >= 2.
    """
    if lut.shape[0] < 2:
        return torch.zeros((), dtype=lut.dtype, device=lut.device)
    last = lut[:-1, -1]                      # (K-1,)
    second_last = lut[:-1, -2]               # (K-1,)
    slope = last - second_last               # (K-1,) per-sample slope in last pair
    predicted = last + slope                 # extrapolate one step to the knot
    first_next = lut[1:, 0]                  # (K-1,) value at left edge of next segment
    gap = predicted - first_next
    return (gap ** 2).mean()


def boundary_slope_penalty(lut: torch.Tensor) -> torch.Tensor:
    """Penalize slope discontinuity at segment boundaries.

    Compares the rightmost in-segment slope of segment k to the leftmost
    in-segment slope of segment k+1:

        right_slope_k  = lut[k, L-1]  - lut[k, L-2]
        left_slope_k+1 = lut[k+1, 1]  - lut[k+1, 0]
        penalty = (right_slope_k - left_slope_k+1)**2

    Args:
        lut: (K, L) tensor. Requires K >= 2 and L >= 2.
    """
    if lut.shape[0] < 2:
        return torch.zeros((), dtype=lut.dtype, device=lut.device)
    right_slope = lut[:-1, -1] - lut[:-1, -2]   # (K-1,)
    left_slope = lut[1:, 1] - lut[1:, 0]        # (K-1,)
    return ((right_slope - left_slope) ** 2).mean()


def combined_penalty(
    lut: torch.Tensor,
    lambda_1: float = 0.0,
    lambda_2: float = 0.0,
    lambda_bv: float = 0.0,
    lambda_bs: float = 0.0,
) -> torch.Tensor:
    """Sum of regularization terms with the given weights.

    Args:
        lut: (K, L)
        lambda_1: weight for first_diff_penalty
        lambda_2: weight for second_diff_penalty
        lambda_bv: weight for boundary_continuity_penalty (value continuity)
        lambda_bs: weight for boundary_slope_penalty (slope continuity)
    """
    loss = torch.zeros((), dtype=lut.dtype, device=lut.device)
    if lambda_1 > 0:
        loss = loss + lambda_1 * first_diff_penalty(lut)
    if lambda_2 > 0:
        loss = loss + lambda_2 * second_diff_penalty(lut)
    if lambda_bv > 0:
        loss = loss + lambda_bv * boundary_continuity_penalty(lut)
    if lambda_bs > 0:
        loss = loss + lambda_bs * boundary_slope_penalty(lut)
    return loss
