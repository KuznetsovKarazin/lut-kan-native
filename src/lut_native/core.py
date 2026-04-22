"""
LUT edge: differentiable segment-wise lookup table.

Design choices (stated explicitly because they matter for the science):

1. Sampling convention is HALF-OPEN inside each segment:
       x_grid[k, i] = x_min + k*seg_width + i * (seg_width/L),  i = 0..L-1
   i.e. lut[k, L-1] corresponds to x = (knot_{k+1} - step), NOT knot_{k+1}.
   This matches src/quant/lut_builder.py in the main lut-kan repo exactly.

   Consequence: lut[k, L-1] and lut[k+1, 0] are sampled at x values separated
   by (seg_width/L), so they SHOULD differ for any non-flat function.
   A naive "continuity" penalty ((lut[:-1,-1] - lut[1:,0])**2) is wrong here;
   see regularizers.py for the correctly-formulated version.

2. Forward mapping at runtime:
       t = (x - x_min) / seg_width           # in [0, K)
       k = clamp(floor(t), 0, K-1)            # segment index
       u = clamp(t - k, 0, 1)                 # in-segment coord in [0, 1]
       pos = u * (L - 1)                      # LUT index position
       r0 = clamp(floor(pos), 0, L-1)
       r1 = clamp(r0 + 1, 0, L-1)
       w  = pos - r0
       y  = (1 - w) * lut[k, r0] + w * lut[k, r1]

   Note: pos = u * (L-1) uses ENDPOINT-INCLUSIVE interpolation indices.
   This is asymmetric with the half-open sampling (point 1) but matches
   the main repo's runtime. Keeping both conventions exactly as-is so
   post-training LUT from main repo and direct-LUT from here are comparable
   byte-for-byte.

3. Forward is fully differentiable wrt lut values (gather + linear combine).
   k, r0, r1, u, w depend on x only, not on lut, so no index-gradients needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# NumPy reference forward (for testing & post-training LUT evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def lut_forward_numpy(
    x: np.ndarray,
    lut: np.ndarray,
    x_min: float = -1.0,
    x_max: float = 1.0,
) -> np.ndarray:
    """
    Segment-wise LUT forward with linear interpolation.

    Args:
        x: input array, any shape. Clipped to [x_min, x_max).
        lut: float array of shape (K, L).
        x_min, x_max: domain endpoints.

    Returns:
        Array with same shape as x, dtype float32.
    """
    K, L = lut.shape
    x = np.asarray(x, dtype=np.float32)
    # Clip to [x_min, x_max - eps] for half-open upper boundary
    hi = np.nextafter(np.float32(x_max), np.float32(-np.inf))
    x_clip = np.clip(x, np.float32(x_min), hi)
    seg_width = (x_max - x_min) / K
    t = (x_clip - x_min) / seg_width
    k = np.clip(np.floor(t).astype(np.int32), 0, K - 1)
    u = np.clip(t - k.astype(np.float32), 0.0, 1.0)
    pos = u * (L - 1)
    r0 = np.clip(np.floor(pos).astype(np.int32), 0, L - 1)
    r1 = np.clip(r0 + 1, 0, L - 1)
    w = pos - r0.astype(np.float32)
    v0 = lut[k, r0]
    v1 = lut[k, r1]
    return (v0 * (1.0 - w) + v1 * w).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch differentiable LUT edge
# ─────────────────────────────────────────────────────────────────────────────

class LUTEdge(nn.Module):
    """
    Trainable segment-wise LUT edge, shape (K, L), linear interpolation.

    Forward is bit-compatible with lut_forward_numpy above so that values
    trained here can be exported and evaluated identically with numpy kernel
    (or the main repo's C kernel).
    """

    def __init__(
        self,
        K: int,
        L: int,
        x_min: float = -1.0,
        x_max: float = 1.0,
    ):
        super().__init__()
        if K < 1 or L < 2:
            raise ValueError("Require K >= 1 and L >= 2")
        self.K = int(K)
        self.L = int(L)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        # Fixed-domain buffers; not trained.
        self.register_buffer("_x_min", torch.tensor(x_min, dtype=torch.float32))
        self.register_buffer("_x_max", torch.tensor(x_max, dtype=torch.float32))
        self.register_buffer(
            "_seg_width",
            torch.tensor((x_max - x_min) / K, dtype=torch.float32),
        )
        # The actual LUT parameter.
        self.lut = nn.Parameter(torch.zeros(self.K, self.L, dtype=torch.float32))

    # ---- initialization ----------------------------------------------------

    @torch.no_grad()
    def init_from_array(self, lut_values: np.ndarray) -> None:
        arr = np.asarray(lut_values, dtype=np.float32)
        if arr.shape != (self.K, self.L):
            raise ValueError(f"lut_values shape {arr.shape} != ({self.K}, {self.L})")
        self.lut.copy_(torch.from_numpy(arr))

    @torch.no_grad()
    def add_init_noise(self, std_absolute: float, generator: torch.Generator | None = None) -> None:
        """Add Gaussian noise with given absolute std to the current LUT."""
        if std_absolute <= 0:
            return
        noise = torch.empty_like(self.lut).normal_(generator=generator) * std_absolute
        self.lut.add_(noise)

    # ---- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass; differentiable wrt self.lut.

        Args:
            x: float tensor of any shape. Values outside [x_min, x_max) are clipped.

        Returns:
            Tensor of same shape as x.
        """
        # Clip to [x_min, x_max - eps] — use a fixed small offset because
        # torch.finfo(float32).eps is too small for this domain.
        hi = self._x_max - torch.tensor(1e-7, dtype=torch.float32, device=x.device)
        x_clip = torch.clamp(x, self._x_min, hi)
        t = (x_clip - self._x_min) / self._seg_width
        k = torch.clamp(torch.floor(t).long(), 0, self.K - 1)
        u = torch.clamp(t - k.to(t.dtype), 0.0, 1.0)
        pos = u * (self.L - 1)
        r0 = torch.clamp(torch.floor(pos).long(), 0, self.L - 1)
        r1 = torch.clamp(r0 + 1, 0, self.L - 1)
        w = pos - r0.to(pos.dtype)
        # Gather - this is the part that carries gradients to self.lut
        v0 = self.lut[k, r0]
        v1 = self.lut[k, r1]
        return v0 * (1.0 - w) + v1 * w

    # ---- utility ----------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"K={self.K}, L={self.L}, "
            f"x_min={self.x_min}, x_max={self.x_max}, "
            f"n_params={self.K * self.L}"
        )

    @property
    def memory_bytes_float32(self) -> int:
        return self.K * self.L * 4

    @property
    def memory_bytes_uint8(self) -> int:
        """Memory after per-segment uint8 quantization (matches main repo format)."""
        # q_table: K*L bytes + scale: K * 2 bytes (float16) + y_min: K * 2 bytes
        return self.K * self.L + self.K * 4
