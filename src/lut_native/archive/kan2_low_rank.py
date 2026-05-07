"""
Low-rank residual LUT-KAN2.

Parameterizes the delta as:
    delta[i, h, k, l] = sum_r U[i, h, k, r] * V[i, h, r, l]

i.e. each edge (i, h)'s (K, L) delta matrix has rank <= rank_r. U and V
are learned parameters; delta is reconstructed on-the-fly.

Motivation (M5a hypothesis):
    M4a showed unstructured trust-region (small alpha on full-rank delta)
    does not change best-val MSE. Maybe structured constraint works better:
    force the delta to be low-rank, letting the optimizer only explore
    globally-coherent perturbations of the LUT, not per-cell noise.

Parameter count comparison (per edge, K=16, L=32):
    full delta:       K * L = 512
    rank-1 delta:     K + L = 48    (90% reduction)
    rank-2 delta:     2(K+L) = 96   (81% reduction)
    rank-4 delta:     4(K+L) = 192  (63% reduction)
    rank-8 delta:     8(K+L) = 384  (25% reduction)
    rank-16 delta:    16(K+L) = 768 (actually MORE than full — no point)

So useful range is r ∈ {1, 2, 4, 8}. "full" rank = min(K, L) is handled by
falling back to the existing ResidualLUTKAN2Layer.

Design notes:
- U is initialized to zeros, V to small random. This keeps initial delta
  at zero (matching ResidualLUTKAN2Layer's init convention), so
  poly-init MSE is preserved at step 0. Any init where U or V is exactly
  zero gives delta=0 and unblocks gradient flow through the other factor.
- We don't parameterize the alpha into U or V; we keep alpha as a
  separate scalar (same semantics as ResidualLUTKAN2Layer).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class LowRankResidualLUTKAN2Layer(nn.Module):
    """Residual LUT-KAN2 where delta_l1, delta_l2 are low-rank per-edge.

    For each edge (i, h) of layer 1, delta is a (K, L) matrix reconstructed
    from U[i, h] (K, r1) and V[i, h] (r1, L): delta[i, h] = U[i, h] @ V[i, h].
    Same for layer 2 edges (h, o) with its own rank r2.

    When rank_l1 or rank_l2 equals min(K, L), there is no reduction vs
    full-rank delta; use ResidualLUTKAN2Layer directly instead.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        K: int,
        L: int,
        alpha: float,
        rank_l1: int,
        rank_l2: int,
        x_min: float = -1.0,
        x_max: float = 1.0,
        activation: str = "tanh",
        x_min_l2: float = -1.0,
        x_max_l2: float = 1.0,
        init_scale: float = 0.01,
    ):
        super().__init__()
        if activation not in ("tanh", "zscore"):
            raise ValueError("activation must be 'tanh' or 'zscore'")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        max_rank = min(K, L)
        if not (1 <= rank_l1 <= max_rank):
            raise ValueError(f"rank_l1 must be in [1, {max_rank}], got {rank_l1}")
        if not (1 <= rank_l2 <= max_rank):
            raise ValueError(f"rank_l2 must be in [1, {max_rank}], got {rank_l2}")

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.K = int(K)
        self.L = int(L)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.alpha = float(alpha)
        self.rank_l1 = int(rank_l1)
        self.rank_l2 = int(rank_l2)
        self.activation = activation

        if activation == "tanh":
            self.x_min_l2 = -1.0
            self.x_max_l2 = 1.0
        else:
            if not (x_min_l2 < x_max_l2):
                raise ValueError("require x_min_l2 < x_max_l2")
            self.x_min_l2 = float(x_min_l2)
            self.x_max_l2 = float(x_max_l2)

        # Frozen init
        self.register_buffer(
            "lut_l1_init",
            torch.zeros(in_dim, hidden_dim, K, L, dtype=torch.float32),
        )
        self.register_buffer(
            "lut_l2_init",
            torch.zeros(hidden_dim, out_dim, K, L, dtype=torch.float32),
        )

        # Low-rank factors. U is zero-init so initial delta = 0.
        # V has small random init so gradients can flow through both factors
        # from step 1 (if V were also zero, gradient through U would be 0).
        self.U_l1 = nn.Parameter(
            torch.zeros(in_dim, hidden_dim, K, rank_l1, dtype=torch.float32)
        )
        self.V_l1 = nn.Parameter(
            torch.randn(in_dim, hidden_dim, rank_l1, L, dtype=torch.float32) * init_scale
        )
        self.U_l2 = nn.Parameter(
            torch.zeros(hidden_dim, out_dim, K, rank_l2, dtype=torch.float32)
        )
        self.V_l2 = nn.Parameter(
            torch.randn(hidden_dim, out_dim, rank_l2, L, dtype=torch.float32) * init_scale
        )

        # Standard buffers
        self.register_buffer("_x_min_l1", torch.tensor(x_min, dtype=torch.float32))
        self.register_buffer("_x_max_l1", torch.tensor(x_max, dtype=torch.float32))
        self.register_buffer("_seg_width_l1",
                             torch.tensor((x_max - x_min) / K, dtype=torch.float32))
        self.register_buffer("_x_min_l2",
                             torch.tensor(self.x_min_l2, dtype=torch.float32))
        self.register_buffer("_x_max_l2",
                             torch.tensor(self.x_max_l2, dtype=torch.float32))
        self.register_buffer("_seg_width_l2",
                             torch.tensor((self.x_max_l2 - self.x_min_l2) / K,
                                          dtype=torch.float32))
        self.register_buffer("_z_mean", torch.zeros(hidden_dim, dtype=torch.float32))
        self.register_buffer("_z_std", torch.ones(hidden_dim, dtype=torch.float32))
        self.register_buffer("_calibrated", torch.tensor(False))

    # ---- effective LUT views ----------------------------------------------

    @property
    def delta_l1(self) -> torch.Tensor:
        """Reconstruct delta from U @ V. Shape (in_dim, hidden_dim, K, L)."""
        # U_l1: (in_dim, hidden_dim, K, rank_l1)
        # V_l1: (in_dim, hidden_dim, rank_l1, L)
        # matmul over last two dims: einsum ihkr,ihrl->ihkl
        return torch.einsum("ihkr,ihrl->ihkl", self.U_l1, self.V_l1)

    @property
    def delta_l2(self) -> torch.Tensor:
        return torch.einsum("hokr,horl->hokl", self.U_l2, self.V_l2)

    @property
    def lut_l1(self) -> torch.Tensor:
        return self.lut_l1_init + self.alpha * self.delta_l1

    @property
    def lut_l2(self) -> torch.Tensor:
        return self.lut_l2_init + self.alpha * self.delta_l2

    # ---- initialization ---------------------------------------------------

    @torch.no_grad()
    def init_layer1_from_arrays(self, luts: np.ndarray) -> None:
        expected = (self.in_dim, self.hidden_dim, self.K, self.L)
        if luts.shape != expected:
            raise ValueError(f"Expected {expected}, got {luts.shape}")
        self.lut_l1_init.copy_(torch.from_numpy(luts.astype(np.float32)))
        # Reset factors: U=0 makes delta=0
        self.U_l1.zero_()

    @torch.no_grad()
    def init_layer2_from_arrays(self, luts: np.ndarray) -> None:
        expected = (self.hidden_dim, self.out_dim, self.K, self.L)
        if luts.shape != expected:
            raise ValueError(f"Expected {expected}, got {luts.shape}")
        self.lut_l2_init.copy_(torch.from_numpy(luts.astype(np.float32)))
        self.U_l2.zero_()

    # ---- forward ----------------------------------------------------------

    def _edge_forward_bulk(
        self,
        x: torch.Tensor,
        lut_block: torch.Tensor,
        x_min_t: torch.Tensor,
        x_max_t: torch.Tensor,
        seg_width_t: torch.Tensor,
    ) -> torch.Tensor:
        N = x.shape[0]
        n_src, n_dst, K, L = lut_block.shape
        hi = x_max_t - torch.tensor(1e-7, dtype=x.dtype, device=x.device)
        x_clip = torch.clamp(x, x_min_t, hi)
        t = (x_clip - x_min_t) / seg_width_t
        k = torch.clamp(torch.floor(t).long(), 0, K - 1)
        u = torch.clamp(t - k.to(t.dtype), 0.0, 1.0)
        pos = u * (L - 1)
        r0 = torch.clamp(torch.floor(pos).long(), 0, L - 1)
        r1 = torch.clamp(r0 + 1, 0, L - 1)
        w = pos - r0.to(pos.dtype)
        src_idx = torch.arange(n_src, device=x.device).view(1, n_src).expand(N, n_src)
        v0 = lut_block[src_idx, :, k, r0]
        v1 = lut_block[src_idx, :, k, r1]
        w_ = w.unsqueeze(-1)
        edges_out = v0 * (1.0 - w_) + v1 * w_
        return edges_out.sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        z = self._edge_forward_bulk(
            x, self.lut_l1, self._x_min_l1, self._x_max_l1, self._seg_width_l1,
        )
        if self.activation == "tanh":
            a = torch.tanh(z)
        else:
            a = (z - self._z_mean) / self._z_std
        y = self._edge_forward_bulk(
            a, self.lut_l2, self._x_min_l2, self._x_max_l2, self._seg_width_l2,
        )
        return y

    # ---- calibration (for zscore compatibility) --------------------------

    @torch.no_grad()
    def calibrate_activation_stats(self, x_train: torch.Tensor, eps: float = 1e-6) -> dict:
        if self.activation != "zscore":
            raise RuntimeError("calibrate only with activation='zscore'")
        if x_train.dim() == 1:
            x_train = x_train.unsqueeze(-1)
        z = self._edge_forward_bulk(
            x_train, self.lut_l1, self._x_min_l1, self._x_max_l1, self._seg_width_l1,
        )
        mean = z.mean(dim=0)
        std = z.std(dim=0, unbiased=False).clamp(min=eps)
        self._z_mean.copy_(mean)
        self._z_std.copy_(std)
        self._calibrated.fill_(True)
        return {"z_mean": mean.cpu().numpy().copy(),
                "z_std": std.cpu().numpy().copy()}

    # ---- utility ----------------------------------------------------------

    @torch.no_grad()
    def effective_lut_l1_numpy(self) -> np.ndarray:
        return self.lut_l1.detach().cpu().numpy().copy()

    @torch.no_grad()
    def effective_lut_l2_numpy(self) -> np.ndarray:
        return self.lut_l2.detach().cpu().numpy().copy()

    @torch.no_grad()
    def delta_norms(self) -> dict:
        d1 = self.delta_l1
        d2 = self.delta_l2
        return {
            "delta_l1_l2": float(d1.pow(2).sum().sqrt().item()),
            "delta_l2_l2": float(d2.pow(2).sum().sqrt().item()),
            "delta_l1_linf": float(d1.abs().max().item()),
            "delta_l2_linf": float(d2.abs().max().item()),
        }

    def n_delta_params(self) -> int:
        """Total trainable parameters for the two deltas."""
        n = 0
        n += self.U_l1.numel() + self.V_l1.numel()
        n += self.U_l2.numel() + self.V_l2.numel()
        return n
