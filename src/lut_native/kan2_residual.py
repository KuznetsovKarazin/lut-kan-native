"""
Residual-parameterized two-layer LUT-KAN.

Architecture:
    LUT_layer = LUT_init + alpha * Delta_layer

where:
    - LUT_init_l1, LUT_init_l2 are frozen (registered as buffers) — the
      polynomial-sampled initialization
    - Delta_l1, Delta_l2 are nn.Parameters, initialized to zeros
    - alpha is a scalar hyperparameter controlling trust-region size.
      alpha=0     -> no deviation allowed (training does nothing)
      alpha=0.1   -> small trust region
      alpha=1.0   -> equivalent to unconstrained direct-LUT
      alpha=inf   -> not supported; just use LUTKAN2Layer directly

Forward math is identical to LUTKAN2Layer; only the source of the LUT
tensors differs. This means the numpy reference (kan2_forward_numpy) and
coverage diagnostics work unchanged as long as we flatten back to
(LUT_init + alpha * delta) at evaluation time.

Two design choices explained:

1. Why a separate class rather than `@property`-hacking LUTKAN2Layer?
   nn.Module parameter registration happens in __setattr__. A @property
   cannot be both a Parameter (for optimizer) and a computed tensor. Keeping
   a parallel class keeps behavior explicit.

2. Why alpha is a plain float, not a Parameter?
   This is M4a: we want to isolate the "trust region" effect. A learnable
   alpha would let the optimizer open the trust region on its own, which
   would confound the ablation. Future work could add a learnable alpha.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class ResidualLUTKAN2Layer(nn.Module):
    """Residual-parameterized two-layer LUT-KAN.

    Effective LUT = LUT_init + alpha * delta, where delta starts at zero.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        K: int,
        L: int,
        alpha: float,
        x_min: float = -1.0,
        x_max: float = 1.0,
        activation: str = "tanh",
        x_min_l2: float = -1.0,
        x_max_l2: float = 1.0,
    ):
        super().__init__()
        if activation not in ("tanh", "zscore"):
            raise ValueError(f"activation must be 'tanh' or 'zscore'")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.K = int(K)
        self.L = int(L)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.alpha = float(alpha)
        self.activation = activation

        if activation == "tanh":
            self.x_min_l2 = -1.0
            self.x_max_l2 = 1.0
        else:
            if not (x_min_l2 < x_max_l2):
                raise ValueError("require x_min_l2 < x_max_l2")
            self.x_min_l2 = float(x_min_l2)
            self.x_max_l2 = float(x_max_l2)

        # Frozen init buffers
        self.register_buffer(
            "lut_l1_init",
            torch.zeros(in_dim, hidden_dim, K, L, dtype=torch.float32),
        )
        self.register_buffer(
            "lut_l2_init",
            torch.zeros(hidden_dim, out_dim, K, L, dtype=torch.float32),
        )

        # Learnable deltas, start at zero
        self.delta_l1 = nn.Parameter(
            torch.zeros(in_dim, hidden_dim, K, L, dtype=torch.float32)
        )
        self.delta_l2 = nn.Parameter(
            torch.zeros(hidden_dim, out_dim, K, L, dtype=torch.float32)
        )

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

    # ---- effective LUT views (read-only) ---------------------------------

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
        self.delta_l1.zero_()

    @torch.no_grad()
    def init_layer2_from_arrays(self, luts: np.ndarray) -> None:
        expected = (self.hidden_dim, self.out_dim, self.K, self.L)
        if luts.shape != expected:
            raise ValueError(f"Expected {expected}, got {luts.shape}")
        self.lut_l2_init.copy_(torch.from_numpy(luts.astype(np.float32)))
        self.delta_l2.zero_()

    # ---- forward ---------------------------------------------------------

    def _edge_forward_bulk(
        self,
        x: torch.Tensor,
        lut_block: torch.Tensor,
        x_min_t: torch.Tensor,
        x_max_t: torch.Tensor,
        seg_width_t: torch.Tensor,
    ) -> torch.Tensor:
        # Copy of LUTKAN2Layer._edge_forward_bulk; duplicated here so we
        # don't have to subclass.
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

    # ---- calibration -----------------------------------------------------

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
        a = (z - mean) / std
        frac_clipped = float(((a < self._x_min_l2) | (a >= self._x_max_l2)).float().mean())
        return {"z_mean": mean.cpu().numpy().copy(),
                "z_std": std.cpu().numpy().copy(),
                "frac_clipped_after_zscore": frac_clipped}

    # ---- utility ---------------------------------------------------------

    @torch.no_grad()
    def effective_lut_l1_numpy(self) -> np.ndarray:
        return (self.lut_l1_init + self.alpha * self.delta_l1).cpu().numpy().copy()

    @torch.no_grad()
    def effective_lut_l2_numpy(self) -> np.ndarray:
        return (self.lut_l2_init + self.alpha * self.delta_l2).cpu().numpy().copy()

    @torch.no_grad()
    def delta_norms(self) -> dict:
        """Track how far the delta has drifted from zero. For diagnostics."""
        return {
            "delta_l1_l2": float(self.delta_l1.detach().pow(2).sum().sqrt().item()),
            "delta_l2_l2": float(self.delta_l2.detach().pow(2).sum().sqrt().item()),
            "delta_l1_linf": float(self.delta_l1.detach().abs().max().item()),
            "delta_l2_linf": float(self.delta_l2.detach().abs().max().item()),
        }
