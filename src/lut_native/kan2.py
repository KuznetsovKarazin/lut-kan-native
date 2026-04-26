"""
Two-layer KAN with LUT edges.

Architecture: [in_dim -> hidden_dim -> out_dim]

Layer 1 (in_dim -> hidden_dim):
  - For each (i, h) pair: one LUTEdge phi_{i,h}
  - Hidden unit h: z_h = sum_i phi_{i,h}(x_i)
  - Total layer-1 edges: in_dim * hidden_dim

Activation between layers:
  - tanh squash on z_h to bring into [-1, 1] for layer 2's LUTs.
  - Alternative: learnable affine then clamp; we use plain tanh for simplicity.

Layer 2 (hidden_dim -> out_dim):
  - For each (h, o) pair: one LUTEdge psi_{h,o}
  - Output o: y_o = sum_h psi_{h,o}(tanh(z_h))
  - Total layer-2 edges: hidden_dim * out_dim

All edges are independent LUTEdge instances (no parameter sharing).

Design choice (matches paper & main repo):
  - Knots are shared across edges (same segmentation on same domain).
  - Each edge has its own q_table/scale/y_min at deployment time.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from .core import LUTEdge, lut_forward_numpy


class LUTKAN2Layer(nn.Module):
    """
    Two-layer LUT-KAN: [in_dim] -> [hidden_dim] -> [out_dim].

    Forward pass:
        z_h = sum_i edge_L1[i, h](x_i)                  (summed across inputs)
        a_h = tanh(z_h)                                 (squash to layer-2 domain)
        y_o = sum_h edge_L2[h, o](a_h)                  (summed across hidden)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        K: int,
        L: int,
        x_min: float = -1.0,
        x_max: float = 1.0,
        activation: str = "tanh",
        x_min_l2: float = -1.0,
        x_max_l2: float = 1.0,
    ):
        """
        activation: 'tanh' (default, matches original) or 'zscore'.
            - 'tanh':   layer-2 input = tanh(z), so layer-2 domain is fixed to
                        [-1, 1] regardless of x_min_l2/x_max_l2 args.
            - 'zscore': layer-2 input = (z - z_mean) / z_std, where z_mean,
                        z_std are buffers set by calibrate_activation_stats().
                        Layer-2 LUT domain should span several sigma; a common
                        choice is x_min_l2=-3, x_max_l2=+3 (covers ~99.7% of
                        N(0,1) inputs).

        Z-score inputs that fall outside [x_min_l2, x_max_l2] are clipped
        (same half-open convention as every other LUT in this code).
        """
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.K = int(K)
        self.L = int(L)
        self.x_min = float(x_min)
        self.x_max = float(x_max)

        if activation not in ("tanh", "zscore"):
            raise ValueError(f"activation must be 'tanh' or 'zscore', got {activation}")
        self.activation = activation
        # For 'tanh', L2 domain is fixed to [-1, 1] regardless of the args
        # (because tanh output is bounded). For 'zscore', caller picks.
        if activation == "tanh":
            self.x_min_l2 = -1.0
            self.x_max_l2 = 1.0
        else:
            if not (x_min_l2 < x_max_l2):
                raise ValueError("require x_min_l2 < x_max_l2")
            self.x_min_l2 = float(x_min_l2)
            self.x_max_l2 = float(x_max_l2)

        # Layer 1 edges: (in_dim, hidden_dim) grid
        # Store LUT values as one big tensor of shape (in_dim, hidden_dim, K, L)
        # so training is a single optimizer update per batch.
        self.lut_l1 = nn.Parameter(
            torch.zeros(in_dim, hidden_dim, K, L, dtype=torch.float32)
        )
        # Layer 2 edges: (hidden_dim, out_dim)
        self.lut_l2 = nn.Parameter(
            torch.zeros(hidden_dim, out_dim, K, L, dtype=torch.float32)
        )

        self.register_buffer("_x_min_l1", torch.tensor(x_min, dtype=torch.float32))
        self.register_buffer("_x_max_l1", torch.tensor(x_max, dtype=torch.float32))
        self.register_buffer("_seg_width_l1", torch.tensor((x_max - x_min) / K, dtype=torch.float32))
        self.register_buffer("_x_min_l2", torch.tensor(self.x_min_l2, dtype=torch.float32))
        self.register_buffer("_x_max_l2", torch.tensor(self.x_max_l2, dtype=torch.float32))
        self.register_buffer("_seg_width_l2",
                             torch.tensor((self.x_max_l2 - self.x_min_l2) / K, dtype=torch.float32))

        # Calibration buffers for z-score normalization.
        # Initialized to identity (mean=0, std=1) so an uncalibrated zscore
        # model behaves like "no normalization" until calibrate() is called.
        self.register_buffer("_z_mean", torch.zeros(hidden_dim, dtype=torch.float32))
        self.register_buffer("_z_std", torch.ones(hidden_dim, dtype=torch.float32))
        self.register_buffer("_calibrated", torch.tensor(False))

    # ---- initialization ---------------------------------------------------

    @torch.no_grad()
    def init_layer1_from_arrays(self, luts: np.ndarray) -> None:
        """luts shape: (in_dim, hidden_dim, K, L)."""
        expected = (self.in_dim, self.hidden_dim, self.K, self.L)
        if luts.shape != expected:
            raise ValueError(f"Expected shape {expected}, got {luts.shape}")
        self.lut_l1.copy_(torch.from_numpy(luts.astype(np.float32)))

    @torch.no_grad()
    def init_layer2_from_arrays(self, luts: np.ndarray) -> None:
        """luts shape: (hidden_dim, out_dim, K, L)."""
        expected = (self.hidden_dim, self.out_dim, self.K, self.L)
        if luts.shape != expected:
            raise ValueError(f"Expected shape {expected}, got {luts.shape}")
        self.lut_l2.copy_(torch.from_numpy(luts.astype(np.float32)))

    @torch.no_grad()
    def add_init_noise(self, std_absolute: float) -> None:
        if std_absolute <= 0:
            return
        self.lut_l1.add_(torch.randn_like(self.lut_l1) * std_absolute)
        self.lut_l2.add_(torch.randn_like(self.lut_l2) * std_absolute)

    @torch.no_grad()
    def cheby_init(self, scale: float = 1.5, noise: float = 0.05) -> None:
        """
        Initialise both LUT layers with Chebyshev polynomial basis functions.

        Fixes the dead-init failure: with noise std=0.05 only 4/16 LUT
        segments are active in layer 2, causing the model to predict a
        constant (MSE ≈ Var(y)) for all of training.

        Sets lut[si, di, k, r] = T_{di}(x_val) × scale / in_dim for each
        cell, then adds small per-cell Gaussian noise for diversity.
        """
        import numpy as np

        def _fill(lut_param, in_dim, out_dim, K, L, x_min=-1.0, x_max=1.0):
            seg_w = (x_max - x_min) / K
            for si in range(in_dim):
                for di in range(out_dim):
                    degree = di % max(1, out_dim)
                    for k in range(K):
                        for r in range(L):
                            xv = x_min + (k + r / (L - 1)) * seg_w
                            xn = 2.0 * (xv - x_min) / (x_max - x_min) - 1.0
                            xn = float(np.clip(xn, -1 + 1e-6, 1 - 1e-6))
                            tk = float(np.cos(degree * np.arccos(xn)))
                            lut_param[si, di, k, r] = tk * scale / in_dim
            if noise > 0:
                lut_param.add_(torch.randn_like(lut_param) * noise)

        in_dim, hidden_dim, K, L = self.lut_l1.shape
        hidden_dim2, out_dim, K2, L2 = self.lut_l2.shape
        _fill(self.lut_l1, in_dim,     hidden_dim, K,  L)
        _fill(self.lut_l2, hidden_dim2, out_dim,   K2, L2)

    # ---- forward ----------------------------------------------------------

    def _edge_forward_bulk(
        self,
        x: torch.Tensor,           # (N, n_src)
        lut_block: torch.Tensor,   # (n_src, n_dst, K, L)
        x_min_t: torch.Tensor,
        x_max_t: torch.Tensor,
        seg_width_t: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate all edges in a layer in one fused pass.

        Returns (N, n_dst).
        """
        N = x.shape[0]
        n_src, n_dst, K, L = lut_block.shape
        # Clip input, compute segment index and in-segment position
        hi = x_max_t - torch.tensor(1e-7, dtype=x.dtype, device=x.device)
        x_clip = torch.clamp(x, x_min_t, hi)              # (N, n_src)
        t = (x_clip - x_min_t) / seg_width_t               # (N, n_src)
        k = torch.clamp(torch.floor(t).long(), 0, K - 1)   # (N, n_src)
        u = torch.clamp(t - k.to(t.dtype), 0.0, 1.0)       # (N, n_src)
        pos = u * (L - 1)
        r0 = torch.clamp(torch.floor(pos).long(), 0, L - 1)  # (N, n_src)
        r1 = torch.clamp(r0 + 1, 0, L - 1)                   # (N, n_src)
        w = pos - r0.to(pos.dtype)                            # (N, n_src)

        # Gather: we need lut_block[i, :, k[n, i], r0[n, i]] for all (n, i)
        # Easiest with advanced indexing: reshape lut_block for gather.
        # lut_block: (n_src, n_dst, K, L)
        # We want result of shape (N, n_src, n_dst)
        # Per (n, i): v0[n, i, :] = lut_block[i, :, k[n,i], r0[n,i]]
        src_idx = torch.arange(n_src, device=x.device).view(1, n_src).expand(N, n_src)  # (N, n_src)
        # Fancy index: lut_block[src_idx, :, k, r0] - ':' on n_dst makes this
        # return shape (N, n_src, n_dst)
        v0 = lut_block[src_idx, :, k, r0]  # (N, n_src, n_dst)
        v1 = lut_block[src_idx, :, k, r1]  # (N, n_src, n_dst)

        # Lerp, then sum over source axis
        w_ = w.unsqueeze(-1)  # (N, n_src, 1)
        edges_out = v0 * (1.0 - w_) + v1 * w_   # (N, n_src, n_dst)
        return edges_out.sum(dim=1)              # (N, n_dst)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, in_dim). Returns (N, out_dim)."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # treat as (N, 1) if a plain 1D array passed
        z = self._edge_forward_bulk(
            x, self.lut_l1, self._x_min_l1, self._x_max_l1, self._seg_width_l1,
        )  # (N, hidden_dim)

        if self.activation == "tanh":
            a = torch.tanh(z)
        elif self.activation == "zscore":
            # (z - mu) / sigma.  Buffers default to (0, 1) = identity until calibrate().
            a = (z - self._z_mean) / self._z_std
        else:  # pragma: no cover — __init__ already validated
            raise ValueError(self.activation)

        y = self._edge_forward_bulk(
            a, self.lut_l2, self._x_min_l2, self._x_max_l2, self._seg_width_l2,
        )  # (N, out_dim)
        return y

    # ---- calibration -----------------------------------------------------

    @torch.no_grad()
    def calibrate_activation_stats(self, x_train: torch.Tensor,
                                   eps: float = 1e-6) -> dict:
        """Set z_mean, z_std buffers from a forward pass on x_train.

        Only meaningful when activation='zscore'. Can be called multiple
        times (e.g. after re-initialization), each call overwrites.

        Returns a dict with the measured stats for logging.
        """
        if self.activation != "zscore":
            raise RuntimeError(
                "calibrate_activation_stats() requires activation='zscore'; "
                f"this model was created with activation='{self.activation}'."
            )
        if x_train.dim() == 1:
            x_train = x_train.unsqueeze(-1)
        # Compute z = layer-1 forward output on all of x_train
        z = self._edge_forward_bulk(
            x_train, self.lut_l1, self._x_min_l1, self._x_max_l1, self._seg_width_l1,
        )  # (N, hidden_dim)
        mean = z.mean(dim=0)
        std = z.std(dim=0, unbiased=False).clamp(min=eps)
        self._z_mean.copy_(mean)
        self._z_std.copy_(std)
        self._calibrated.fill_(True)
        # Additional diagnostic: fraction of resulting a's outside [x_min_l2, x_max_l2]
        a = (z - mean) / std
        frac_clipped = float(((a < self._x_min_l2) | (a >= self._x_max_l2)).float().mean())
        return {
            "z_mean": mean.cpu().numpy().copy(),
            "z_std": std.cpu().numpy().copy(),
            "frac_clipped_after_zscore": frac_clipped,
        }

    # ---- utility ----------------------------------------------------------

    def extra_repr(self) -> str:
        n_edges = self.in_dim * self.hidden_dim + self.hidden_dim * self.out_dim
        n_lut_params = n_edges * self.K * self.L
        return (
            f"{self.in_dim}->{self.hidden_dim}->{self.out_dim}, "
            f"K={self.K}, L={self.L}, n_edges={n_edges}, "
            f"n_lut_params={n_lut_params}"
        )

    def total_lut_params(self) -> int:
        n_edges = self.in_dim * self.hidden_dim + self.hidden_dim * self.out_dim
        return n_edges * self.K * self.L

    def memory_bytes_uint8(self) -> int:
        """Total bytes if all LUTs were uint8-quantized per-segment."""
        n_edges = self.in_dim * self.hidden_dim + self.hidden_dim * self.out_dim
        # per edge: K*L (q) + 2*K*2 (scale+ymin at float16)
        return n_edges * (self.K * self.L + 4 * self.K)


# ─────────────────────────────────────────────────────────────────────────────
# Numpy forward for deployment / evaluation (mirrors LUTKAN2Layer.forward)
# ─────────────────────────────────────────────────────────────────────────────

def kan2_forward_numpy(
    x: np.ndarray,
    lut_l1: np.ndarray,     # (in_dim, hidden_dim, K, L)
    lut_l2: np.ndarray,     # (hidden_dim, out_dim, K, L)
    x_min_l1: float = -1.0,
    x_max_l1: float = 1.0,
    activation: str = "tanh",
    x_min_l2: float = -1.0,
    x_max_l2: float = 1.0,
    z_mean: np.ndarray = None,   # (hidden_dim,), required when activation='zscore'
    z_std: np.ndarray = None,    # (hidden_dim,), required when activation='zscore'
) -> np.ndarray:
    """NumPy reference forward. Matches LUTKAN2Layer.forward bit-for-bit.

    When activation='zscore', z_mean and z_std MUST be provided (usually
    saved from LUTKAN2Layer._z_mean and ._z_std after calibration).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float32).reshape(-1, lut_l1.shape[0]))
    N = x.shape[0]
    in_dim, hidden_dim, K, L = lut_l1.shape
    _, out_dim, _, _ = lut_l2.shape

    # Layer 1
    z = np.zeros((N, hidden_dim), dtype=np.float32)
    for i in range(in_dim):
        for h in range(hidden_dim):
            z[:, h] += lut_forward_numpy(
                x[:, i], lut_l1[i, h], x_min=x_min_l1, x_max=x_max_l1,
            )

    if activation == "tanh":
        a = np.tanh(z).astype(np.float32)
    elif activation == "zscore":
        if z_mean is None or z_std is None:
            raise ValueError("zscore activation requires z_mean and z_std")
        z_mean = np.asarray(z_mean, dtype=np.float32).reshape(1, -1)
        z_std = np.asarray(z_std, dtype=np.float32).reshape(1, -1)
        a = ((z - z_mean) / z_std).astype(np.float32)
    else:
        raise ValueError(f"activation must be 'tanh' or 'zscore', got {activation}")

    # Layer 2
    y = np.zeros((N, out_dim), dtype=np.float32)
    for h in range(hidden_dim):
        for o in range(out_dim):
            y[:, o] += lut_forward_numpy(
                a[:, h], lut_l2[h, o], x_min=x_min_l2, x_max=x_max_l2,
            )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial-KAN baseline (for matched-budget comparison)
# ─────────────────────────────────────────────────────────────────────────────

def polynomial_kan2_forward_numpy(
    x: np.ndarray,
    coeffs_l1: np.ndarray,   # (in_dim, hidden_dim, degree+1)
    coeffs_l2: np.ndarray,   # (hidden_dim, out_dim, degree+1)
) -> np.ndarray:
    """Two-layer polynomial KAN with Chebyshev basis and tanh squash."""
    from .baselines import eval_chebyshev

    x = np.atleast_2d(np.asarray(x, dtype=np.float32).reshape(-1, coeffs_l1.shape[0]))
    N = x.shape[0]
    in_dim, hidden_dim, _ = coeffs_l1.shape
    _, out_dim, _ = coeffs_l2.shape

    z = np.zeros((N, hidden_dim), dtype=np.float32)
    for i in range(in_dim):
        for h in range(hidden_dim):
            z[:, h] += eval_chebyshev(x[:, i], coeffs_l1[i, h])
    a = np.tanh(z).astype(np.float32)

    y = np.zeros((N, out_dim), dtype=np.float32)
    for h in range(hidden_dim):
        for o in range(out_dim):
            y[:, o] += eval_chebyshev(a[:, h], coeffs_l2[h, o])
    return y
