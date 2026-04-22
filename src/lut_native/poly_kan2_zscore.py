"""
Polynomial KAN2 with z-score layer-2 normalization.

Matches LUTKAN2Layer(activation='zscore') conventions so that poly-init can
produce meaningful LUT initializations for the zscore mode.

Key differences from the tanh-activation PolyKAN2:
  - Between layer 1 and layer 2: (z - mu) / sigma instead of tanh(z)
  - Layer 2's Chebyshev basis operates on [x_min_l2, x_max_l2] (clamped + mapped to [-1,1])
  - mu, sigma are not learned parameters — they're buffers set by calibration
    on the training set (analogous to LUTKAN2Layer.calibrate_activation_stats)

This is the correct polynomial reference for z-score LUT-KAN experiments.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


class PolyKAN2Zscore(nn.Module):
    """Two-layer polynomial KAN with z-score normalization between layers.

    Forward:
        T_x = Chebyshev basis on x (clamped to [-1, 1])
        z   = sum_i c_l1[i, h, :] · T_x[:, i, :]         # (N, hidden)
        a   = (z - mu) / sigma                            # (N, hidden)
        T_a_mapped = Chebyshev basis on (2a - (xmax+xmin)) / (xmax - xmin), clamped
        y   = sum_h c_l2[h, o, :] · T_a[:, h, :]          # (N, out_dim)

    The mapping to [-1, 1] for the Chebyshev basis of layer 2 is done by
    rescaling from [x_min_l2, x_max_l2].
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        degree: int,
        x_min_l2: float = -3.0,
        x_max_l2: float = 3.0,
    ):
        super().__init__()
        if not (x_min_l2 < x_max_l2):
            raise ValueError("require x_min_l2 < x_max_l2")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.degree = int(degree)
        self.x_min_l2 = float(x_min_l2)
        self.x_max_l2 = float(x_max_l2)

        self.c_l1 = nn.Parameter(torch.randn(in_dim, hidden_dim, degree + 1) * 0.1)
        self.c_l2 = nn.Parameter(torch.randn(hidden_dim, out_dim, degree + 1) * 0.1)

        # Buffers set by calibrate_activation_stats()
        self.register_buffer("_z_mean", torch.zeros(hidden_dim, dtype=torch.float32))
        self.register_buffer("_z_std", torch.ones(hidden_dim, dtype=torch.float32))
        self.register_buffer("_calibrated", torch.tensor(False))

    # ---- Chebyshev helpers -----------------------------------------------

    def _cheb_basis_on_unit(self, xn: torch.Tensor) -> torch.Tensor:
        """xn already in [-1, 1]. Returns (..., degree+1)."""
        T = [torch.ones_like(xn), xn]
        for n in range(2, self.degree + 1):
            T.append(2 * xn * T[-1] - T[-2])
        return torch.stack(T, dim=-1)

    def _cheb_basis_layer1(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, in_dim) on [-1, 1] (clamped). -> (N, in_dim, degree+1)."""
        xn = torch.clamp(x, -1.0, 1.0)
        return self._cheb_basis_on_unit(xn)

    def _cheb_basis_layer2(self, a: torch.Tensor) -> torch.Tensor:
        """a: (N, hidden_dim) on [x_min_l2, x_max_l2]. Map to [-1, 1], clamp, eval."""
        # Rescale to [-1, 1]: u = 2*(a - x_min_l2) / (x_max_l2 - x_min_l2) - 1
        width = self.x_max_l2 - self.x_min_l2
        un = 2.0 * (a - self.x_min_l2) / width - 1.0
        un = torch.clamp(un, -1.0, 1.0)
        return self._cheb_basis_on_unit(un)

    # ---- calibration -----------------------------------------------------

    @torch.no_grad()
    def calibrate_activation_stats(self, x_train: torch.Tensor, eps: float = 1e-6) -> Dict:
        """Set z_mean, z_std buffers from the layer-1 outputs on x_train."""
        if x_train.dim() == 1:
            x_train = x_train.unsqueeze(-1)
        T = self._cheb_basis_layer1(x_train)
        z = torch.einsum("nid,ihd->nh", T, self.c_l1)
        mean = z.mean(dim=0)
        std = z.std(dim=0, unbiased=False).clamp(min=eps)
        self._z_mean.copy_(mean)
        self._z_std.copy_(std)
        self._calibrated.fill_(True)
        a = (z - mean) / std
        frac_clipped = float(((a < self.x_min_l2) | (a >= self.x_max_l2)).float().mean())
        return {
            "z_mean": mean.cpu().numpy().copy(),
            "z_std": std.cpu().numpy().copy(),
            "frac_clipped_after_zscore": frac_clipped,
        }

    # ---- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        T = self._cheb_basis_layer1(x)
        z = torch.einsum("nid,ihd->nh", T, self.c_l1)
        a = (z - self._z_mean) / self._z_std
        Ta = self._cheb_basis_layer2(a)
        y = torch.einsum("nhd,hod->no", Ta, self.c_l2)
        return y

    # ---- utility ----------------------------------------------------------

    def total_coeffs(self) -> int:
        n_edges = self.in_dim * self.hidden_dim + self.hidden_dim * self.out_dim
        return n_edges * (self.degree + 1)

    def memory_bytes_float32(self) -> int:
        return self.total_coeffs() * 4


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_poly_kan2_zscore(
    in_dim, hidden_dim, out_dim, degree,
    x_tr, y_tr, x_v, y_v, x_te, y_te,
    x_min_l2=-3.0, x_max_l2=3.0,
    lr=5e-3, epochs=2000, batch_size=128, seed=0, eval_every=50,
    recalibrate_every: int = 50,
) -> Dict:
    """Train PolyKAN2Zscore. Recalibrates z-mean/std every `recalibrate_every`
    epochs because the distribution of z drifts as layer-1 weights change.

    Returns dict with best-val coefficients, z_mean, z_std, test_mse."""
    torch.manual_seed(seed)
    model = PolyKAN2Zscore(
        in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, degree=degree,
        x_min_l2=x_min_l2, x_max_l2=x_max_l2,
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    x_tr_t = torch.from_numpy(x_tr.astype(np.float32))
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).view(-1, out_dim)
    x_v_t = torch.from_numpy(x_v.astype(np.float32))
    y_v_t = torch.from_numpy(y_v.astype(np.float32)).view(-1, out_dim)
    x_e_t = torch.from_numpy(x_te.astype(np.float32))
    y_e_t = torch.from_numpy(y_te.astype(np.float32)).view(-1, out_dim)

    # Initial calibration
    model.calibrate_activation_stats(x_tr_t)

    N = x_tr_t.shape[0]
    best_val = float("inf")
    best_c1 = model.c_l1.detach().cpu().numpy().copy()
    best_c2 = model.c_l2.detach().cpu().numpy().copy()
    best_z_mean = model._z_mean.detach().cpu().numpy().copy()
    best_z_std = model._z_std.detach().cpu().numpy().copy()
    best_epoch = 0

    for ep in range(epochs):
        # Periodic recalibration to track layer-1 drift
        if recalibrate_every > 0 and ep > 0 and ep % recalibrate_every == 0:
            with torch.no_grad():
                model.calibrate_activation_stats(x_tr_t)

        perm = torch.randperm(N)
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            optim.zero_grad()
            loss = ((model(x_tr_t[idx]) - y_tr_t[idx]) ** 2).mean()
            loss.backward()
            optim.step()

        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            with torch.no_grad():
                mse_v = ((model(x_v_t) - y_v_t) ** 2).mean().item()
            if mse_v < best_val:
                best_val = mse_v
                best_c1 = model.c_l1.detach().cpu().numpy().copy()
                best_c2 = model.c_l2.detach().cpu().numpy().copy()
                best_z_mean = model._z_mean.detach().cpu().numpy().copy()
                best_z_std = model._z_std.detach().cpu().numpy().copy()
                best_epoch = ep + 1

    # Restore best and evaluate on test
    with torch.no_grad():
        model.c_l1.data = torch.from_numpy(best_c1)
        model.c_l2.data = torch.from_numpy(best_c2)
        model._z_mean.copy_(torch.from_numpy(best_z_mean))
        model._z_std.copy_(torch.from_numpy(best_z_std))
        mse_test = float(((model(x_e_t) - y_e_t) ** 2).mean().item())

    return {
        "coeffs_l1": best_c1,
        "coeffs_l2": best_c2,
        "z_mean": best_z_mean,
        "z_std": best_z_std,
        "x_min_l2": x_min_l2,
        "x_max_l2": x_max_l2,
        "test_mse": mse_test,
        "val_mse": best_val,
        "best_epoch": best_epoch,
    }
