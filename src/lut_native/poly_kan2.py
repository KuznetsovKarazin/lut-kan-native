"""
Two-layer polynomial KAN (Chebyshev basis) - matched architecture to LUTKAN2Layer.

Used as the "honest" baseline for multi-edge experiments: same layout
(in_dim -> hidden_dim -> out_dim, with tanh squash between layers), same
training procedure, so the only difference is the edge representation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def _cheb_basis_torch(x: torch.Tensor, degree: int) -> torch.Tensor:
    """Chebyshev basis evaluated via 3-term recurrence. Input shape (..., 1) or
    (...,). Output shape (..., degree+1) with last dim being the basis values."""
    x = x if x.dim() > 0 else x.unsqueeze(0)
    Ts = [torch.ones_like(x), x]
    for _ in range(2, degree + 1):
        Ts.append(2.0 * x * Ts[-1] - Ts[-2])
    return torch.stack(Ts, dim=-1)  # (..., degree+1)


class PolyKAN2Layer(nn.Module):
    """Two-layer polynomial KAN with Chebyshev basis.

    Same overall shape as LUTKAN2Layer: in_dim -> hidden_dim -> out_dim,
    tanh between layers. Stored as:
      c_l1: (in_dim, hidden_dim, degree+1)
      c_l2: (hidden_dim, out_dim, degree+1)

    For layer-1 inputs, we assume the caller already normalizes to [-1, 1]
    (done by clamp in forward). Layer-2 inputs are tanh outputs, already in (-1, 1).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, degree: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.degree = int(degree)
        # Small-scale init to keep outputs bounded
        scale = 1.0 / float(degree + 1)
        self.c_l1 = nn.Parameter(
            torch.randn(in_dim, hidden_dim, degree + 1) * scale
        )
        self.c_l2 = nn.Parameter(
            torch.randn(hidden_dim, out_dim, degree + 1) * scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        xc = torch.clamp(x, -1.0, 1.0)  # (N, in_dim)
        # Basis for each input: (N, in_dim, degree+1)
        T = _cheb_basis_torch(xc, self.degree)
        # Layer 1: sum over in_dim of T[n, i, :] @ c_l1[i, h, :]
        z = torch.einsum("nid,ihd->nh", T, self.c_l1)     # (N, hidden_dim)
        a = torch.tanh(z)                                   # (N, hidden_dim)
        # Layer 2
        Ta = _cheb_basis_torch(a, self.degree)              # (N, hidden_dim, degree+1)
        y = torch.einsum("nhd,hod->no", Ta, self.c_l2)      # (N, out_dim)
        return y

    def total_coeffs(self) -> int:
        n_edges = self.in_dim * self.hidden_dim + self.hidden_dim * self.out_dim
        return n_edges * (self.degree + 1)

    def memory_bytes_float32(self) -> int:
        return self.total_coeffs() * 4


def train_poly_kan2(
    model: PolyKAN2Layer,
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    lr: float = 5e-3, epochs: int = 2000, batch_size: int = 64,
    seed: int = 0, eval_every_epochs: int = 20,
) -> Dict:
    """Train a polynomial KAN2. Same val-best protocol as train_kan2."""
    torch.manual_seed(seed)

    def _to2d(a, width):
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            return a.reshape(-1, 1) if width == 1 else a.reshape(-1, width)
        return a

    xt = torch.from_numpy(_to2d(x_train, model.in_dim))
    yt = torch.from_numpy(_to2d(y_train, model.out_dim))
    xv = torch.from_numpy(_to2d(x_val, model.in_dim))
    yv = torch.from_numpy(_to2d(y_val, model.out_dim))
    xe = torch.from_numpy(_to2d(x_test, model.in_dim))
    ye = torch.from_numpy(_to2d(y_test, model.out_dim))
    N = xt.shape[0]

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    with torch.no_grad():
        mse_val_init = ((model(xv) - yv) ** 2).mean().item()

    best_val = mse_val_init
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    trace = {"epoch": [0], "mse_val": [mse_val_init]}

    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            optim.zero_grad()
            loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
            loss.backward()
            optim.step()

        if (ep + 1) % eval_every_epochs == 0 or (ep + 1) == epochs:
            with torch.no_grad():
                v = ((model(xv) - yv) ** 2).mean().item()
            trace["epoch"].append(ep + 1)
            trace["mse_val"].append(v)
            if v < best_val:
                best_val = v
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
                best_epoch = ep + 1

    model.load_state_dict(best_state)
    with torch.no_grad():
        mse_train_b = ((model(xt) - yt) ** 2).mean().item()
        mse_val_b = ((model(xv) - yv) ** 2).mean().item()
        mse_test_b = ((model(xe) - ye) ** 2).mean().item()

    return {
        "mse_train_at_best": float(mse_train_b),
        "mse_val_at_best": float(mse_val_b),
        "mse_test_at_best": float(mse_test_b),
        "best_epoch": int(best_epoch),
        "trace": trace,
    }
