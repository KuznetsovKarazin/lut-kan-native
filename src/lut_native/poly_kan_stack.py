"""
poly_kan_stack.py — N-layer polynomial KAN stack for fair comparison with LUTKANStack.

Mirrors LUTKANStack exactly:
  - same dims, same tanh inter-layer nonlinearity
  - degree = K*L - 1 per edge (equal parameter count to LUT)
  - same Adam, lr, lambda_2, seed
  - only the activation representation differs

Used in sweep_full.py as the matched poly baseline.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .poly_kan2 import _cheb_basis_torch  # reuse existing Chebyshev basis


class PolyKANStack(nn.Module):
    """
    N-layer Chebyshev polynomial KAN.

    Each edge maps one input feature to one output feature via a degree-d
    Chebyshev polynomial sum.  Inter-layer activation: tanh (matching
    LUTKANStack default with smooth_norms=True).

    Parameters
    ----------
    dims : list of int
        Layer widths, e.g. [1, 4, 1] for a 2-layer network.
    degree : int
        Polynomial degree per edge.  Set to K*L - 1 to match a LUT with
        K segments and L cells per segment.
    """

    def __init__(self, dims: List[int], degree: int):
        super().__init__()
        self.dims   = list(dims)
        self.degree = int(degree)

        self.blocks = nn.ParameterList()
        scale = 1.0 / float(degree + 1)
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            self.blocks.append(
                nn.Parameter(torch.randn(in_d, out_d, degree + 1) * scale)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        z = x.float()
        for i, w in enumerate(self.blocks):
            zc = z.clamp(-1.0, 1.0)
            # T: (N, in_d, degree+1)
            T = _cheb_basis_torch(zc, self.degree)
            # z: (N, out_d)
            z = torch.einsum("nid,iod->no", T, w)
            if i < len(self.blocks) - 1:
                z = torch.tanh(z)
        return z

    def total_params(self) -> int:
        n = 0
        for i in range(len(self.dims) - 1):
            n += self.dims[i] * self.dims[i + 1] * (self.degree + 1)
        return n

    def memory_bytes_float32(self) -> int:
        return self.total_params() * 4


def train_poly_stack(
    model: PolyKANStack,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    lr: float = 1e-2,
    epochs: int = 1500,
    batch_size: int = 64,
    seed: int = 0,
    lambda_2: float = 1.0,
    patience: int = 0,
    eval_every: int = 20,
) -> Dict:
    """
    Train a PolyKANStack with Adam, early stopping, and lambda_2 L2 regularisation.

    Parameters mirror train_lut_stack for direct comparability.

    Returns
    -------
    dict with keys:
        mse_train_at_best, mse_val_at_best, mse_test_at_best,
        best_epoch, val_trace (list of val MSE at each eval)
    """
    torch.manual_seed(seed)

    def _tensor(a: np.ndarray, width: int) -> torch.Tensor:
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1 and width == 1:
            return torch.from_numpy(a.reshape(-1, 1))
        if a.ndim == 1:
            return torch.from_numpy(a.reshape(-1, width))
        return torch.from_numpy(a)

    in_d  = model.dims[0]
    out_d = model.dims[-1]
    xt = _tensor(x_train, in_d)
    yt = _tensor(y_train, out_d)
    xv = _tensor(x_val,   in_d)
    yv = _tensor(y_val,   out_d)
    xe = _tensor(x_test,  in_d)
    ye = _tensor(y_test,  out_d)
    N  = xt.shape[0]

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    def _reg() -> torch.Tensor:
        s = torch.tensor(0.0)
        for w in model.blocks:
            s = s + (w ** 2).sum()
        return s

    with torch.no_grad():
        best_val = float(((model(xv) - yv) ** 2).mean())
    best_state  = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch  = 0
    no_improve  = 0
    val_trace: List[float] = [best_val]

    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N, batch_size):
            idx = perm[s : s + batch_size]
            optim.zero_grad()
            loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
            if lambda_2 > 0:
                loss = loss + lambda_2 * _reg() / N
            loss.backward()
            optim.step()

        if (ep + 1) % eval_every == 0 or (ep + 1) == epochs:
            with torch.no_grad():
                v = float(((model(xv) - yv) ** 2).mean())
            val_trace.append(v)
            if v < best_val:
                best_val   = v
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
                best_epoch = ep + 1
                no_improve = 0
            else:
                no_improve += 1
            if patience > 0 and no_improve * eval_every >= patience:
                break

    model.load_state_dict(best_state)
    with torch.no_grad():
        mse_tr = float(((model(xt) - yt) ** 2).mean())
        mse_v  = float(((model(xv) - yv) ** 2).mean())
        mse_te = float(((model(xe) - ye) ** 2).mean())

    return {
        "mse_train_at_best": mse_tr,
        "mse_val_at_best":   mse_v,
        "mse_test_at_best":  mse_te,
        "best_epoch":        best_epoch,
        "val_trace":         val_trace,
    }
