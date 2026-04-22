"""
Training loop for direct-LUT.

Key features:
  - Val-split-based best-model selection (guards against overfitting).
  - Per-cell gradient accumulators (for liveness diagnostics).
  - Explicit separation of optimizer randomness (torch) and init-noise
    randomness (numpy RandomState) for reproducibility.

Optimization choices:
  - Adam with lr=1e-2 by default; works well for this problem because
    gradients on LUT cells are sparse (each sample touches only 2 cells),
    so per-parameter adaptive scaling helps unvisited cells not drift.
  - Mini-batch SGD (bs=64). Tiny dataset (500 points), so roughly 8 steps/epoch.
  - No weight decay; regularization is explicit via difference penalties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .core import LUTEdge
from .regularizers import combined_penalty


@dataclass
class TrainConfig:
    lambda_1: float = 0.0     # first-diff
    lambda_2: float = 0.0     # second-diff
    lambda_bv: float = 0.0    # boundary value continuity
    lambda_bs: float = 0.0    # boundary slope continuity
    lr: float = 1e-2
    epochs: int = 1500
    batch_size: int = 64
    init_noise_std_rel: float = 0.01  # fraction of lut-init value range
    seed: int = 0
    # Validation-based best-model selection
    eval_every_epochs: int = 10


@dataclass
class TrainResult:
    lut_final: np.ndarray          # LUT at end of training (FYI; not the one used to report MSE)
    lut_best: np.ndarray           # LUT with lowest val MSE across training
    lut_init_clean: np.ndarray     # polynomial init (before noise)
    mse_train_at_best: float
    mse_val_at_best: float
    mse_test_at_best: float        # reported on a separate held-out test set
    mse_val_final: float
    best_epoch: int
    grad_abs_sum: np.ndarray       # per-cell cumulative |grad| over training
    cell_change_vs_init: np.ndarray  # |lut_best - lut_init_clean|
    trace: Dict                    # list[epoch], list[mse_train], list[mse_val]
    n_updates: int                 # gradient steps actually taken


def train_lut_edge(
    lut_init: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_min: float,
    x_max: float,
    cfg: TrainConfig,
) -> TrainResult:
    """Train a single LUTEdge, using val MSE to pick best model.

    Returns TrainResult with both final and best-val LUTs.
    """
    K, L = lut_init.shape

    # Reproducibility: torch controls optimizer stochasticity (shuffling);
    # numpy controls init noise.
    torch.manual_seed(cfg.seed)
    np_rng = np.random.RandomState(cfg.seed)

    # Apply init noise in numpy so the noisy init is fully reproducible.
    lut_range = float(lut_init.max() - lut_init.min())
    init_noise_std = cfg.init_noise_std_rel * lut_range
    lut_init_noisy = lut_init.astype(np.float32).copy()
    if init_noise_std > 0:
        lut_init_noisy += np_rng.randn(*lut_init.shape).astype(np.float32) * init_noise_std

    edge = LUTEdge(K=K, L=L, x_min=x_min, x_max=x_max)
    edge.init_from_array(lut_init_noisy)

    optim = torch.optim.Adam(edge.parameters(), lr=cfg.lr)

    # Tensors
    xt = torch.from_numpy(x_train.astype(np.float32))
    yt = torch.from_numpy(y_train.astype(np.float32))
    xv = torch.from_numpy(x_val.astype(np.float32))
    yv = torch.from_numpy(y_val.astype(np.float32))
    xe = torch.from_numpy(x_test.astype(np.float32))
    ye = torch.from_numpy(y_test.astype(np.float32))
    N = xt.shape[0]

    grad_abs_sum = torch.zeros(K, L)
    n_updates = 0

    # Initial val MSE
    with torch.no_grad():
        mse_val_init = ((edge(xv) - yv) ** 2).mean().item()
    best_val = mse_val_init
    best_lut = edge.lut.detach().cpu().numpy().copy()
    best_epoch = 0

    trace = {"epoch": [0], "mse_train": [None], "mse_val": [mse_val_init]}

    for ep in range(cfg.epochs):
        # Shuffle each epoch (torch-seeded)
        perm = torch.randperm(N)
        for s in range(0, N, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            xb, yb = xt[idx], yt[idx]

            optim.zero_grad()
            y_pred = edge(xb)
            data_loss = ((y_pred - yb) ** 2).mean()
            reg = combined_penalty(
                edge.lut,
                lambda_1=cfg.lambda_1,
                lambda_2=cfg.lambda_2,
                lambda_bv=cfg.lambda_bv,
                lambda_bs=cfg.lambda_bs,
            )
            loss = data_loss + reg
            loss.backward()

            with torch.no_grad():
                grad_abs_sum += edge.lut.grad.detach().abs()
                n_updates += 1

            optim.step()

        if (ep + 1) % cfg.eval_every_epochs == 0 or (ep + 1) == cfg.epochs:
            with torch.no_grad():
                mse_t = ((edge(xt) - yt) ** 2).mean().item()
                mse_v = ((edge(xv) - yv) ** 2).mean().item()
            trace["epoch"].append(ep + 1)
            trace["mse_train"].append(mse_t)
            trace["mse_val"].append(mse_v)
            if mse_v < best_val:
                best_val = mse_v
                best_lut = edge.lut.detach().cpu().numpy().copy()
                best_epoch = ep + 1

    # Final evaluation on all splits using best-val LUT
    import torch as _t
    best_edge = LUTEdge(K=K, L=L, x_min=x_min, x_max=x_max)
    best_edge.init_from_array(best_lut)
    with _t.no_grad():
        mse_train_best = ((best_edge(xt) - yt) ** 2).mean().item()
        mse_val_best = ((best_edge(xv) - yv) ** 2).mean().item()
        mse_test_best = ((best_edge(xe) - ye) ** 2).mean().item()
        mse_val_final = ((edge(xv) - yv) ** 2).mean().item()

    return TrainResult(
        lut_final=edge.lut.detach().cpu().numpy().copy(),
        lut_best=best_lut,
        lut_init_clean=lut_init.astype(np.float32),
        mse_train_at_best=float(mse_train_best),
        mse_val_at_best=float(mse_val_best),
        mse_test_at_best=float(mse_test_best),
        mse_val_final=float(mse_val_final),
        best_epoch=int(best_epoch),
        grad_abs_sum=grad_abs_sum.cpu().numpy(),
        cell_change_vs_init=np.abs(best_lut - lut_init.astype(np.float32)),
        trace=trace,
        n_updates=n_updates,
    )
