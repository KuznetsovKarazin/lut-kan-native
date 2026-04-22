"""
Training loop for ResidualLUTKAN2Layer.

Similar to train_kan2 in training_kan2.py, but:
  - Accepts ResidualLUTKAN2Layer instead of LUTKAN2Layer
  - Optimizer sees deltas (not luts)
  - Regularization applied to EFFECTIVE lut (init + alpha * delta), matching
    what the kernel actually uses at inference
  - Records delta norms in the trace for diagnostics
  - best_model is reconstructed with the same alpha/activation/domain as the
    training model (this is the lesson from the M3 bug)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from .kan2_residual import ResidualLUTKAN2Layer


@dataclass
class ResidualTrainConfig:
    lambda_1: float = 0.0       # first-diff on effective LUT
    lambda_2: float = 0.0       # second-diff on effective LUT
    # Proximal penalty to keep the EFFECTIVE lut near init.
    # Equivalent to penalizing alpha*delta, so scales with alpha.
    # Typically 0.0 for pure M4a (trust-region-only); nonzero in M4b.
    lambda_init_anchor_l1: float = 0.0
    lambda_init_anchor_l2: float = 0.0
    # Learning rates per layer (for M4c)
    lr_l1: float = 1e-2
    lr_l2: float = 1e-2
    epochs: int = 300
    batch_size: int = 128
    seed: int = 0
    eval_every_epochs: int = 10
    calibrate_at_start: bool = True


@dataclass
class ResidualTrainResult:
    delta_l1_best: np.ndarray
    delta_l2_best: np.ndarray
    delta_l1_final: np.ndarray
    delta_l2_final: np.ndarray
    mse_train_at_best: float
    mse_val_at_best: float
    mse_test_at_best: float
    mse_val_init: float          # MSE before any training (= poly-init MSE)
    mse_val_final: float
    best_epoch: int
    trace: Dict
    n_updates: int
    alpha: float


def _effective_first_diff(lut_eff: torch.Tensor) -> torch.Tensor:
    d = lut_eff[..., 1:] - lut_eff[..., :-1]
    return (d ** 2).mean()


def _effective_second_diff(lut_eff: torch.Tensor) -> torch.Tensor:
    d = lut_eff[..., 2:] - 2 * lut_eff[..., 1:-1] + lut_eff[..., :-2]
    return (d ** 2).mean()


def train_residual_kan2(
    model: ResidualLUTKAN2Layer,
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    cfg: ResidualTrainConfig,
) -> ResidualTrainResult:
    torch.manual_seed(cfg.seed)

    if cfg.calibrate_at_start and model.activation == "zscore":
        with torch.no_grad():
            x_cal = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
            if x_cal.dim() == 1 and model.in_dim == 1:
                x_cal = x_cal.reshape(-1, 1)
            model.calibrate_activation_stats(x_cal)

    # Per-parameter group learning rates
    optim = torch.optim.Adam([
        {"params": [model.delta_l1], "lr": cfg.lr_l1},
        {"params": [model.delta_l2], "lr": cfg.lr_l2},
    ])

    def _to2d(a, width):
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            if width == 1:
                return a.reshape(-1, 1)
            raise ValueError(f"1D array with width={width}")
        return a

    xt = torch.from_numpy(_to2d(x_train, model.in_dim))
    yt = torch.from_numpy(_to2d(y_train, model.out_dim))
    xv = torch.from_numpy(_to2d(x_val, model.in_dim))
    yv = torch.from_numpy(_to2d(y_val, model.out_dim))
    xe = torch.from_numpy(_to2d(x_test, model.in_dim))
    ye = torch.from_numpy(_to2d(y_test, model.out_dim))
    N = xt.shape[0]

    n_updates = 0
    with torch.no_grad():
        mse_val_init = ((model(xv) - yv) ** 2).mean().item()
    best_val = mse_val_init
    best_d1 = model.delta_l1.detach().cpu().numpy().copy()
    best_d2 = model.delta_l2.detach().cpu().numpy().copy()
    best_epoch = 0

    trace = {"epoch": [0], "mse_train": [None], "mse_val": [mse_val_init],
             "delta_l1_l2": [0.0], "delta_l2_l2": [0.0]}

    for ep in range(cfg.epochs):
        perm = torch.randperm(N)
        for s in range(0, N, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            xb, yb = xt[idx], yt[idx]
            optim.zero_grad()
            y_pred = model(xb)
            data_loss = ((y_pred - yb) ** 2).mean()

            reg = torch.zeros((), dtype=torch.float32)
            # Smoothness regs on effective LUT (which is what the kernel uses)
            if cfg.lambda_1 > 0:
                reg = reg + cfg.lambda_1 * (
                    _effective_first_diff(model.lut_l1)
                    + _effective_first_diff(model.lut_l2)
                ) / 2.0
            if cfg.lambda_2 > 0:
                reg = reg + cfg.lambda_2 * (
                    _effective_second_diff(model.lut_l1)
                    + _effective_second_diff(model.lut_l2)
                ) / 2.0
            # Proximal-to-init: penalize the magnitude of the (alpha*delta) term
            # Equivalent to |lut - lut_init|^2 per-layer
            if cfg.lambda_init_anchor_l1 > 0:
                reg = reg + cfg.lambda_init_anchor_l1 * (
                    model.alpha ** 2 * (model.delta_l1 ** 2).mean()
                )
            if cfg.lambda_init_anchor_l2 > 0:
                reg = reg + cfg.lambda_init_anchor_l2 * (
                    model.alpha ** 2 * (model.delta_l2 ** 2).mean()
                )
            loss = data_loss + reg
            loss.backward()
            optim.step()
            n_updates += 1

        if (ep + 1) % cfg.eval_every_epochs == 0 or (ep + 1) == cfg.epochs:
            with torch.no_grad():
                mse_t = ((model(xt) - yt) ** 2).mean().item()
                mse_v = ((model(xv) - yv) ** 2).mean().item()
                dn = model.delta_norms()
            trace["epoch"].append(ep + 1)
            trace["mse_train"].append(mse_t)
            trace["mse_val"].append(mse_v)
            trace["delta_l1_l2"].append(dn["delta_l1_l2"])
            trace["delta_l2_l2"].append(dn["delta_l2_l2"])
            if mse_v < best_val:
                best_val = mse_v
                best_d1 = model.delta_l1.detach().cpu().numpy().copy()
                best_d2 = model.delta_l2.detach().cpu().numpy().copy()
                best_epoch = ep + 1

    delta_l1_final = model.delta_l1.detach().cpu().numpy().copy()
    delta_l2_final = model.delta_l2.detach().cpu().numpy().copy()

    # Reconstruct best-model with MATCHING alpha/activation/domain.
    # Lesson from M3 bug: must carry all attributes.
    best_model = ResidualLUTKAN2Layer(
        in_dim=model.in_dim, hidden_dim=model.hidden_dim, out_dim=model.out_dim,
        K=model.K, L=model.L, alpha=model.alpha,
        x_min=model.x_min, x_max=model.x_max,
        activation=model.activation,
        x_min_l2=model.x_min_l2, x_max_l2=model.x_max_l2,
    )
    with torch.no_grad():
        best_model.lut_l1_init.copy_(model.lut_l1_init)
        best_model.lut_l2_init.copy_(model.lut_l2_init)
        best_model.delta_l1.copy_(torch.from_numpy(best_d1))
        best_model.delta_l2.copy_(torch.from_numpy(best_d2))
        if model.activation == "zscore":
            best_model._z_mean.copy_(model._z_mean)
            best_model._z_std.copy_(model._z_std)
            best_model._calibrated.copy_(model._calibrated)

        mse_train_best = ((best_model(xt) - yt) ** 2).mean().item()
        mse_val_best = ((best_model(xv) - yv) ** 2).mean().item()
        mse_test_best = ((best_model(xe) - ye) ** 2).mean().item()
        mse_val_final = ((model(xv) - yv) ** 2).mean().item()

    return ResidualTrainResult(
        delta_l1_best=best_d1, delta_l2_best=best_d2,
        delta_l1_final=delta_l1_final, delta_l2_final=delta_l2_final,
        mse_train_at_best=float(mse_train_best),
        mse_val_at_best=float(mse_val_best),
        mse_test_at_best=float(mse_test_best),
        mse_val_init=float(mse_val_init),
        mse_val_final=float(mse_val_final),
        best_epoch=int(best_epoch),
        trace=trace,
        n_updates=n_updates,
        alpha=model.alpha,
    )
