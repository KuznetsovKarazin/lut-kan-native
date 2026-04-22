"""
Training loop for LUTKAN2Layer (two-layer multi-edge KAN with LUT edges).

Mirrors src/lut_native/training.py but operates on the full 2-layer model.
Same conventions: val-based best-model selection, Adam, mini-batch SGD,
explicit RNG separation.

Key difference from single-edge: regularization is applied separately to
every LUT in both layers, then averaged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .kan2 import LUTKAN2Layer
from .regularizers import first_diff_penalty, second_diff_penalty


@dataclass
class KAN2TrainConfig:
    lambda_1: float = 0.0
    lambda_2: float = 0.0
    lr: float = 1e-2
    epochs: int = 1500
    batch_size: int = 64
    init_noise_std_absolute: float = 0.1   # absolute std added to zero-init LUTs
    seed: int = 0
    eval_every_epochs: int = 10
    # If True AND model.activation=='zscore', run calibrate_activation_stats()
    # once on x_train BEFORE training. Applies noise first (so calibration
    # measures the distribution the optimizer will see).
    calibrate_at_start: bool = True
    # If True AND activation=='zscore', re-calibrate every N epochs. 0 = never.
    # Off by default; use only if you have evidence layer-1 drifts far.
    recalibrate_every_epochs: int = 0


@dataclass
class KAN2TrainResult:
    lut_l1_best: np.ndarray
    lut_l2_best: np.ndarray
    lut_l1_final: np.ndarray
    lut_l2_final: np.ndarray
    mse_train_at_best: float
    mse_val_at_best: float
    mse_test_at_best: float
    mse_val_final: float
    best_epoch: int
    trace: Dict
    n_updates: int


def _kan_reg(model: LUTKAN2Layer, lambda_1: float, lambda_2: float) -> torch.Tensor:
    """Sum of first-/second-diff penalties over all LUTs in both layers,
    averaged over edges to keep scale comparable to single-edge training."""
    loss = torch.zeros((), dtype=torch.float32, device=model.lut_l1.device)
    if lambda_1 == 0 and lambda_2 == 0:
        return loss

    in_dim, hidden_dim, K, L = model.lut_l1.shape
    _, out_dim, _, _ = model.lut_l2.shape

    # Apply penalty per-LUT. Flatten first dim to iterate.
    l1_flat = model.lut_l1.view(-1, K, L)  # (in*hidden, K, L)
    l2_flat = model.lut_l2.view(-1, K, L)  # (hidden*out, K, L)

    if lambda_1 > 0:
        # d1 shape: (n_edges, K, L-1) -> mean across all axes
        d1_l1 = (l1_flat[:, :, 1:] - l1_flat[:, :, :-1]) ** 2
        d1_l2 = (l2_flat[:, :, 1:] - l2_flat[:, :, :-1]) ** 2
        loss = loss + lambda_1 * (d1_l1.mean() + d1_l2.mean()) / 2.0
    if lambda_2 > 0:
        d2_l1 = (l1_flat[:, :, 2:] - 2 * l1_flat[:, :, 1:-1] + l1_flat[:, :, :-2]) ** 2
        d2_l2 = (l2_flat[:, :, 2:] - 2 * l2_flat[:, :, 1:-1] + l2_flat[:, :, :-2]) ** 2
        loss = loss + lambda_2 * (d2_l1.mean() + d2_l2.mean()) / 2.0
    return loss


def train_kan2(
    model: LUTKAN2Layer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: KAN2TrainConfig,
) -> KAN2TrainResult:
    """
    Train a LUTKAN2Layer. x arrays can be 1D (treated as (N, 1)) or 2D (N, in_dim).
    y arrays can be 1D (treated as (N, 1)) or 2D (N, out_dim).
    """
    torch.manual_seed(cfg.seed)
    np_rng = np.random.RandomState(cfg.seed)

    # Init noise (applied in numpy space for reproducibility)
    with torch.no_grad():
        noise_l1 = torch.from_numpy(
            np_rng.randn(*model.lut_l1.shape).astype(np.float32) * cfg.init_noise_std_absolute
        )
        noise_l2 = torch.from_numpy(
            np_rng.randn(*model.lut_l2.shape).astype(np.float32) * cfg.init_noise_std_absolute
        )
        model.lut_l1.add_(noise_l1)
        model.lut_l2.add_(noise_l2)

    # Optional: z-score calibration BEFORE training.
    # We do this after init-noise so the calibration reflects what the
    # optimizer will see at step 0.
    if getattr(model, "activation", "tanh") == "zscore" and cfg.calibrate_at_start:
        with torch.no_grad():
            # Calibrate on x_train (full, not mini-batched, since dataset fits)
            x_cal = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
            if x_cal.dim() == 1 and model.in_dim == 1:
                x_cal = x_cal.reshape(-1, 1)
            model.calibrate_activation_stats(x_cal)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def _to2d(a, width):
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            if width == 1:
                return a.reshape(-1, 1)
            raise ValueError(f"1D array passed with width={width}")
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
    best_l1 = model.lut_l1.detach().cpu().numpy().copy()
    best_l2 = model.lut_l2.detach().cpu().numpy().copy()
    best_epoch = 0

    trace = {"epoch": [0], "mse_train": [None], "mse_val": [mse_val_init]}

    for ep in range(cfg.epochs):
        # Optional periodic recalibration (off by default)
        if (cfg.recalibrate_every_epochs > 0
                and getattr(model, "activation", "tanh") == "zscore"
                and ep > 0
                and ep % cfg.recalibrate_every_epochs == 0):
            with torch.no_grad():
                model.calibrate_activation_stats(xt)

        perm = torch.randperm(N)
        for s in range(0, N, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            xb, yb = xt[idx], yt[idx]

            optim.zero_grad()
            y_pred = model(xb)
            data_loss = ((y_pred - yb) ** 2).mean()
            reg = _kan_reg(model, cfg.lambda_1, cfg.lambda_2)
            loss = data_loss + reg
            loss.backward()
            optim.step()
            n_updates += 1

        if (ep + 1) % cfg.eval_every_epochs == 0 or (ep + 1) == cfg.epochs:
            with torch.no_grad():
                mse_t = ((model(xt) - yt) ** 2).mean().item()
                mse_v = ((model(xv) - yv) ** 2).mean().item()
            trace["epoch"].append(ep + 1)
            trace["mse_train"].append(mse_t)
            trace["mse_val"].append(mse_v)
            if mse_v < best_val:
                best_val = mse_v
                best_l1 = model.lut_l1.detach().cpu().numpy().copy()
                best_l2 = model.lut_l2.detach().cpu().numpy().copy()
                best_epoch = ep + 1

    lut_l1_final = model.lut_l1.detach().cpu().numpy().copy()
    lut_l2_final = model.lut_l2.detach().cpu().numpy().copy()
    # Final evaluation using best LUTs.
    # IMPORTANT: best_model must carry the SAME activation/domain/calibration
    # as `model`. Otherwise, for activation='zscore', a best_model initialized
    # with the default (tanh, L2=[-1,1]) would evaluate using a different
    # forward than what was trained — silently giving wrong MSE numbers.
    best_model = LUTKAN2Layer(
        in_dim=model.in_dim, hidden_dim=model.hidden_dim, out_dim=model.out_dim,
        K=model.K, L=model.L, x_min=model.x_min, x_max=model.x_max,
        activation=model.activation,
        x_min_l2=model.x_min_l2, x_max_l2=model.x_max_l2,
    )
    best_model.init_layer1_from_arrays(best_l1)
    best_model.init_layer2_from_arrays(best_l2)
    # Copy calibration buffers when applicable
    if model.activation == "zscore":
        best_model._z_mean.copy_(model._z_mean)
        best_model._z_std.copy_(model._z_std)
        best_model._calibrated.copy_(model._calibrated)
    with torch.no_grad():
        mse_train_best = ((best_model(xt) - yt) ** 2).mean().item()
        mse_val_best = ((best_model(xv) - yv) ** 2).mean().item()
        mse_test_best = ((best_model(xe) - ye) ** 2).mean().item()
        mse_val_final = ((model(xv) - yv) ** 2).mean().item()

    return KAN2TrainResult(
        lut_l1_best=best_l1, lut_l2_best=best_l2,
        lut_l1_final=lut_l1_final, lut_l2_final=lut_l2_final,
        mse_train_at_best=float(mse_train_best),
        mse_val_at_best=float(mse_val_best),
        mse_test_at_best=float(mse_test_best),
        mse_val_final=float(mse_val_final),
        best_epoch=int(best_epoch),
        trace=trace,
        n_updates=n_updates,
    )
