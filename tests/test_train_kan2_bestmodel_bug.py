"""
Regression test for a bug where train_kan2 evaluated best_model with default
tanh activation even when the input model used zscore. This caused
mse_test_at_best to be computed with a different forward than what was trained,
silently producing wrong numbers.

The symptom: mse_val_at_best > mse_val_final, which is impossible if the
tracking were correct (best is always the minimum seen).
"""

from __future__ import annotations

import numpy as np
import torch

from lut_native import (
    KAN2TrainConfig,
    LUTKAN2Layer,
    train_kan2,
)


def _make_simple_task(seed=0, n=300):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1, (n, 1)).astype(np.float32)
    y = np.sin(np.pi * x).astype(np.float32)
    return x, y


def test_train_kan2_zscore_best_and_final_consistent():
    """best and final should both be reasonable numbers, with
    mse_val_at_best <= mse_val_final (best is by definition a minimum)."""
    x_tr, y_tr = _make_simple_task(seed=0, n=500)
    x_v, y_v = _make_simple_task(seed=1, n=200)
    x_te, y_te = _make_simple_task(seed=2, n=200)

    model = LUTKAN2Layer(
        in_dim=1, hidden_dim=3, out_dim=1, K=8, L=16,
        activation="zscore", x_min_l2=-3.0, x_max_l2=3.0,
    )
    with torch.no_grad():
        rng = np.random.RandomState(0)
        model.lut_l1.copy_(torch.from_numpy(
            rng.randn(1, 3, 8, 16).astype(np.float32) * 0.3))
        model.lut_l2.copy_(torch.from_numpy(
            rng.randn(3, 1, 8, 16).astype(np.float32) * 0.3))

    cfg = KAN2TrainConfig(
        lambda_1=0.0, lambda_2=0.0, lr=5e-4, epochs=80,
        batch_size=64, init_noise_std_absolute=0.0,
        seed=0, eval_every_epochs=10, calibrate_at_start=True,
    )
    res = train_kan2(model, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)

    # By definition of "best": mse_val_at_best must be <= mse_val_final
    # (otherwise it wasn't the best seen during training).
    assert res.mse_val_at_best <= res.mse_val_final + 1e-6, (
        f"BUG: best-val MSE ({res.mse_val_at_best:.6e}) > final-val MSE "
        f"({res.mse_val_final:.6e}). This indicates best_model was evaluated "
        f"with a different forward than `model`."
    )

    # Also: mse_val_at_best must equal what the trace says was best
    best_in_trace = min(v for v in res.trace["mse_val"] if v is not None)
    assert abs(res.mse_val_at_best - best_in_trace) < 1e-5, (
        f"BUG: mse_val_at_best ({res.mse_val_at_best}) disagrees with "
        f"the trace minimum ({best_in_trace})"
    )


def test_train_kan2_tanh_best_and_final_consistent():
    """Same invariant for the tanh activation (backward-compat check)."""
    x_tr, y_tr = _make_simple_task(seed=0, n=500)
    x_v, y_v = _make_simple_task(seed=1, n=200)
    x_te, y_te = _make_simple_task(seed=2, n=200)

    model = LUTKAN2Layer(in_dim=1, hidden_dim=3, out_dim=1, K=8, L=16)
    # (no explicit activation -> default 'tanh')
    with torch.no_grad():
        rng = np.random.RandomState(1)
        model.lut_l1.copy_(torch.from_numpy(
            rng.randn(1, 3, 8, 16).astype(np.float32) * 0.3))
        model.lut_l2.copy_(torch.from_numpy(
            rng.randn(3, 1, 8, 16).astype(np.float32) * 0.3))

    cfg = KAN2TrainConfig(
        lambda_1=0.0, lambda_2=0.0, lr=5e-4, epochs=80, batch_size=64,
        init_noise_std_absolute=0.0, seed=0, eval_every_epochs=10,
    )
    res = train_kan2(model, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)

    assert res.mse_val_at_best <= res.mse_val_final + 1e-6
