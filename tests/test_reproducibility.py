"""
Verify that train_lut_edge with a fixed seed produces bit-identical results
on repeated runs. This is the minimum bar for "reproducible research".
"""

from __future__ import annotations

import numpy as np

from lut_native import (
    TrainConfig,
    fit_chebyshev_ls,
    generate_data,
    sample_polynomial_to_lut,
    train_lut_edge,
)


def _run_once(seed: int = 0, epochs: int = 50):
    x_train, y_train, x_val, y_val, x_test, y_test = generate_data(
        "sine", n_train=100, n_val=50, n_test=50, seed=42
    )
    coeffs = fit_chebyshev_ls(x_train, y_train, degree=16)
    lut_init = sample_polynomial_to_lut(coeffs, K=8, L=16)

    cfg = TrainConfig(
        lambda_2=1.0, lr=1e-2, epochs=epochs, batch_size=32,
        init_noise_std_rel=0.01, seed=seed, eval_every_epochs=10,
    )
    return train_lut_edge(
        lut_init=lut_init,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
        x_min=-1.0, x_max=1.0,
        cfg=cfg,
    )


def test_same_seed_produces_same_result():
    r1 = _run_once(seed=0)
    r2 = _run_once(seed=0)
    np.testing.assert_array_equal(r1.lut_final, r2.lut_final)
    np.testing.assert_array_equal(r1.lut_best, r2.lut_best)
    assert r1.mse_test_at_best == r2.mse_test_at_best
    assert r1.best_epoch == r2.best_epoch


def test_different_seed_produces_different_result():
    r1 = _run_once(seed=0)
    r2 = _run_once(seed=1)
    # They had better be different — otherwise our stochasticity is broken
    assert not np.array_equal(r1.lut_final, r2.lut_final)
