"""
Tests for ResidualLUTKAN2Layer.

Critical invariants:
  1. With delta=0, forward matches LUTKAN2Layer with the same init.
  2. With alpha=0, delta updates do NOT change the output (since alpha*delta=0).
  3. Parameters registered correctly: delta_l1 and delta_l2 only.
  4. train_residual_kan2 respects mse_val_at_best <= mse_val_final invariant.
  5. Effective LUT matches init + alpha*delta exactly.
  6. Numpy reference (kan2_forward_numpy) matches given effective_lut_*_numpy().
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lut_native import (
    LUTKAN2Layer,
    ResidualLUTKAN2Layer,
    ResidualTrainConfig,
    kan2_forward_numpy,
    train_residual_kan2,
)


def _rng_lut(shape, seed):
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


def test_parameters_are_only_deltas():
    """The optimizer should only see deltas as trainable."""
    m = ResidualLUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=4, L=8, alpha=0.1)
    names = [n for n, _ in m.named_parameters()]
    assert set(names) == {"delta_l1", "delta_l2"}


def test_delta_zero_matches_plain_lut_kan2():
    """At delta=0, residual model's forward matches a plain LUTKAN2Layer
    with the same init."""
    K, L = 4, 8
    init_l1 = _rng_lut((2, 3, K, L), seed=0)
    init_l2 = _rng_lut((3, 1, K, L), seed=1)

    plain = LUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=K, L=L)
    plain.init_layer1_from_arrays(init_l1)
    plain.init_layer2_from_arrays(init_l2)

    for alpha in [0.0, 0.1, 1.0]:
        res = ResidualLUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=K, L=L,
                                   alpha=alpha)
        res.init_layer1_from_arrays(init_l1)
        res.init_layer2_from_arrays(init_l2)
        # deltas are zero after init, so alpha*delta = 0 regardless of alpha

        rng = np.random.RandomState(42)
        x = torch.from_numpy(rng.uniform(-1, 1, (20, 2)).astype(np.float32))
        with torch.no_grad():
            y_plain = plain(x)
            y_res = res(x)
        torch.testing.assert_close(y_plain, y_res, rtol=1e-5, atol=1e-5)


def test_alpha_zero_freezes_output():
    """With alpha=0, even after setting deltas to arbitrary large values,
    the output is unchanged from the init-only forward."""
    K, L = 4, 8
    init_l1 = _rng_lut((2, 3, K, L), seed=0)
    init_l2 = _rng_lut((3, 1, K, L), seed=1)

    res = ResidualLUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=K, L=L,
                               alpha=0.0)
    res.init_layer1_from_arrays(init_l1)
    res.init_layer2_from_arrays(init_l2)

    rng = np.random.RandomState(42)
    x = torch.from_numpy(rng.uniform(-1, 1, (20, 2)).astype(np.float32))
    with torch.no_grad():
        y_before = res(x)
        # Now clobber deltas with big random values
        res.delta_l1.data.copy_(torch.from_numpy(
            rng.randn(*res.delta_l1.shape).astype(np.float32) * 10.0))
        res.delta_l2.data.copy_(torch.from_numpy(
            rng.randn(*res.delta_l2.shape).astype(np.float32) * 10.0))
        y_after = res(x)
    # alpha=0 means alpha*delta=0 regardless
    torch.testing.assert_close(y_before, y_after)


def test_effective_lut_matches_init_plus_alpha_delta():
    """effective_lut_l1_numpy() must equal lut_l1_init + alpha * delta_l1."""
    K, L = 4, 8
    init_l1 = _rng_lut((2, 3, K, L), seed=7)
    init_l2 = _rng_lut((3, 1, K, L), seed=8)

    for alpha in [0.1, 0.5, 1.0]:
        m = ResidualLUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=K, L=L,
                                 alpha=alpha)
        m.init_layer1_from_arrays(init_l1)
        m.init_layer2_from_arrays(init_l2)
        # Write a random delta
        rng = np.random.RandomState(13)
        d1 = rng.randn(*m.delta_l1.shape).astype(np.float32) * 0.3
        d2 = rng.randn(*m.delta_l2.shape).astype(np.float32) * 0.3
        m.delta_l1.data.copy_(torch.from_numpy(d1))
        m.delta_l2.data.copy_(torch.from_numpy(d2))

        eff1 = m.effective_lut_l1_numpy()
        eff2 = m.effective_lut_l2_numpy()
        np.testing.assert_allclose(eff1, init_l1 + alpha * d1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(eff2, init_l2 + alpha * d2, rtol=1e-6, atol=1e-6)


def test_numpy_reference_matches_torch_forward():
    """With a non-trivial delta, the numpy reference evaluated on the
    effective LUT must match the torch forward."""
    K, L = 4, 8
    init_l1 = _rng_lut((2, 3, K, L), seed=7)
    init_l2 = _rng_lut((3, 1, K, L), seed=8)

    m = ResidualLUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=K, L=L,
                             alpha=0.5)
    m.init_layer1_from_arrays(init_l1)
    m.init_layer2_from_arrays(init_l2)
    rng = np.random.RandomState(13)
    m.delta_l1.data.copy_(torch.from_numpy(
        rng.randn(*m.delta_l1.shape).astype(np.float32) * 0.3))
    m.delta_l2.data.copy_(torch.from_numpy(
        rng.randn(*m.delta_l2.shape).astype(np.float32) * 0.3))

    x_np = rng.uniform(-1, 1, (30, 2)).astype(np.float32)
    with torch.no_grad():
        y_torch = m(torch.from_numpy(x_np)).cpu().numpy()

    y_numpy = kan2_forward_numpy(
        x_np,
        lut_l1=m.effective_lut_l1_numpy(),
        lut_l2=m.effective_lut_l2_numpy(),
    )
    np.testing.assert_allclose(y_torch, y_numpy, rtol=1e-5, atol=1e-5)


def test_training_preserves_best_le_final_invariant():
    """Regression of the M3 bug: train_residual_kan2's best-val MSE must
    be <= final-val MSE."""
    rng = np.random.RandomState(0)
    x_tr = rng.uniform(-1, 1, (400, 1)).astype(np.float32)
    y_tr = np.sin(np.pi * x_tr.ravel()).astype(np.float32)
    x_v = rng.uniform(-1, 1, (100, 1)).astype(np.float32)
    y_v = np.sin(np.pi * x_v.ravel()).astype(np.float32)
    x_te = np.linspace(-1, 1, 100, endpoint=False).reshape(-1, 1).astype(np.float32)
    y_te = np.sin(np.pi * x_te.ravel()).astype(np.float32)

    m = ResidualLUTKAN2Layer(in_dim=1, hidden_dim=3, out_dim=1, K=8, L=16,
                             alpha=0.1)
    # Random init so there's room to learn
    rng = np.random.RandomState(42)
    m.lut_l1_init.copy_(torch.from_numpy(
        rng.randn(*m.lut_l1_init.shape).astype(np.float32) * 0.3))
    m.lut_l2_init.copy_(torch.from_numpy(
        rng.randn(*m.lut_l2_init.shape).astype(np.float32) * 0.3))

    cfg = ResidualTrainConfig(
        lambda_2=0.0, lr_l1=5e-3, lr_l2=5e-3, epochs=40,
        batch_size=64, seed=0, eval_every_epochs=5,
    )
    res = train_residual_kan2(m, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)

    assert res.mse_val_at_best <= res.mse_val_final + 1e-6


def test_alpha_zero_delta_stays_zero_after_training():
    """alpha=0 means deltas receive no gradient that affects output.
    But Adam noise etc. might still wiggle them. Verify that MSE stays
    constant."""
    rng = np.random.RandomState(0)
    x_tr = rng.uniform(-1, 1, (200, 1)).astype(np.float32)
    y_tr = np.sin(np.pi * x_tr.ravel()).astype(np.float32)
    x_v = x_tr[:50]; y_v = y_tr[:50]
    x_te = x_tr[50:100]; y_te = y_tr[50:100]

    m = ResidualLUTKAN2Layer(in_dim=1, hidden_dim=2, out_dim=1, K=4, L=8, alpha=0.0)
    rng = np.random.RandomState(42)
    m.lut_l1_init.copy_(torch.from_numpy(
        rng.randn(*m.lut_l1_init.shape).astype(np.float32) * 0.3))
    m.lut_l2_init.copy_(torch.from_numpy(
        rng.randn(*m.lut_l2_init.shape).astype(np.float32) * 0.3))

    cfg = ResidualTrainConfig(lr_l1=1e-2, lr_l2=1e-2, epochs=30,
                              batch_size=32, seed=0, eval_every_epochs=5)
    res = train_residual_kan2(m, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)

    # mse_val_init should equal mse_val_final to high precision: alpha=0
    # means the output is identical to init regardless of delta updates.
    assert abs(res.mse_val_init - res.mse_val_final) < 1e-4
