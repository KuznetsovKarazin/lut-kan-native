"""Tests for PolyKAN2Zscore — the zscore polynomial reference."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lut_native import PolyKAN2Zscore, train_poly_kan2_zscore


def test_construction_and_bad_domain():
    PolyKAN2Zscore(in_dim=2, hidden_dim=3, out_dim=1, degree=4)
    with pytest.raises(ValueError):
        PolyKAN2Zscore(in_dim=2, hidden_dim=3, out_dim=1, degree=4,
                       x_min_l2=1.0, x_max_l2=-1.0)


def test_calibrate_sets_stats():
    m = PolyKAN2Zscore(in_dim=2, hidden_dim=3, out_dim=1, degree=4)
    rng = np.random.RandomState(42)
    x = torch.from_numpy(rng.randn(200, 2).astype(np.float32))
    s = m.calibrate_activation_stats(x)
    assert bool(m._calibrated.item())
    assert m._z_std.shape == (3,)
    assert (m._z_std > 0).all()
    assert "frac_clipped_after_zscore" in s


def test_forward_after_calibration_a_is_unit_normal():
    """After calibration, layer-2 input has ~0 mean, ~1 std per hidden unit."""
    m = PolyKAN2Zscore(in_dim=2, hidden_dim=3, out_dim=1, degree=4)
    rng = np.random.RandomState(7)
    x = torch.from_numpy(rng.randn(500, 2).astype(np.float32))
    m.calibrate_activation_stats(x)

    with torch.no_grad():
        T = m._cheb_basis_layer1(x)
        z = torch.einsum("nid,ihd->nh", T, m.c_l1)
        a = (z - m._z_mean) / m._z_std

    per_h_mean = a.mean(dim=0)
    per_h_std = a.std(dim=0, unbiased=False)
    assert torch.allclose(per_h_mean, torch.zeros_like(per_h_mean), atol=1e-5)
    assert torch.allclose(per_h_std, torch.ones_like(per_h_std), atol=1e-5)


def test_training_reduces_loss():
    """Minimal sanity: the model trains and MSE decreases."""
    rng = np.random.RandomState(42)
    x_tr = rng.uniform(-1, 1, (500, 1)).astype(np.float32)
    y_tr = np.sin(np.pi * x_tr.ravel()).astype(np.float32)
    x_v = rng.uniform(-1, 1, (200, 1)).astype(np.float32)
    y_v = np.sin(np.pi * x_v.ravel()).astype(np.float32)
    x_te = np.linspace(-1, 1, 200, endpoint=False).reshape(-1, 1).astype(np.float32)
    y_te = np.sin(np.pi * x_te.ravel()).astype(np.float32)

    res = train_poly_kan2_zscore(
        in_dim=1, hidden_dim=2, out_dim=1, degree=8,
        x_tr=x_tr, y_tr=y_tr, x_v=x_v, y_v=y_v, x_te=x_te, y_te=y_te,
        epochs=100, batch_size=64, seed=0, eval_every=20,
    )
    assert res["test_mse"] < 1e-2  # should learn sine well enough
    assert res["z_mean"].shape == (2,)
    assert res["z_std"].shape == (2,)
