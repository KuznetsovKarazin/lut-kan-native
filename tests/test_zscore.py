"""
Tests for z-score activation mode in LUTKAN2Layer.

Verify:
  1. Default activation is 'tanh' (backward compat)
  2. activation='zscore' before calibration behaves like identity normalization
  3. calibrate_activation_stats() produces correct (mean, std)
  4. Forward with zscore uses calibrated stats correctly
  5. Numpy reference forward matches PyTorch forward in both modes
  6. Invalid activation raises
  7. Calling calibrate on a 'tanh' model raises
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lut_native import LUTKAN2Layer, kan2_forward_numpy


def _make_model(activation="tanh", x_min_l2=-1.0, x_max_l2=1.0, seed=0):
    """Small test model with non-trivial LUT values."""
    torch.manual_seed(seed)
    model = LUTKAN2Layer(
        in_dim=2, hidden_dim=3, out_dim=1, K=4, L=8,
        activation=activation, x_min_l2=x_min_l2, x_max_l2=x_max_l2,
    )
    rng = np.random.RandomState(seed)
    with torch.no_grad():
        model.lut_l1.data.copy_(torch.from_numpy(
            rng.randn(2, 3, 4, 8).astype(np.float32)
        ))
        model.lut_l2.data.copy_(torch.from_numpy(
            rng.randn(3, 1, 4, 8).astype(np.float32)
        ))
    return model


def test_default_is_tanh():
    m = LUTKAN2Layer(in_dim=1, hidden_dim=2, out_dim=1, K=4, L=8)
    assert m.activation == "tanh"
    assert m.x_min_l2 == -1.0 and m.x_max_l2 == 1.0


def test_tanh_mode_ignores_l2_domain_args():
    # Passing x_min_l2=-3, x_max_l2=3 with activation='tanh' should be ignored
    m = LUTKAN2Layer(in_dim=1, hidden_dim=2, out_dim=1, K=4, L=8,
                     activation="tanh", x_min_l2=-3.0, x_max_l2=3.0)
    assert m.x_min_l2 == -1.0 and m.x_max_l2 == 1.0


def test_invalid_activation_raises():
    with pytest.raises(ValueError):
        LUTKAN2Layer(in_dim=1, hidden_dim=2, out_dim=1, K=4, L=8,
                     activation="relu")


def test_zscore_with_bad_domain_raises():
    with pytest.raises(ValueError):
        LUTKAN2Layer(in_dim=1, hidden_dim=2, out_dim=1, K=4, L=8,
                     activation="zscore", x_min_l2=1.0, x_max_l2=-1.0)


def test_calibrate_requires_zscore():
    m = _make_model(activation="tanh")
    x = torch.randn(100, 2)
    with pytest.raises(RuntimeError):
        m.calibrate_activation_stats(x)


def test_zscore_before_calibration_uses_identity():
    """An uncalibrated zscore model should use _z_mean=0, _z_std=1 buffers,
    which makes the forward equivalent to just dividing z by 1 — identity."""
    m = _make_model(activation="zscore", x_min_l2=-3.0, x_max_l2=3.0)
    assert not bool(m._calibrated.item())
    # _z_mean all zeros, _z_std all ones
    assert torch.allclose(m._z_mean, torch.zeros_like(m._z_mean))
    assert torch.allclose(m._z_std, torch.ones_like(m._z_std))

    x = torch.randn(50, 2)
    with torch.no_grad():
        y = m(x)
    # Should not produce NaN or inf even when z values extend outside domain
    assert torch.isfinite(y).all()


def test_calibrate_sets_stats():
    m = _make_model(activation="zscore", x_min_l2=-3.0, x_max_l2=3.0)
    rng = np.random.RandomState(42)
    x = torch.from_numpy(rng.randn(500, 2).astype(np.float32))

    stats = m.calibrate_activation_stats(x)

    assert bool(m._calibrated.item())
    assert m._z_mean.shape == (3,)
    assert m._z_std.shape == (3,)
    # std must be strictly positive
    assert (m._z_std > 0).all()
    # Sanity: z_mean in stats dict matches buffer
    assert np.allclose(stats["z_mean"], m._z_mean.cpu().numpy())
    assert np.allclose(stats["z_std"], m._z_std.cpu().numpy())
    # frac_clipped is reported
    assert "frac_clipped_after_zscore" in stats
    assert 0.0 <= stats["frac_clipped_after_zscore"] <= 1.0


def test_after_calibration_layer2_input_is_unit_normal():
    """After calibration on x, when we feed x through the model the inputs to
    layer 2 should have ~zero mean and ~unit std per hidden unit."""
    m = _make_model(activation="zscore", x_min_l2=-3.0, x_max_l2=3.0)
    rng = np.random.RandomState(42)
    x = torch.from_numpy(rng.randn(500, 2).astype(np.float32))
    m.calibrate_activation_stats(x)

    # Manually run layer 1 + normalize, check mean/std
    with torch.no_grad():
        z = m._edge_forward_bulk(
            x, m.lut_l1, m._x_min_l1, m._x_max_l1, m._seg_width_l1,
        )
        a = (z - m._z_mean) / m._z_std

    # Per-hidden-unit stats should be ~0 mean, ~1 std (unbiased=False matches
    # what the calibration does).
    per_h_mean = a.mean(dim=0)
    per_h_std = a.std(dim=0, unbiased=False)
    assert torch.allclose(per_h_mean, torch.zeros_like(per_h_mean), atol=1e-5)
    assert torch.allclose(per_h_std, torch.ones_like(per_h_std), atol=1e-5)


def test_numpy_forward_matches_torch_tanh_mode():
    m = _make_model(activation="tanh")
    rng = np.random.RandomState(0)
    x = rng.uniform(-1, 1, size=(50, 2)).astype(np.float32)

    with torch.no_grad():
        y_t = m(torch.from_numpy(x)).cpu().numpy()

    y_np = kan2_forward_numpy(
        x,
        lut_l1=m.lut_l1.detach().cpu().numpy(),
        lut_l2=m.lut_l2.detach().cpu().numpy(),
        x_min_l1=m.x_min, x_max_l1=m.x_max,
        activation="tanh",
    )
    np.testing.assert_allclose(y_t, y_np, atol=1e-5, rtol=1e-5)


def test_numpy_forward_matches_torch_zscore_mode():
    m = _make_model(activation="zscore", x_min_l2=-3.0, x_max_l2=3.0)
    rng = np.random.RandomState(1)
    x = rng.uniform(-1, 1, size=(200, 2)).astype(np.float32)
    m.calibrate_activation_stats(torch.from_numpy(x))

    with torch.no_grad():
        y_t = m(torch.from_numpy(x)).cpu().numpy()

    y_np = kan2_forward_numpy(
        x,
        lut_l1=m.lut_l1.detach().cpu().numpy(),
        lut_l2=m.lut_l2.detach().cpu().numpy(),
        x_min_l1=m.x_min, x_max_l1=m.x_max,
        activation="zscore",
        x_min_l2=m.x_min_l2, x_max_l2=m.x_max_l2,
        z_mean=m._z_mean.cpu().numpy(),
        z_std=m._z_std.cpu().numpy(),
    )
    np.testing.assert_allclose(y_t, y_np, atol=1e-5, rtol=1e-5)


def test_zscore_numpy_missing_stats_raises():
    m = _make_model(activation="zscore", x_min_l2=-3.0, x_max_l2=3.0)
    x = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        kan2_forward_numpy(
            x,
            lut_l1=m.lut_l1.detach().cpu().numpy(),
            lut_l2=m.lut_l2.detach().cpu().numpy(),
            activation="zscore",
            # no z_mean, z_std
        )


def test_calibrate_idempotent_with_same_input():
    """Calling calibrate twice on the same data should yield the same stats."""
    m = _make_model(activation="zscore", x_min_l2=-3.0, x_max_l2=3.0)
    rng = np.random.RandomState(7)
    x = torch.from_numpy(rng.randn(300, 2).astype(np.float32))
    s1 = m.calibrate_activation_stats(x)
    s2 = m.calibrate_activation_stats(x)
    np.testing.assert_allclose(s1["z_mean"], s2["z_mean"])
    np.testing.assert_allclose(s1["z_std"], s2["z_std"])
