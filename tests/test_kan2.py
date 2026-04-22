"""
Verify LUTKAN2Layer is correct:
  1. Bulk PyTorch forward equals per-edge NumPy reference forward.
  2. Gradient flows to both layers of parameters.
  3. Shapes are correct for non-symmetric dimensions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lut_native.kan2 import LUTKAN2Layer, kan2_forward_numpy


@pytest.mark.parametrize(
    "in_dim, hidden_dim, out_dim, K, L",
    [
        (1, 4, 1, 8, 16),
        (1, 3, 1, 16, 32),
        (2, 3, 1, 4, 8),
        (1, 6, 2, 4, 8),
    ],
)
def test_kan2_forward_matches_numpy(in_dim, hidden_dim, out_dim, K, L):
    rng = np.random.RandomState(0)
    lut_l1 = rng.randn(in_dim, hidden_dim, K, L).astype(np.float32)
    lut_l2 = rng.randn(hidden_dim, out_dim, K, L).astype(np.float32) * 0.1
    x = rng.uniform(-1.0, 1.0, size=(50, in_dim)).astype(np.float32)

    # NumPy ref
    y_np = kan2_forward_numpy(x, lut_l1, lut_l2)

    # PyTorch bulk
    model = LUTKAN2Layer(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                         K=K, L=L)
    model.init_layer1_from_arrays(lut_l1)
    model.init_layer2_from_arrays(lut_l2)
    with torch.no_grad():
        y_t = model(torch.from_numpy(x)).numpy()

    np.testing.assert_allclose(y_np, y_t, rtol=1e-5, atol=1e-5)


def test_kan2_gradient_flows_to_both_layers():
    # Must have non-zero init on L2: with all-zero LUTs, hidden activations are 0
    # => L2 inputs are all tanh(0) = 0 => L2 only sees ONE LUT cell (position 0).
    # That one cell gets gradient, but L1 has no signal because L2 is flat elsewhere.
    # This is the expected "dead-init" pathology; real training must use init noise.
    model = LUTKAN2Layer(in_dim=1, hidden_dim=4, out_dim=1, K=8, L=16)
    with torch.no_grad():
        model.lut_l1.normal_(mean=0.0, std=0.3)
        model.lut_l2.normal_(mean=0.0, std=0.3)
    x = torch.linspace(-0.9, 0.9, 30).view(-1, 1)
    y_target = torch.sin(np.pi * x.squeeze()).view(-1, 1)
    y = model(x)
    loss = ((y - y_target) ** 2).mean()
    loss.backward()
    assert model.lut_l1.grad is not None
    assert model.lut_l2.grad is not None
    assert (model.lut_l1.grad.abs().sum() > 0).item(), "no gradient on layer 1"
    assert (model.lut_l2.grad.abs().sum() > 0).item(), "no gradient on layer 2"


def test_kan2_parameter_count():
    model = LUTKAN2Layer(in_dim=1, hidden_dim=4, out_dim=1, K=16, L=32)
    # L1: 1*4 edges * 16*32 cells = 2048
    # L2: 4*1 edges * 16*32 cells = 2048
    expected = 4096
    assert model.total_lut_params() == expected
    # uint8 memory: 8 edges * (16*32 + 4*16) bytes = 8 * 576 = 4608
    expected_bytes = 8 * (16 * 32 + 4 * 16)
    assert model.memory_bytes_uint8() == expected_bytes


def test_kan2_handles_oob_inputs():
    """Out-of-domain layer-1 inputs should be clipped, not error."""
    model = LUTKAN2Layer(in_dim=1, hidden_dim=3, out_dim=1, K=4, L=8)
    x = torch.tensor([[-10.0], [10.0], [0.0]], dtype=torch.float32)
    y = model(x)
    assert y.shape == (3, 1)
    assert torch.isfinite(y).all()
