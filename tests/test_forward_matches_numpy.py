"""
Verify that LUTEdge (PyTorch) forward and lut_forward_numpy produce matching
output. This guards against the "your training-time forward silently differs
from your deployment-time forward" bug class.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lut_native import LUTEdge, lut_forward_numpy


@pytest.mark.parametrize("K, L", [(4, 8), (16, 32), (8, 64), (32, 16)])
def test_forward_matches_numpy(K, L):
    rng = np.random.RandomState(123)
    lut = rng.randn(K, L).astype(np.float32)
    x = rng.uniform(-1.0, 1.0, size=500).astype(np.float32)

    # NumPy ref
    y_np = lut_forward_numpy(x, lut, x_min=-1.0, x_max=1.0)

    # PyTorch edge
    edge = LUTEdge(K=K, L=L, x_min=-1.0, x_max=1.0)
    edge.init_from_array(lut)
    with torch.no_grad():
        y_t = edge(torch.from_numpy(x)).numpy()

    # Very tight tolerance: same math, same dtype, just two implementations.
    np.testing.assert_allclose(y_np, y_t, rtol=1e-6, atol=1e-6)


def test_forward_handles_domain_boundary():
    """Values at x_max should be clipped to last valid segment without blowing up."""
    K, L = 4, 8
    lut = np.arange(K * L, dtype=np.float32).reshape(K, L)
    edge = LUTEdge(K=K, L=L, x_min=-1.0, x_max=1.0)
    edge.init_from_array(lut)
    # Include exact boundary
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0 - 1e-6, 1.0], dtype=torch.float32)
    y = edge(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_forward_out_of_domain_is_clipped():
    K, L = 4, 8
    rng = np.random.RandomState(0)
    lut = rng.randn(K, L).astype(np.float32)
    edge = LUTEdge(K=K, L=L, x_min=-1.0, x_max=1.0)
    edge.init_from_array(lut)
    # Out-of-domain inputs should be clipped, not error, and equal boundary values
    x_oob = torch.tensor([-10.0, 10.0], dtype=torch.float32)
    x_bnd = torch.tensor([-1.0, 1.0 - 1e-6], dtype=torch.float32)
    y_oob = edge(x_oob)
    y_bnd = edge(x_bnd)
    torch.testing.assert_close(y_oob, y_bnd, rtol=1e-5, atol=1e-5)
