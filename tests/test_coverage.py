"""
Tests for coverage diagnostics.

Approach: build synthetic input distributions where we know exactly which
LUT cells should be hit, and verify the diagnostics produce the expected
numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from lut_native.coverage import (
    _effective_support,
    _gini,
    _range_utilization,
    _visited_fraction,
    _per_sample_indices,
    _accumulate_cell_hits_for_layer,
    compute_kan2_coverage,
)
from lut_native.kan2 import LUTKAN2Layer


# ─────────────────────────────────────────────────────────────────────────────
# Metric primitives
# ─────────────────────────────────────────────────────────────────────────────

def test_visited_fraction_trivial():
    # all cells hit
    assert _visited_fraction(np.ones((4, 8))) == 1.0
    # no cells hit
    assert _visited_fraction(np.zeros((4, 8))) == 0.0
    # half hit
    arr = np.zeros((4, 8))
    arr[0, :] = 5.0
    arr[1, :] = 3.0
    assert _visited_fraction(arr) == 0.5


def test_effective_support_uniform_is_N():
    # Uniform distribution over N cells -> effective_support = N
    N = 32
    arr = np.full(N, 1.0 / N)
    assert _effective_support(arr) == pytest.approx(N, rel=1e-6)


def test_effective_support_concentrated_is_1():
    arr = np.zeros(32)
    arr[7] = 1.0
    assert _effective_support(arr) == pytest.approx(1.0, rel=1e-6)


def test_effective_support_empty_is_0():
    assert _effective_support(np.zeros(16)) == 0.0


def test_gini_uniform_is_0():
    assert _gini(np.ones(10)) == pytest.approx(0.0, abs=1e-9)


def test_gini_concentrated_approaches_1():
    x = np.zeros(100)
    x[0] = 1.0
    g = _gini(x)
    assert g > 0.98


def test_range_utilization():
    # sample covers exactly half the domain
    x = np.linspace(-1.0, 0.0, 100)
    assert _range_utilization(x, x_min=-1.0, x_max=1.0) == pytest.approx(0.5, abs=1e-3)
    # sample covers the whole domain
    x = np.linspace(-1.0, 1.0 - 1e-6, 100)
    assert _range_utilization(x, x_min=-1.0, x_max=1.0) == pytest.approx(1.0, abs=1e-3)
    # sample well outside — should clip at 0
    x = np.array([5.0, 6.0])
    assert _range_utilization(x, x_min=-1.0, x_max=1.0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Indexing primitives match LUTKAN2Layer's forward
# ─────────────────────────────────────────────────────────────────────────────

def test_per_sample_indices_consistent_with_forward():
    """The indices returned by _per_sample_indices must be exactly the indices
    LUTKAN2Layer uses internally. Otherwise coverage measures something
    different from what training actually sees."""
    import torch
    K, L = 4, 8
    x_min, x_max = -1.0, 1.0
    model = LUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=K, L=L,
                         x_min=x_min, x_max=x_max)
    # Known LUT values where each cell stores its (k, r) index (encoded).
    # Then forward result for (n, i, h) with known k, r0, r1, w should match.
    with torch.no_grad():
        for ki in range(K):
            for ri in range(L):
                model.lut_l1[:, :, ki, ri] = float(ki * 1000 + ri)

    N = 50
    rng = np.random.RandomState(0)
    x = rng.uniform(x_min, x_max, size=(N, 2)).astype(np.float32)
    xt = torch.from_numpy(x)

    # Indices via our helper
    k, r0, r1, w = _per_sample_indices(xt, K=K, L=L, x_min=x_min, x_max=x_max)

    # Reconstruct expected output of layer1 for n_dst=3 (broadcast):
    # edge value = lerp( lut[k,r0], lut[k,r1], w ) = (ki*1000+r0)*(1-w) + (ki*1000+r1)*w
    v0 = (k.float() * 1000.0 + r0.float())
    v1 = (k.float() * 1000.0 + r1.float())
    expected_edge_val = v0 * (1.0 - w) + v1 * w  # (N, 2)
    # Summed over src, broadcast across n_dst, should equal the first-layer forward:
    expected_z = expected_edge_val.sum(dim=1, keepdim=True).expand(-1, 3)

    z = model._edge_forward_bulk(
        xt, model.lut_l1, model._x_min_l1, model._x_max_l1, model._seg_width_l1,
    )
    assert torch.allclose(z, expected_z, atol=1e-4, rtol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Hit accumulation sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_hits_sum_to_N_per_src():
    """For each sample, exactly two cells of each src edge receive weights
    summing to 1. So total hit mass per src = N."""
    K, L = 4, 8
    N = 200
    rng = np.random.RandomState(42)
    x = rng.uniform(-1.0, 1.0, size=(N, 3)).astype(np.float32)
    hits = _accumulate_cell_hits_for_layer(
        x, n_dst=2, K=K, L=L, x_min=-1.0, x_max=1.0,
    )
    # Shape check
    assert hits.shape == (3, 2, K, L)
    # Per-src total (broadcast across dst), we look at dst=0
    for i in range(3):
        total = hits[i, 0].sum()
        assert total == pytest.approx(N, abs=1e-4)
    # Check broadcast — all dst copies are identical
    assert np.allclose(hits[:, 0], hits[:, 1])


def test_uniform_inputs_give_high_coverage():
    """Dense uniform inputs should visit most cells and concentrate very little."""
    K, L = 8, 16
    N = 5000
    rng = np.random.RandomState(0)
    x = rng.uniform(-1.0, 1.0, size=(N, 1)).astype(np.float32)
    hits = _accumulate_cell_hits_for_layer(
        x, n_dst=1, K=K, L=L, x_min=-1.0, x_max=1.0,
    )
    # Every cell should be hit (N is huge)
    h = hits[0, 0]  # (K, L)
    assert _visited_fraction(h) > 0.95
    # Effective support should be close to K*L
    assert _effective_support(h) > 0.8 * K * L


def test_concentrated_inputs_give_low_coverage():
    """Inputs all near x=0 should only visit cells near the middle."""
    K, L = 8, 16
    N = 5000
    x = np.random.RandomState(0).normal(0.0, 0.01, size=(N, 1)).astype(np.float32)
    x = np.clip(x, -1.0, 1.0 - 1e-6)
    hits = _accumulate_cell_hits_for_layer(
        x, n_dst=1, K=K, L=L, x_min=-1.0, x_max=1.0,
    )
    h = hits[0, 0]
    # Very few cells visited
    assert _visited_fraction(h) < 0.2
    # Effective support is small
    assert _effective_support(h) < 10.0


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end on a KAN2
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_kan2_coverage_end_to_end():
    """Sanity check: run compute_kan2_coverage on a small model and verify
    the report has the right shape and no NaNs."""
    K, L = 4, 8
    model = LUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=K, L=L)
    rng = np.random.RandomState(0)
    x = rng.uniform(-1.0, 1.0, size=(100, 2)).astype(np.float32)

    report = compute_kan2_coverage(model, x)

    # Layer 1
    assert report.layer1.n_src == 2
    assert report.layer1.n_dst == 3
    assert len(report.layer1.visited_fraction_per_src) == 2
    assert 0.0 <= report.layer1.visited_fraction_mean <= 1.0
    assert 0.0 <= report.layer1.range_utilization_mean <= 1.0
    assert report.layer1.effective_support_mean <= K * L + 1e-6

    # Layer 2
    assert report.layer2.n_src == 3
    assert report.layer2.n_dst == 1

    # Hidden stats
    stats = report.hidden_activation_stats
    assert all(-1.0 - 1e-6 <= stats[k] <= 1.0 + 1e-6
               for k in ("tanh_z_mean", "tanh_z_min", "tanh_z_max"))
