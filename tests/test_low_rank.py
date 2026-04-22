"""Tests for LowRankResidualLUTKAN2Layer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lut_native import (
    LowRankResidualLUTKAN2Layer,
    LowRankTrainConfig,
    LUTKAN2Layer,
    kan2_forward_numpy,
    train_low_rank_kan2,
)


def _rng_lut(shape, seed):
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


def test_construction_and_bad_rank():
    LowRankResidualLUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=8, L=16,
                                alpha=0.1, rank_l1=2, rank_l2=4)
    with pytest.raises(ValueError):
        LowRankResidualLUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=8, L=16,
                                    alpha=0.1, rank_l1=0, rank_l2=2)
    with pytest.raises(ValueError):
        LowRankResidualLUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=8, L=16,
                                    alpha=0.1, rank_l1=2, rank_l2=100)


def test_parameters_are_four_factors():
    m = LowRankResidualLUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=4, L=8,
                                    alpha=0.1, rank_l1=2, rank_l2=2)
    names = {n for n, _ in m.named_parameters()}
    assert names == {"U_l1", "V_l1", "U_l2", "V_l2"}


def test_U_zero_init_makes_delta_zero():
    """At init, U is zero so delta = U @ V = 0 regardless of V.
    This makes the forward match a plain LUTKAN2Layer at init time."""
    K, L = 4, 8
    init_l1 = _rng_lut((2, 3, K, L), seed=0)
    init_l2 = _rng_lut((3, 1, K, L), seed=1)

    plain = LUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=K, L=L)
    plain.init_layer1_from_arrays(init_l1)
    plain.init_layer2_from_arrays(init_l2)

    for rank in [1, 2, 4]:
        m = LowRankResidualLUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=K, L=L,
                                        alpha=0.5, rank_l1=rank, rank_l2=rank)
        m.init_layer1_from_arrays(init_l1)
        m.init_layer2_from_arrays(init_l2)
        # V has random non-zero init; U is zero; so delta = 0.
        assert torch.allclose(m.delta_l1, torch.zeros_like(m.delta_l1))
        assert torch.allclose(m.delta_l2, torch.zeros_like(m.delta_l2))

        x = torch.from_numpy(np.random.RandomState(42).uniform(-1, 1, (10, 2)).astype(np.float32))
        with torch.no_grad():
            y_plain = plain(x)
            y_m = m(x)
        torch.testing.assert_close(y_plain, y_m, atol=1e-5, rtol=1e-5)


def test_delta_after_nonzero_U_is_rank_constrained():
    """After writing non-zero U and V, delta should be low-rank."""
    K, L = 8, 16
    m = LowRankResidualLUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=K, L=L,
                                    alpha=0.5, rank_l1=2, rank_l2=2)
    m.init_layer1_from_arrays(_rng_lut((2, 3, K, L), 0))
    m.init_layer2_from_arrays(_rng_lut((3, 1, K, L), 1))

    # Write random U (V already random)
    rng = np.random.RandomState(7)
    m.U_l1.data.copy_(torch.from_numpy(
        rng.randn(*m.U_l1.shape).astype(np.float32) * 0.3))

    d1 = m.delta_l1.detach().cpu().numpy()
    # For edge (0, 0), the (K, L) delta matrix must have rank <= 2
    for i in range(2):
        for h in range(3):
            mat = d1[i, h]
            # SVD rank check
            s = np.linalg.svd(mat, compute_uv=False)
            assert (s > 1e-6).sum() <= 2


def test_rank1_delta_is_rank1():
    K, L = 8, 16
    m = LowRankResidualLUTKAN2Layer(in_dim=1, hidden_dim=2, out_dim=1, K=K, L=L,
                                    alpha=1.0, rank_l1=1, rank_l2=1)
    m.init_layer1_from_arrays(_rng_lut((1, 2, K, L), 0))
    m.init_layer2_from_arrays(_rng_lut((2, 1, K, L), 1))
    rng = np.random.RandomState(0)
    m.U_l1.data.copy_(torch.from_numpy(
        rng.randn(*m.U_l1.shape).astype(np.float32) * 0.3))

    d1 = m.delta_l1.detach().cpu().numpy()
    for i in range(1):
        for h in range(2):
            s = np.linalg.svd(d1[i, h], compute_uv=False)
            # rank-1 matrix: only one non-negligible singular value
            assert s[0] > 1e-6
            assert all(v < 1e-5 for v in s[1:])


def test_numpy_forward_matches_torch():
    K, L = 4, 8
    m = LowRankResidualLUTKAN2Layer(in_dim=2, hidden_dim=3, out_dim=1, K=K, L=L,
                                    alpha=0.5, rank_l1=2, rank_l2=2)
    m.init_layer1_from_arrays(_rng_lut((2, 3, K, L), 7))
    m.init_layer2_from_arrays(_rng_lut((3, 1, K, L), 8))
    rng = np.random.RandomState(13)
    m.U_l1.data.copy_(torch.from_numpy(
        rng.randn(*m.U_l1.shape).astype(np.float32) * 0.2))
    m.U_l2.data.copy_(torch.from_numpy(
        rng.randn(*m.U_l2.shape).astype(np.float32) * 0.2))

    x_np = rng.uniform(-1, 1, (30, 2)).astype(np.float32)
    with torch.no_grad():
        y_torch = m(torch.from_numpy(x_np)).cpu().numpy()

    y_numpy = kan2_forward_numpy(
        x_np,
        lut_l1=m.effective_lut_l1_numpy(),
        lut_l2=m.effective_lut_l2_numpy(),
    )
    np.testing.assert_allclose(y_torch, y_numpy, rtol=1e-5, atol=1e-5)


def test_training_best_le_final():
    """Regression of the M3 bug, now for low-rank class."""
    rng = np.random.RandomState(0)
    x_tr = rng.uniform(-1, 1, (400, 1)).astype(np.float32)
    y_tr = np.sin(np.pi * x_tr.ravel()).astype(np.float32)
    x_v = rng.uniform(-1, 1, (100, 1)).astype(np.float32)
    y_v = np.sin(np.pi * x_v.ravel()).astype(np.float32)
    x_te = np.linspace(-1, 1, 100, endpoint=False).reshape(-1, 1).astype(np.float32)
    y_te = np.sin(np.pi * x_te.ravel()).astype(np.float32)

    m = LowRankResidualLUTKAN2Layer(in_dim=1, hidden_dim=3, out_dim=1, K=8, L=16,
                                    alpha=0.1, rank_l1=2, rank_l2=2)
    rng = np.random.RandomState(42)
    m.lut_l1_init.copy_(torch.from_numpy(
        rng.randn(*m.lut_l1_init.shape).astype(np.float32) * 0.3))
    m.lut_l2_init.copy_(torch.from_numpy(
        rng.randn(*m.lut_l2_init.shape).astype(np.float32) * 0.3))

    cfg = LowRankTrainConfig(lr_l1=5e-3, lr_l2=5e-3, epochs=40,
                             batch_size=64, seed=0, eval_every_epochs=5)
    res = train_low_rank_kan2(m, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)

    assert res.mse_val_at_best <= res.mse_val_final + 1e-6


def test_n_delta_params_is_reduced():
    """The whole point: fewer parameters than full delta."""
    K, L = 16, 32
    full_delta = 2 * 4 * K * L + 4 * 1 * K * L  # in=2, hidden=4, out=1 → (2+4 edges)*K*L  = 12*512 = 6144
    full_params = 2 * 4 * K * L + 4 * 1 * K * L

    m = LowRankResidualLUTKAN2Layer(in_dim=2, hidden_dim=4, out_dim=1,
                                    K=K, L=L, alpha=0.1,
                                    rank_l1=2, rank_l2=2)
    n_lr = m.n_delta_params()
    # rank-2 per edge: U has K*rank, V has rank*L → per edge 2*(K+L)=96
    # 8 edges (L1) + 4 edges (L2) = 12 edges * 96 = 1152
    assert n_lr == 12 * 2 * (K + L)
    assert n_lr < full_params   # should be substantially smaller
