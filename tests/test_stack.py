"""
Tests for LUTKANStack, LUTInterLayerNorm, LUTBlock, and stack_forward_numpy.

Run with:  pytest tests/test_stack.py -v
"""

import numpy as np
import pytest
import torch

from lut_native.kan_stack import (
    LUTBlock,
    LUTInterLayerNorm,
    LUTKANStack,
    stack_forward_numpy,
)
from lut_native.coverage_stack import compute_stack_coverage, stack_coverage_to_dict
from lut_native.training_stack import StackTrainConfig, train_lut_stack


# ─────────────────────────────────────────────────────────────────────────────
# LUTInterLayerNorm
# ─────────────────────────────────────────────────────────────────────────────

class TestLUTInterLayerNorm:
    def test_output_shape(self):
        norm = LUTInterLayerNorm(dim=4)
        z = torch.randn(32, 4)
        a = norm(z)
        assert a.shape == (32, 4)

    def test_output_within_domain(self):
        # smooth=True (tanh): output bounded by (x_min, x_max) in float64,
        # but float32 tanh saturates exactly to ±1.0. Check [-1, 1] range.
        norm = LUTInterLayerNorm(dim=8, x_min=-1.0, x_max=1.0, smooth=True)
        z = torch.randn(200, 8) * 10
        a = norm(z)
        assert (a >= -1.0).all(), "tanh output must be >= x_min"
        assert (a <= 1.0).all(),  "tanh output must be <= x_max"
        # smooth=False (hard clamp): output in [x_min, x_max)
        norm2 = LUTInterLayerNorm(dim=8, x_min=-1.0, x_max=1.0, smooth=False)
        a2 = norm2(z)
        assert (a2 >= -1.0).all()
        assert (a2 < 1.0).all()

    def test_calibrate_shifts_output(self):
        """After calibration, the bulk of activations should be in [x_min, x_max]."""
        norm = LUTInterLayerNorm(dim=4, x_min=-1.0, x_max=1.0)
        torch.manual_seed(0)
        z = torch.randn(2000, 4) * 5 + 3.0    # biased distribution
        stats = norm.calibrate(z, plo=5.0, phi=95.0)
        a = norm(z)
        # Most of the distribution (90%) should be inside [-1, 1]
        frac_inside = ((a > -0.99) & (a < 0.99)).float().mean().item()
        assert frac_inside > 0.80, f"Only {frac_inside:.2f} fraction inside domain after calibration"

    def test_calibrate_identity_on_uniform(self):
        """On z ~ Uniform[-1, 1], calibration should produce near-identity mapping."""
        # smooth=True: tanh(u)*1 where u maps p5/p95 to atanh(0.9)
        norm = LUTInterLayerNorm(dim=1, x_min=-1.0, x_max=1.0, smooth=True)
        z = torch.linspace(-1.0, 1.0, 1000).unsqueeze(1)
        norm.calibrate(z, plo=5.0, phi=95.0)
        a = norm(z)
        # tanh squash: output should be monotone and span most of [-1, 1]
        assert float(a.min().detach()) > -1.0 and float(a.max().detach()) < 1.0
        assert float(a.min()) < -0.7, f"Expected coverage below -0.7, got {a.min():.3f}"
        assert float(a.max()) >  0.7, f"Expected coverage above  0.7, got {a.max():.3f}"

    def test_calibrated_flag(self):
        norm = LUTInterLayerNorm(dim=2)
        assert not bool(norm._calibrated.item())
        norm.calibrate(torch.randn(100, 2))
        assert bool(norm._calibrated.item())

    def test_export_numpy(self):
        norm = LUTInterLayerNorm(dim=3)
        norm.calibrate(torch.randn(100, 3) * 2 + 1)
        d = norm.export_numpy()
        assert set(d.keys()) == {"shift", "scale", "x_min", "x_max", "smooth"}
        assert d["shift"].shape == (3,)
        assert d["scale"].shape == (3,)
        assert (d["scale"] > 0).all()
        assert d["smooth"] is True  # default

    def test_scale_always_positive(self):
        """log_scale can be any real; scale = exp(log_scale) must be > 0."""
        norm = LUTInterLayerNorm(dim=4)
        norm.log_scale.data.fill_(-10.0)
        assert (norm.scale > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# LUTBlock
# ─────────────────────────────────────────────────────────────────────────────

class TestLUTBlock:
    def test_output_shape(self):
        blk = LUTBlock(in_dim=2, out_dim=4, K=8, L=16)
        x = torch.randn(64, 2)
        z = blk(x)
        assert z.shape == (64, 4)

    def test_zero_init_gives_zero(self):
        blk = LUTBlock(in_dim=3, out_dim=5, K=8, L=16)
        x = torch.randn(32, 3)
        z = blk(x)
        assert z.abs().max().item() == 0.0

    def test_gradient_flows(self):
        blk = LUTBlock(in_dim=2, out_dim=2, K=4, L=8)
        blk.add_init_noise(0.1)
        x = torch.randn(16, 2)
        loss = blk(x).sum()
        loss.backward()
        assert blk.lut.grad is not None
        assert blk.lut.grad.abs().sum().item() > 0

    def test_n_lut_params(self):
        blk = LUTBlock(in_dim=2, out_dim=3, K=8, L=16)
        assert blk.n_lut_params() == 2 * 3 * 8 * 16

    def test_init_from_array(self):
        blk = LUTBlock(in_dim=1, out_dim=1, K=4, L=8)
        arr = np.random.randn(1, 1, 4, 8).astype(np.float32)
        blk.init_from_array(arr)
        np.testing.assert_allclose(
            blk.lut.detach().numpy(), arr, rtol=1e-6
        )


# ─────────────────────────────────────────────────────────────────────────────
# LUTKANStack
# ─────────────────────────────────────────────────────────────────────────────

class TestLUTKANStack:
    def _make_model(self, dims=(2, 4, 4, 1), K=8, L=16):
        return LUTKANStack(dims=list(dims), K=K, L=L)

    def test_output_shape(self):
        model = self._make_model()
        x = torch.randn(32, 2)
        y = model(x)
        assert y.shape == (32, 1)

    def test_n_blocks_and_norms(self):
        model = self._make_model(dims=[2, 4, 4, 1])
        assert len(model.blocks) == 3
        assert len(model.norms) == 2

    def test_two_layer_has_no_norm(self):
        model = LUTKANStack(dims=[1, 1], K=8, L=16)
        assert len(model.norms) == 0

    def test_calibrate_sets_norms(self):
        model = self._make_model()
        model.add_init_noise(0.1)
        x = torch.randn(200, 2)
        before_shift = model.norms[0].shift.clone()
        model.calibrate(x)
        after_shift = model.norms[0].shift
        # shift should have changed from zero after calibration
        assert not torch.allclose(before_shift, after_shift)

    def test_gradient_flows_through_norms(self):
        """Gradients must flow through LUTInterLayerNorm to earlier blocks."""
        model = self._make_model(dims=[2, 4, 1])
        model.add_init_noise(0.1)
        torch.manual_seed(0)
        x = torch.randn(32, 2)
        y = torch.randn(32, 1)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        # Block 0 (before the norm) must have non-zero gradients
        assert model.blocks[0].lut.grad is not None
        assert model.blocks[0].lut.grad.abs().sum().item() > 0.0

    def test_1d_input_accepted(self):
        model = LUTKANStack(dims=[1, 4, 1], K=8, L=16)
        x = torch.randn(50)
        y = model(x)
        assert y.shape == (50, 1)

    def test_export_and_load_snapshot(self):
        model = self._make_model()
        model.add_init_noise(0.1)
        model.calibrate(torch.randn(100, 2))
        luts  = model.snapshot_luts()
        norms = model.snapshot_norms()
        # Modify model
        for blk in model.blocks:
            blk.lut.data.fill_(0.0)
        # Restore and check
        model.load_snapshot(luts, norms)
        for blk, arr in zip(model.blocks, luts):
            np.testing.assert_allclose(
                blk.lut.detach().numpy(), arr, rtol=1e-6
            )

    def test_total_params(self):
        model = self._make_model(dims=[2, 4, 1], K=8, L=16)
        expected_lut = 2*4*8*16 + 4*1*8*16
        expected_norm = 2 * 4   # 1 norm of dim=4: shift + log_scale
        assert model.n_lut_params() == expected_lut
        assert model.n_norm_params() == expected_norm
        assert model.total_params() == expected_lut + expected_norm


# ─────────────────────────────────────────────────────────────────────────────
# NumPy reference vs PyTorch
# ─────────────────────────────────────────────────────────────────────────────

class TestNumpyParity:
    def test_single_block_no_norm(self):
        model = LUTKANStack(dims=[2, 1], K=4, L=8)
        model.add_init_noise(0.3)
        x = np.random.randn(50, 2).astype(np.float32)
        y_torch = model(torch.from_numpy(x)).detach().numpy()
        y_numpy = stack_forward_numpy(x, model.export_luts(), model.export_norms())
        np.testing.assert_allclose(y_torch, y_numpy, atol=1e-5)

    def test_three_block_with_norms(self):
        torch.manual_seed(42)
        model = LUTKANStack(dims=[2, 4, 4, 1], K=8, L=16)
        model.cheby_init(scale=1.5, noise=0.05)
        x_calib = torch.randn(200, 2)
        model.calibrate(x_calib)
        np.random.seed(42)
        x = np.random.randn(100, 2).astype(np.float32)
        y_torch = model(torch.from_numpy(x)).detach().numpy()
        y_numpy = stack_forward_numpy(x, model.export_luts(), model.export_norms())
        # cheby_init gives bounded slopes; tanh norm accumulates ~3e-8/step
        np.testing.assert_allclose(y_torch, y_numpy, atol=5e-4,
                                   err_msg="Torch and numpy forwards disagree")

    def test_deeper_stack(self):
        """4-block parity with cheby_init for bounded LUT slopes."""
        torch.manual_seed(7)
        model = LUTKANStack(dims=[1, 8, 8, 8, 1], K=8, L=16)
        model.cheby_init(scale=1.5, noise=0.05)
        model.calibrate(torch.randn(300, 1))
        np.random.seed(7)
        x = np.random.randn(50, 1).astype(np.float32)
        y_torch = model(torch.from_numpy(x)).detach().numpy()
        y_numpy = stack_forward_numpy(x, model.export_luts(), model.export_norms())
        np.testing.assert_allclose(y_torch, y_numpy, atol=5e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Coverage diagnostics
# ─────────────────────────────────────────────────────────────────────────────

class TestStackCoverage:
    def test_coverage_improves_after_calibration(self):
        """
        A calibrated norm should produce higher segment uniformity than
        the uncalibrated (identity) norm on the same activation distribution.

        Setup: inputs in [-0.8, 0.8] (well within block domain) so that
        the block's output activations are NOT dominated by boundary clamping.
        The raw model's norm is identity (scale=1, shift=0) so activations
        stay wherever the LUT puts them.  The calibrated model's norm maps
        p5/p95 of those activations to [-1, 1], giving better segment spread.
        """
        torch.manual_seed(7)
        # Build two models with identical LUT weights
        model_calib = LUTKANStack(dims=[1, 8, 1], K=8, L=16)
        model_raw   = LUTKANStack(dims=[1, 8, 1], K=8, L=16)
        torch.manual_seed(3)
        noise_lut = torch.randn_like(model_calib.blocks[0].lut) * 1.5
        model_calib.blocks[0].lut.data.copy_(noise_lut)
        model_raw.blocks[0].lut.data.copy_(noise_lut)

        # Calibration inputs: uniform in [-0.8, 0.8] — well within block domain
        x_calib = (torch.rand(1000, 1) * 1.6 - 0.8)
        model_calib.calibrate(x_calib)

        # Measure coverage on the same distribution
        report_calib = compute_stack_coverage(model_calib, x_calib.numpy())
        report_raw   = compute_stack_coverage(model_raw,   x_calib.numpy())

        uni_calib = report_calib.norm_reports[0].uniformity_mean
        uni_raw   = report_raw.norm_reports[0].uniformity_mean
        assert uni_calib > uni_raw, (
            f"Calibrated uniformity {uni_calib:.3f} should be > raw {uni_raw:.3f}"
        )

    def test_full_coverage_structure(self):
        model = LUTKANStack(dims=[2, 4, 4, 1], K=8, L=16)
        model.add_init_noise(0.1)
        model.calibrate(torch.randn(200, 2))
        x = np.random.randn(300, 2).astype(np.float32)
        report = compute_stack_coverage(model, x)
        assert len(report.block_reports) == 3
        assert len(report.norm_reports) == 2
        assert report.n_samples == 300

    def test_to_dict_is_serialisable(self):
        import json
        model = LUTKANStack(dims=[1, 4, 1], K=4, L=8)
        model.add_init_noise(0.1)
        model.calibrate(torch.randn(100, 1))
        x = np.random.randn(50, 1).astype(np.float32)
        report = compute_stack_coverage(model, x)
        d = stack_coverage_to_dict(report)
        # Must be JSON-serialisable (no numpy arrays, no tensors)
        json.dumps(d)

    def test_coverage_summary_string(self):
        model = LUTKANStack(dims=[2, 4, 1], K=8, L=16)
        model.add_init_noise(0.1)
        model.calibrate(torch.randn(100, 2))
        x = np.random.randn(50, 2).astype(np.float32)
        report = compute_stack_coverage(model, x)
        s = str(report)
        assert "block" in s
        assert "norm" in s


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (smoke tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestStackTraining:
    def _make_sine_data(self, N=400, seed=0):
        rng = np.random.RandomState(seed)
        x = rng.uniform(-1, 1, (N, 1)).astype(np.float32)
        y = np.sin(2 * np.pi * x).astype(np.float32)
        return x[:300], y[:300], x[300:350], y[300:350], x[350:], y[350:]

    def test_train_returns_result(self):
        model = LUTKANStack(dims=[1, 4, 1], K=8, L=16)
        cfg   = StackTrainConfig(epochs=20, batch_size=32, seed=0)
        xt, yt, xv, yv, xe, ye = self._make_sine_data()
        res = train_lut_stack(model, xt, yt, xv, yv, xe, ye, cfg)
        assert res.mse_test_at_best > 0.0
        assert res.best_epoch >= 0
        assert res.n_updates > 0

    def test_train_reduces_mse(self):
        """MSE should drop from initial value after a real training run."""
        model = LUTKANStack(dims=[1, 8, 1], K=16, L=32)
        cfg   = StackTrainConfig(
            epochs=200, batch_size=64, lr=1e-2,
            init_noise_std=0.05, seed=42,
        )
        xt, yt, xv, yv, xe, ye = self._make_sine_data()
        res = train_lut_stack(model, xt, yt, xv, yv, xe, ye, cfg)
        assert res.mse_val_at_best < 0.5, (
            f"Val MSE={res.mse_val_at_best:.4f} — model did not learn"
        )

    def test_train_3layer_stack(self):
        """Three-block stack trains without errors."""
        model = LUTKANStack(dims=[1, 4, 4, 1], K=8, L=16)
        cfg   = StackTrainConfig(epochs=50, batch_size=32, seed=1)
        xt, yt, xv, yv, xe, ye = self._make_sine_data()
        res = train_lut_stack(model, xt, yt, xv, yv, xe, ye, cfg)
        assert isinstance(res.mse_test_at_best, float)

    def test_best_model_is_restored_after_training(self):
        """After training, model state should match the best snapshot."""
        model = LUTKANStack(dims=[1, 4, 1], K=8, L=16)
        cfg   = StackTrainConfig(epochs=40, eval_every_epochs=10, seed=3)
        xt, yt, xv, yv, xe, ye = self._make_sine_data()
        res = train_lut_stack(model, xt, yt, xv, yv, xe, ye, cfg)
        # train_lut_stack loads the best snapshot before returning
        # Re-evaluate: should match mse_val_at_best
        xv_t = torch.from_numpy(xv)
        yv_t = torch.from_numpy(yv)
        with torch.no_grad():
            mse_check = ((model(xv_t) - yv_t) ** 2).mean().item()
        assert abs(mse_check - res.mse_val_at_best) < 1e-7

    def test_track_coverage_flag(self):
        model = LUTKANStack(dims=[1, 4, 1], K=8, L=16)
        cfg   = StackTrainConfig(
            epochs=20, eval_every_epochs=10, seed=5, track_coverage=True
        )
        xt, yt, xv, yv, xe, ye = self._make_sine_data()
        res = train_lut_stack(model, xt, yt, xv, yv, xe, ye, cfg)
        assert "coverage_uniformity" in res.trace
        # Should have recorded values at eval epochs
        non_none = [v for v in res.trace["coverage_uniformity"] if v is not None]
        assert len(non_none) > 0

    def test_recalibrate_flag(self):
        """recalibrate_every_epochs should run without errors."""
        model = LUTKANStack(dims=[1, 4, 1], K=8, L=16)
        cfg   = StackTrainConfig(
            epochs=30, batch_size=32, seed=7,
            recalibrate_every_epochs=10
        )
        xt, yt, xv, yv, xe, ye = self._make_sine_data()
        res = train_lut_stack(model, xt, yt, xv, yv, xe, ye, cfg)
        assert isinstance(res.mse_test_at_best, float)


class TestChebyInit:
    """Chebyshev polynomial initialisation — the fix for dead-init failure."""

    def test_all_segments_active_after_cheby(self):
        """cheby_init must activate all K LUT segments from step 0."""
        import sys; sys.path.insert(0,'src')
        from lut_native.targets import generate_data_2d
        data = generate_data_2d('feynman_2d', n_train=400, n_val=100, n_test=100, seed=42)
        xt = torch.from_numpy(data[0])
        K = 16
        blk = LUTBlock(2, 4, K, 32)
        blk.cheby_init(scale=1.5, noise=0.05)
        with torch.no_grad():
            z = blk(xt)
            segs = ((z.clamp(-1, 1-1e-7)+1)/2*K).long().clamp(0, K-1)
            counts = torch.zeros(K)
            for k in range(K): counts[k] = (segs==k).float().mean()
            active = int((counts > 0.001).sum())
        assert active == K, f"Only {active}/{K} segments active — dead init not fixed"

    def test_noise_init_dead_by_comparison(self):
        """noise std=0.05 activates fewer segments than cheby — documents the bug."""
        import sys; sys.path.insert(0,'src')
        from lut_native.targets import generate_data_2d
        data = generate_data_2d('feynman_2d', n_train=400, n_val=100, n_test=100, seed=42)
        xt = torch.from_numpy(data[0])
        K = 16
        blk = LUTBlock(2, 4, K, 32)
        blk.add_init_noise(0.05)
        with torch.no_grad():
            z = blk(xt)
            segs = ((z.clamp(-1, 1-1e-7)+1)/2*K).long().clamp(0, K-1)
            counts = torch.zeros(K)
            for k in range(K): counts[k] = (segs==k).float().mean()
            active_noise = int((counts > 0.001).sum())
        blk2 = LUTBlock(2, 4, K, 32)
        blk2.cheby_init(scale=1.5, noise=0.05)
        with torch.no_grad():
            z2 = blk2(xt)
            segs2 = ((z2.clamp(-1, 1-1e-7)+1)/2*K).long().clamp(0, K-1)
            counts2 = torch.zeros(K)
            for k in range(K): counts2[k] = (segs2==k).float().mean()
            active_cheby = int((counts2 > 0.001).sum())
        assert active_cheby > active_noise, (
            f"Cheby {active_cheby} should beat noise {active_noise}")

    def test_stack_cheby_init_non_constant_output(self):
        """After cheby_init, stack output must have meaningful variance."""
        import sys; sys.path.insert(0,'src')
        from lut_native.targets import generate_data_2d
        data = generate_data_2d('feynman_2d', n_train=400, n_val=100, n_test=100, seed=42)
        xt = torch.from_numpy(data[0])
        m = LUTKANStack(dims=[2, 4, 1], K=16, L=32)
        m.cheby_init(scale=1.5, noise=0.05)
        m.calibrate(xt)
        with torch.no_grad():
            out = m(xt)
        assert float(out.std()) > 0.01, "Output near-constant after cheby_init"

    def test_train_config_cheby_default(self):
        """StackTrainConfig must default to cheby_init=True."""
        from lut_native.training_stack import StackTrainConfig
        cfg = StackTrainConfig()
        assert cfg.cheby_init is True
        assert cfg.cheby_scale == 1.5

    def test_train_with_cheby_improves_over_noise(self):
        """cheby_init training achieves lower MSE than noise-only init."""
        import sys; sys.path.insert(0,'src')
        from lut_native.targets import generate_data_2d
        from lut_native.training_stack import StackTrainConfig, train_lut_stack
        data = generate_data_2d('feynman_2d', n_train=400, n_val=100, n_test=100, seed=42)
        x_tr,y_tr,x_val,y_val,x_te,y_te = data
        y_tr=y_tr.reshape(-1,1); y_val=y_val.reshape(-1,1); y_te=y_te.reshape(-1,1)

        # Cheby init (recommended)
        m_cheby = LUTKANStack(dims=[2,4,1], K=16, L=32)
        cfg_cheby = StackTrainConfig(cheby_init=True, epochs=100, seed=0,
                                     eval_every_epochs=50)
        r_cheby = train_lut_stack(m_cheby, x_tr, y_tr, x_val, y_val, x_te, y_te, cfg_cheby)

        # Noise init (legacy)
        m_noise = LUTKANStack(dims=[2,4,1], K=16, L=32)
        cfg_noise = StackTrainConfig(cheby_init=False, init_noise_std=0.05,
                                     epochs=100, seed=0, eval_every_epochs=50)
        r_noise = train_lut_stack(m_noise, x_tr, y_tr, x_val, y_val, x_te, y_te, cfg_noise)

        assert r_cheby.mse_test_at_best < r_noise.mse_test_at_best, (
            f"Cheby {r_cheby.mse_test_at_best:.4f} should beat noise "
            f"{r_noise.mse_test_at_best:.4f}")
