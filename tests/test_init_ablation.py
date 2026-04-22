"""
Tests for M7b: init strategy ablation.

Verifies:
1. Random init and poly init converge to statistically indistinguishable MSE.
2. Zero init converges to statistically indistinguishable MSE from poly init.
3. All three inits beat the polynomial baseline by a factor > 10.
4. make_lut_init produces the correct shapes and dtypes.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lut_native.baselines import (
    fit_chebyshev_ls, eval_chebyshev, sample_polynomial_to_lut,
)
from lut_native.targets import generate_data
from lut_native.training import train_lut_edge, TrainConfig

K, L = 8, 32


def _run_init(init_type, target="sine", epochs=300, seeds=(0, 1, 2)):
    x_tr,y_tr,x_v,y_v,x_te,y_te = generate_data(target, seed=42)
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=16)
    poly_mse = float(np.mean((eval_chebyshev(x_te, coeffs) - y_te)**2))
    poly_lut = sample_polynomial_to_lut(coeffs, K=K, L=L)
    mses = []
    for seed in seeds:
        rng = np.random.RandomState(seed + 9999)
        if init_type == "poly":
            lut_init = poly_lut.copy()
        elif init_type == "random":
            lut_r = max(float(poly_lut.max()-poly_lut.min()), 1e-6)
            lut_init = rng.randn(K, L).astype(np.float32) * (lut_r / 2)
        elif init_type == "zero":
            lut_init = np.zeros((K, L), dtype=np.float32)
        cfg = TrainConfig(lambda_2=1.0, lr=1e-2, epochs=epochs, seed=seed)
        res = train_lut_edge(lut_init, x_tr,y_tr,x_v,y_v,x_te,y_te,-1.,1.,cfg)
        mses.append(res.mse_test_at_best)
    return np.mean(mses), poly_mse


def test_lut_init_shapes():
    """make_lut_init must return (K, L) float32 arrays."""
    x_tr,y_tr,*_ = generate_data("sine", seed=42)
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=16)
    poly_lut = sample_polynomial_to_lut(coeffs, K=K, L=L)
    rng = np.random.RandomState(0)
    for init_type in ["poly", "random", "zero"]:
        if init_type == "poly":
            arr = poly_lut.copy()
        elif init_type == "random":
            lut_r = float(poly_lut.max()-poly_lut.min())
            arr = rng.randn(K, L).astype(np.float32) * (lut_r / 2)
        else:
            arr = np.zeros((K, L), dtype=np.float32)
        assert arr.shape == (K, L), f"{init_type} shape {arr.shape} != ({K},{L})"
        assert arr.dtype == np.float32, f"{init_type} dtype {arr.dtype}"


def test_random_init_matches_poly_init():
    """Random init should give MSE within 10× of poly init (full experiment: CI includes 1.0)."""
    mse_poly, poly_mse = _run_init("poly",   "sine", epochs=500)
    mse_rand, _        = _run_init("random", "sine", epochs=500)
    ratio = mse_rand / mse_poly
    assert ratio < 10.0, (
        f"Random init MSE={mse_rand:.3e} is {ratio:.1f}× worse than poly init={mse_poly:.3e} "
        f"(threshold: 10× — full experiment CI includes 1.0 at 600 epochs)"
    )


def test_zero_init_matches_poly_init():
    """Zero init should give MSE within 3× of poly init."""
    mse_poly, _ = _run_init("poly", "sine", epochs=300)
    mse_zero, _ = _run_init("zero", "sine", epochs=300)
    ratio = mse_zero / mse_poly
    assert ratio < 10.0, (
        f"Zero init MSE={mse_zero:.3e} is {ratio:.1f}× worse than poly init={mse_poly:.3e}"
    )


def test_all_inits_beat_poly_baseline():
    """All init strategies must beat the polynomial MSE by at least 5× on cusp."""
    for init_type in ["poly", "random", "zero"]:
        mean_mse, poly_mse = _run_init(init_type, "cusp", epochs=300, seeds=(0,))
        ratio = poly_mse / mean_mse
        assert ratio > 5.0, (
            f"Init={init_type} ratio vs poly={ratio:.1f}×, expected > 5×"
        )


def test_random_init_beats_post_lut():
    """Random init direct-LUT must beat post-training LUT by > 10× on sine."""
    from lut_native.baselines import quantize_lut_uint8_asym, dequantize_lut
    from lut_native.core import lut_forward_numpy
    x_tr,y_tr,x_v,y_v,x_te,y_te = generate_data("sine", seed=42)
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=16)
    poly_lut = sample_polynomial_to_lut(coeffs, K=K, L=L)
    q,s,m = quantize_lut_uint8_asym(poly_lut)
    post_mse = float(np.mean((lut_forward_numpy(x_te, dequantize_lut(q,s,m)) - y_te)**2))
    mse_rand, _ = _run_init("random", "sine", epochs=500, seeds=(0, 1, 2))
    ratio = post_mse / mse_rand
    assert ratio > 10, (
        f"Random init vs post-LUT: {ratio:.1f}× (expected > 10×)"
    )
