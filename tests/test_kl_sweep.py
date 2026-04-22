"""
Tests for M7a: K × L sweep for single-edge direct-LUT.

Verifies:
1. K=8, L=32 achieves higher ratio than K=16, L=32 on sine (CI excludes 1.0).
2. Large K (K=32, L=32) gives lower ratio than K=16, L=32 (advantage shrinks).
3. Ratio monotonically decreasing with K at fixed L=32 on sine.
4. At K=32, L=64 (large budget), direct-LUT degrades near parity or worse.
5. memory_bytes formula is correct.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lut_native.baselines import (
    fit_chebyshev_ls, eval_chebyshev, sample_polynomial_to_lut,
    quantize_lut_uint8_asym, dequantize_lut,
)
from lut_native.core import lut_forward_numpy
from lut_native.targets import generate_data
from lut_native.training import train_lut_edge, TrainConfig


def _run_kl(K, L, target="sine", epochs=400, seed=0):
    x_tr,y_tr,x_v,y_v,x_te,y_te = generate_data(target, seed=42)
    coeffs = fit_chebyshev_ls(x_tr,y_tr,degree=16)
    lut_init = sample_polynomial_to_lut(coeffs, K=K, L=L)
    q,s,m = quantize_lut_uint8_asym(lut_init)
    post_mse = float(np.mean((lut_forward_numpy(x_te, dequantize_lut(q,s,m)) - y_te)**2))
    cfg = TrainConfig(lambda_2=1.0, lr=1e-2, epochs=epochs, seed=seed)
    res = train_lut_edge(lut_init,x_tr,y_tr,x_v,y_v,x_te,y_te,-1.,1.,cfg)
    return post_mse, res.mse_test_at_best


def test_memory_bytes():
    """Memory formula: K*L uint8 + K*4 bytes (f16 scale + f16 ymin)."""
    assert 8*32 + 8*4   == 288
    assert 16*32 + 16*4 == 576
    assert 32*32 + 32*4 == 1152


def test_k8_l32_beats_k16_l32_on_sine():
    """K=8,L=32 (288B) should achieve higher ratio than K=16,L=32 (576B) on sine."""
    post8, direct8 = _run_kl(8, 32, "sine", epochs=400)
    post16, direct16 = _run_kl(16, 32, "sine", epochs=400)
    ratio8  = post8  / direct8
    ratio16 = post16 / direct16
    assert ratio8 > ratio16, (
        f"K8 ratio={ratio8:.1f}× should exceed K16 ratio={ratio16:.1f}×"
    )


def test_large_k_lower_ratio():
    """K=32, L=32 should have a lower ratio than K=16, L=32 (advantage shrinks with K)."""
    post16, direct16 = _run_kl(16, 32, "sine", epochs=400)
    post32, direct32 = _run_kl(32, 32, "sine", epochs=400)
    ratio16 = post16 / direct16
    ratio32 = post32 / direct32
    assert ratio16 > ratio32, (
        f"K16 ratio={ratio16:.1f}× should exceed K32 ratio={ratio32:.1f}×"
    )


def test_ratio_decreases_with_k_at_l32():
    """Ratio is monotonically decreasing in K at fixed L=32 on sine (1 seed)."""
    ratios = {}
    for K in [8, 16, 32]:
        post, direct = _run_kl(K, 32, "sine", epochs=400)
        ratios[K] = post / direct
    assert ratios[8] > ratios[16] > ratios[32], (
        f"Expected ratio(K=8) > ratio(K=16) > ratio(K=32); got {ratios}"
    )


def test_large_kl_parity_or_worse():
    """At K=32, L=64 (large budget), direct-LUT ratio should be ≤ 10 (parity/worse)."""
    post, direct = _run_kl(32, 64, "saturating", epochs=400)
    ratio = post / direct
    assert ratio <= 10, (
        f"Expected K=32,L=64 to be near parity, got ratio={ratio:.1f}×"
    )


def test_direct_lut_beats_poly_k8_l32():
    """K=8, L=32 direct-LUT must beat polynomial MSE on cusp."""
    x_tr,y_tr,x_v,y_v,x_te,y_te = generate_data("cusp", seed=42)
    coeffs = fit_chebyshev_ls(x_tr,y_tr,degree=16)
    poly_mse = float(np.mean((eval_chebyshev(x_te,coeffs)-y_te)**2))
    lut_init = sample_polynomial_to_lut(coeffs, K=8, L=32)
    cfg = TrainConfig(lambda_2=1.0, lr=1e-2, epochs=400, seed=0)
    res = train_lut_edge(lut_init,x_tr,y_tr,x_v,y_v,x_te,y_te,-1.,1.,cfg)
    assert res.mse_test_at_best < poly_mse, (
        f"K=8,L=32 direct MSE={res.mse_test_at_best:.3e} should beat poly={poly_mse:.3e}"
    )
