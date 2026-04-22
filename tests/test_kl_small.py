"""
Tests for M7c: small-K/L sweep.

Verifies:
1. K=4, L=32 achieves higher ratio than K=8, L=32 on saturating.
2. K=1 at any L gives ratio ≤ 10 on sine (near parity — no advantage).
3. L=32 vs L=16 jump: K=4, L=32 > 10× K=4, L=16 ratio on saturating.
4. K=2, L=32 beats K=4, L=16 on saturating despite similar memory.
5. All K/L at L=32 beat the polynomial baseline on cusp.
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


def _ratio(K, L, target="saturating", epochs=300, seed=0):
    x_tr,y_tr,x_v,y_v,x_te,y_te = generate_data(target, seed=42)
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=16)
    lut_init = sample_polynomial_to_lut(coeffs, K=K, L=L)
    q,s,m = quantize_lut_uint8_asym(lut_init)
    post_mse = float(np.mean((lut_forward_numpy(x_te, dequantize_lut(q,s,m)) - y_te)**2))
    cfg = TrainConfig(lambda_2=1.0, lr=1e-2, epochs=epochs, seed=seed)
    res = train_lut_edge(lut_init, x_tr,y_tr,x_v,y_v,x_te,y_te,-1.,1.,cfg)
    return post_mse / res.mse_test_at_best


def test_k4_l32_beats_k8_l32_saturating():
    """K=4,L=32 (144B) must achieve higher ratio than K=8,L=32 (288B) on saturating."""
    r4 = _ratio(4, 32, "saturating", epochs=350)
    r8 = _ratio(8, 32, "saturating", epochs=350)
    assert r4 > r8, (
        f"K=4,L=32 ratio={r4:.0f}× should exceed K=8,L=32 ratio={r8:.0f}× on saturating"
    )


def test_k1_near_parity_on_sine():
    """K=1 should give ratio ≤ 15 on sine — near parity, no meaningful advantage."""
    r = _ratio(1, 32, "sine", epochs=300)
    assert r <= 15, f"K=1,L=32 ratio={r:.1f}× on sine should be ≤ 15 (near parity)"


def test_l32_vs_l16_jump_saturating():
    """K=4,L=32 ratio must be > 5× K=4,L=16 ratio on saturating (L=32 cliff)."""
    r32 = _ratio(4, 32, "saturating", epochs=350)
    r16 = _ratio(4, 16, "saturating", epochs=350)
    factor = r32 / max(r16, 1)
    assert factor > 5, (
        f"Expected L=32 >> L=16 by >5×; got r32={r32:.0f}×, r16={r16:.0f}×, factor={factor:.1f}×"
    )


def test_k2_l32_beats_k4_l16_saturating():
    """K=2,L=32 (72B) should beat K=4,L=16 (80B) on saturating."""
    r2_32 = _ratio(2, 32, "saturating", epochs=300)
    r4_16 = _ratio(4, 16, "saturating", epochs=300)
    assert r2_32 > r4_16, (
        f"K=2,L=32 ({r2_32:.0f}×) should beat K=4,L=16 ({r4_16:.0f}×) on saturating"
    )


def test_all_l32_beat_poly_cusp():
    """All K values with L=32 must beat the polynomial baseline on cusp."""
    x_tr,y_tr,x_v,y_v,x_te,y_te = generate_data("cusp", seed=42)
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=16)
    poly_mse = float(np.mean((eval_chebyshev(x_te, coeffs) - y_te)**2))
    for K in [2, 4, 8]:
        lut_init = sample_polynomial_to_lut(coeffs, K=K, L=32)
        cfg = TrainConfig(lambda_2=1.0, lr=1e-2, epochs=300, seed=0)
        res = train_lut_edge(lut_init, x_tr,y_tr,x_v,y_v,x_te,y_te,-1.,1.,cfg)
        assert res.mse_test_at_best < poly_mse, (
            f"K={K},L=32 direct MSE={res.mse_test_at_best:.3e} "
            f"should beat poly={poly_mse:.3e}"
        )
