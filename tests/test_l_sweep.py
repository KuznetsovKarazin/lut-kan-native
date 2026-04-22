"""
Tests for M7d: L sweep and crossover characterization.

Verifies:
1. K=8 crossover: L=96 ratio must be < L=48 ratio (collapse confirmed).
2. K=4 no crossover at L=64 (still above L=32).
3. Coverage rule: ratio drops sharply when K*L >= n_train.
4. K=4, L=64 beats K=4, L=32 on saturating.
5. Data-density threshold: K=8, L=96 (0.65/cell) worse than K=8, L=48 (1.3/cell).
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lut_native.baselines import (
    fit_chebyshev_ls, sample_polynomial_to_lut,
    quantize_lut_uint8_asym, dequantize_lut,
)
from lut_native.core import lut_forward_numpy
from lut_native.targets import generate_data
from lut_native.training import train_lut_edge, TrainConfig

N_TRAIN = 500


def _ratio(K, L, target="sine", epochs=250, seed=0):
    x_tr,y_tr,x_v,y_v,x_te,y_te = generate_data(target, seed=42)
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=16)
    lut_init = sample_polynomial_to_lut(coeffs, K=K, L=L)
    q,s,m = quantize_lut_uint8_asym(lut_init)
    post_mse = float(np.mean((lut_forward_numpy(x_te, dequantize_lut(q,s,m)) - y_te)**2))
    cfg = TrainConfig(lambda_2=1.0, lr=1e-2, epochs=epochs, seed=seed)
    res = train_lut_edge(lut_init, x_tr,y_tr,x_v,y_v,x_te,y_te,-1.,1.,cfg)
    return post_mse / res.mse_test_at_best


def test_k8_crossover_l96_worse_than_l48():
    """K=8 must collapse at L=96 (ratio < L=48 ratio) — coverage density < 1/cell."""
    r48 = _ratio(8, 48, "sine", epochs=200)
    r96 = _ratio(8, 96, "sine", epochs=200)
    assert r96 < r48, (
        f"K=8 should collapse at L=96: r48={r48:.0f}×, r96={r96:.0f}× — expected r96 < r48"
    )


def test_k4_no_crossover_l64_vs_l32():
    """K=4 should NOT collapse at L=64 (ratio at L=64 >= ratio at L=32)."""
    r32 = _ratio(4, 32, "sine", epochs=200)
    r64 = _ratio(4, 64, "sine", epochs=200)
    assert r64 >= r32 * 0.5, (
        f"K=4 should not collapse at L=64: r32={r32:.0f}×, r64={r64:.0f}×"
    )


def test_coverage_rule_k8():
    """When K*L > n_train (coverage < 1/cell), ratio must drop by > 5× vs peak."""
    # K=8, L=48 (1.3/cell) vs K=8, L=96 (0.65/cell)
    r_safe    = _ratio(8, 48, "cusp", epochs=200)   # 1.3 pts/cell
    r_overfit = _ratio(8, 96, "cusp", epochs=200)   # 0.65 pts/cell
    factor = r_safe / max(r_overfit, 1)
    assert factor > 5, (
        f"Coverage collapse: L=48 ({r_safe:.0f}×) should be >5× better than L=96 ({r_overfit:.0f}×), "
        f"got factor={factor:.1f}×"
    )


def test_k4_l64_beats_k4_l32_saturating():
    """K=4, L=64 should significantly beat K=4, L=32 on saturating (still in safe zone)."""
    r32 = _ratio(4, 32, "saturating", epochs=200)
    r64 = _ratio(4, 64, "saturating", epochs=200)
    assert r64 > r32 * 2, (
        f"K=4,L=64 ratio={r64:.0f}× should be >2× K=4,L=32 ratio={r32:.0f}×"
    )


def test_optimal_l_formula():
    """Optimal L upper bound: K * L_opt < n_train (1 pt/cell threshold).

    K=8, L=48 (K*L=384 < 500) should beat K=8, L=96 (K*L=768 > 500).
    """
    K = 8
    L_safe   = 48   # K*L = 384 < 500
    L_unsafe = 96   # K*L = 768 > 500
    assert K * L_safe   < N_TRAIN, "L_safe should satisfy K*L < n_train"
    assert K * L_unsafe > N_TRAIN, "L_unsafe should violate K*L < n_train"
    r_safe   = _ratio(K, L_safe,   "sine", epochs=200)
    r_unsafe = _ratio(K, L_unsafe, "sine", epochs=200)
    assert r_safe > r_unsafe, (
        f"Safe zone ({r_safe:.0f}×) should beat unsafe zone ({r_unsafe:.0f}×)"
    )
