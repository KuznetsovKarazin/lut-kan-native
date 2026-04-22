"""
Verify that gradients through the LUT forward match numerical gradients.
This is the acid test for any differentiable operation.
"""

from __future__ import annotations

import numpy as np
import torch

from lut_native import LUTEdge


def test_gradcheck_small():
    """torch.autograd.gradcheck on the LUT forward (requires double precision).

    Implemented functionally: the 'function under test' takes the LUT parameter
    as input and performs the same gather+lerp as LUTEdge.forward. We avoid
    nn.Parameter here because swapping .data in and out of a nn.Parameter
    detaches the autograd graph (gradcheck would see analytical grad = 0).
    """
    K, L = 4, 6
    x_min, x_max = -1.0, 1.0
    rng = np.random.RandomState(0)
    lut0 = torch.from_numpy(rng.randn(K, L)).double().requires_grad_(True)
    x = torch.tensor([-0.9, -0.3, 0.0, 0.42, 0.75], dtype=torch.float64)

    def fn(lut):
        # Same math as LUTEdge.forward, in double precision
        seg_width = (x_max - x_min) / K
        hi = torch.tensor(x_max - 1e-7, dtype=torch.float64)
        x_clip = torch.clamp(x, x_min, hi)
        t = (x_clip - x_min) / seg_width
        k = torch.clamp(torch.floor(t).long(), 0, K - 1)
        u = torch.clamp(t - k.to(t.dtype), 0.0, 1.0)
        pos = u * (L - 1)
        r0 = torch.clamp(torch.floor(pos).long(), 0, L - 1)
        r1 = torch.clamp(r0 + 1, 0, L - 1)
        w = pos - r0.to(pos.dtype)
        v0 = lut[k, r0]
        v1 = lut[k, r1]
        return v0 * (1.0 - w) + v1 * w

    assert torch.autograd.gradcheck(
        fn, (lut0,), eps=1e-6, atol=1e-4, rtol=1e-3,
        check_undefined_grad=False,
    )


def test_gradient_reaches_visited_cells_only():
    """A single sample should generate non-zero grad on exactly 2 cells
    (linear interpolation endpoints). No gradient should leak elsewhere."""
    K, L = 4, 8
    edge = LUTEdge(K=K, L=L, x_min=-1.0, x_max=1.0)
    # Place x squarely in segment 2, between positions 3 and 4
    x = torch.tensor([-1.0 + 2.5 * 0.5 + (3.5 / (L - 1)) * 0.5], dtype=torch.float32)
    # Manual: seg_width = 2/4 = 0.5, k=2 means x in [0.0, 0.5)
    x = torch.tensor([0.0 + 0.5 * 3.5 / (L - 1)], dtype=torch.float32)
    y = edge(x).sum()
    y.backward()
    grad = edge.lut.grad.detach().cpu().numpy()
    # Exactly 2 non-zero cells (we land between indices 3 and 4 of segment 2)
    nonzero = (grad != 0).sum()
    assert nonzero == 2, f"Expected 2 non-zero grad cells, got {nonzero}"


def test_gradient_sum_equals_output_sensitivity():
    """d(sum(y))/d(lut) summed over all cells should equal N (one y per sample,
    with interpolation weights summing to 1 per sample)."""
    K, L = 4, 8
    edge = LUTEdge(K=K, L=L, x_min=-1.0, x_max=1.0)
    x = torch.linspace(-0.9, 0.9, 50, dtype=torch.float32)
    y = edge(x).sum()
    y.backward()
    grad_total = edge.lut.grad.sum().item()
    assert abs(grad_total - 50.0) < 1e-4, (
        f"Expected grad sum = N = 50, got {grad_total}"
    )
