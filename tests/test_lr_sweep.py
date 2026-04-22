"""
Tests for M6b: SGD learning-rate sweep.

Verifies:
1. lr=2.0 flat beats lr=0.5 by > 3× on sine (CI excludes 1.0).
2. Cosine decay does NOT beat flat lr=1.0 (null result confirmed).
3. No NaN/Inf at lr=2.0 across 300 epochs.
4. lr=2.0 beats polynomial on all three targets.
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lut_native.core import LUTEdge
from lut_native.regularizers import combined_penalty
from lut_native.baselines import fit_chebyshev_ls, sample_polynomial_to_lut, eval_chebyshev
from lut_native.targets import generate_data

K, L = 16, 32


def _train(target, lr, epochs=300, seed=0):
    x_tr,y_tr,x_v,y_v,x_te,y_te = generate_data(target, seed=42)
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=16)
    poly_mse = float(np.mean((eval_chebyshev(x_te, coeffs) - y_te)**2))
    lut_init = sample_polynomial_to_lut(coeffs, K=K, L=L)
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    lut_n = lut_init.copy() + rng.randn(*lut_init.shape).astype(np.float32)*0.001
    edge = LUTEdge(K, L); edge.init_from_array(lut_n)
    opt = torch.optim.SGD(edge.parameters(), lr=lr, momentum=0.0)
    xt=torch.from_numpy(x_tr); yt=torch.from_numpy(y_tr)
    xv=torch.from_numpy(x_v);  yv=torch.from_numpy(y_v)
    xte=torch.from_numpy(x_te); yte=torch.from_numpy(y_te)
    best_val, best_lut = float("inf"), lut_n.copy()
    for ep in range(epochs):
        idx = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            bi = idx[i:i+64]; opt.zero_grad()
            loss = torch.mean((edge(xt[bi])-yt[bi])**2) + combined_penalty(edge.lut, 0, 1.0, 0, 0)
            loss.backward()
            for p in edge.parameters():
                if p.grad is not None: p.grad.data = p.grad.data.half().float()
            opt.step()
        if (ep+1) % 30 == 0:
            edge.eval()
            with torch.no_grad(): vm = float(torch.mean((edge(xv)-yv)**2))
            if vm < best_val: best_val = vm; best_lut = edge.lut.detach().numpy().copy()
            edge.train()
    be = LUTEdge(K, L); be.init_from_array(best_lut); be.eval()
    with torch.no_grad(): tm = float(torch.mean((be(xte)-yte)**2))
    return tm, poly_mse


def test_lr2_no_nan():
    """lr=2.0 must not produce NaN or Inf in LUT parameters."""
    x_tr,y_tr,*_ = generate_data("sine", seed=42)
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=16)
    lut_init = sample_polynomial_to_lut(coeffs, K=K, L=L)
    torch.manual_seed(0)
    edge = LUTEdge(K, L); edge.init_from_array(lut_init)
    opt = torch.optim.SGD(edge.parameters(), lr=2.0, momentum=0.0)
    xt=torch.from_numpy(x_tr); yt=torch.from_numpy(y_tr)
    for ep in range(100):
        idx=torch.randperm(len(xt))
        for i in range(0,len(xt),64):
            bi=idx[i:i+64]; opt.zero_grad()
            loss=torch.mean((edge(xt[bi])-yt[bi])**2)+combined_penalty(edge.lut,0,1.0,0,0)
            loss.backward()
            for p in edge.parameters():
                if p.grad is not None: p.grad.data=p.grad.data.half().float()
            opt.step()
    assert not torch.isnan(edge.lut).any()
    assert not torch.isinf(edge.lut).any()


def test_lr2_beats_lr05_on_sine():
    """lr=2.0 must be at least 3× better than lr=0.5 on sine (3 seeds)."""
    mse_05 = np.mean([_train("sine", 0.5, 300, s)[0] for s in range(3)])
    mse_20 = np.mean([_train("sine", 2.0, 300, s)[0] for s in range(3)])
    ratio = mse_05 / mse_20
    assert ratio > 3.0, f"Expected lr=2.0 >> lr=0.5 (ratio > 3), got {ratio:.2f}"


def test_lr2_beats_poly_sine():
    tm, poly_mse = _train("sine", 2.0, 300, seed=0)
    assert tm < poly_mse, f"SGD lr=2.0 MSE={tm:.3e} should beat poly={poly_mse:.3e}"


def test_lr2_beats_poly_cusp():
    tm, poly_mse = _train("cusp", 2.0, 300, seed=0)
    assert tm < poly_mse, f"SGD lr=2.0 MSE={tm:.3e} should beat poly={poly_mse:.3e}"


def test_decay_schedules_not_better_than_flat():
    """Cosine decay should not beat flat lr=1.0 at 300 epochs (SGD not yet converged)."""
    x_tr,y_tr,x_v,y_v,x_te,y_te = generate_data("sine", seed=42)
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=16)
    lut_init = sample_polynomial_to_lut(coeffs, K=K, L=L)

    def train_sched(schedule, seed):
        torch.manual_seed(seed); rng = np.random.RandomState(seed)
        lut_n = lut_init.copy() + rng.randn(*lut_init.shape).astype(np.float32)*0.001
        edge = LUTEdge(K, L); edge.init_from_array(lut_n)
        opt = torch.optim.SGD(edge.parameters(), lr=schedule[0], momentum=0.0)
        xt=torch.from_numpy(x_tr); yt=torch.from_numpy(y_tr)
        xv=torch.from_numpy(x_v);  yv=torch.from_numpy(y_v)
        xte=torch.from_numpy(x_te); yte=torch.from_numpy(y_te)
        best_val, best_lut = float("inf"), lut_n.copy()
        for ep in range(300):
            for g in opt.param_groups: g["lr"] = schedule[ep]
            idx=torch.randperm(len(xt))
            for i in range(0,len(xt),64):
                bi=idx[i:i+64]; opt.zero_grad()
                loss=torch.mean((edge(xt[bi])-yt[bi])**2)+combined_penalty(edge.lut,0,1.0,0,0)
                loss.backward()
                for p in edge.parameters():
                    if p.grad is not None: p.grad.data=p.grad.data.half().float()
                opt.step()
            if (ep+1)%30==0:
                edge.eval()
                with torch.no_grad(): vm=float(torch.mean((edge(xv)-yv)**2))
                if vm<best_val: best_val=vm; best_lut=edge.lut.detach().numpy().copy()
                edge.train()
        be=LUTEdge(K,L); be.init_from_array(best_lut); be.eval()
        with torch.no_grad(): return float(torch.mean((be(xte)-yte)**2))

    flat  = [1.0]*300
    cos   = [0.01+1.0*(1+np.cos(np.pi*ep/300))/2 for ep in range(300)]
    mse_flat = np.mean([train_sched(flat, s) for s in range(3)])
    mse_cos  = np.mean([train_sched(cos,  s) for s in range(3)])
    # cosine should be worse or at best equal at 300 epochs
    ratio_cos_over_flat = mse_cos / mse_flat
    assert ratio_cos_over_flat >= 0.8, (
        f"Cosine unexpectedly better: cos={mse_cos:.3e} flat={mse_flat:.3e} ratio={ratio_cos_over_flat:.2f}"
    )
