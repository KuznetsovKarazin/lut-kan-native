"""
validate_before_flash.py
Run this BEFORE flashing to hardware. Simulates the exact same experiment
that lut_kan_hw_bench.ino will run, using PyTorch on your PC.

Expected output should match hardware output within ~2-3× on ratios.
Differences are expected because:
  - PC uses Adam (paper) vs this script uses SGD (hardware config)
  - float32 throughout vs hardware soft-float rounding
  - n_train=200 vs paper's n_train=500

Usage:
    pip install torch numpy
    python validate_before_flash.py
"""

import sys, time
import numpy as np

# ── Try importing torch; fall back to pure numpy SGD if not available ──────────
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not found — using pure NumPy SGD (slower but equivalent).")

# ── Config ────────────────────────────────────────────────────────────────────
KK       = 4
N_TRAIN  = 200
N_VAL    = 50
N_TEST   = 200
EPOCHS   = 800
LAMBDA2  = 1.0
L_VALUES = [8, 16, 32]

# Post-train LUT baseline MSE (from simulation, matches C constants)
POST_LUT_MSE = {8: 3.7201e-3, 16: 9.7252e-4, 32: 2.4582e-4}

# ── Target function ────────────────────────────────────────────────────────────
def target(x):
    return np.tanh(4.0 * x) + 0.15 * x

# ── LUT inference (NumPy, matches C implementation exactly) ──────────────────
def lut_infer_np(x, lut, K, L):
    """x: scalar or array, lut: (K,L) array."""
    x = np.float32(x)
    seg_w = 2.0 / K
    x = np.clip(x, -1.0, 1.0 - 1e-6)
    t = (x + 1.0) / seg_w
    k = np.clip(np.floor(t).astype(int), 0, K - 1)
    u = np.clip(t - k, 0.0, 1.0)
    pos = u * (L - 1)
    r0 = np.clip(np.floor(pos).astype(int), 0, L - 2)
    r1 = r0 + 1
    w = pos - r0
    return lut[k, r0] * (1 - w) + lut[k, r1] * w

# ── λ₂ gradient (NumPy, matches C exactly) ────────────────────────────────────
def second_diff_grad_np(lut, L):
    """Returns gradient array shape (K,L)."""
    K = lut.shape[0]
    n_terms = K * (L - 2)
    scale = 2.0 * LAMBDA2 / n_terms
    g = np.zeros_like(lut)
    d2 = lut[:, :-2] - 2 * lut[:, 1:-1] + lut[:, 2:]  # (K, L-2)
    # Accumulate contributions:
    # j-2 term: d2[:, j-2]  →  g[:, j] += d2[:, j-2]   for j ≥ 2
    # j-1 term: -2*d2[:,j-1] →  g[:, j] -= 2*d2[:,j-1]  for 1 ≤ j ≤ L-2
    # j term:   d2[:, j]     →  g[:, j] += d2[:, j]      for j ≤ L-3
    g[:, 2:]    += d2
    g[:, 1:L-1] -= 2 * d2
    g[:, :L-2]  += d2
    return scale * g

# ── Pure NumPy SGD training ────────────────────────────────────────────────────
def train_numpy(L, lr):
    K = KK
    lut = np.zeros((K, L), dtype=np.float32)
    best_lut = lut.copy()
    best_val = float('inf')
    best_ep  = 0

    x_train = np.linspace(-1, 1, N_TRAIN, dtype=np.float32)
    y_train = target(x_train).astype(np.float32)
    x_val   = np.linspace(-0.995, 0.995, N_VAL, dtype=np.float32)
    y_val   = target(x_val).astype(np.float32)
    x_test  = np.linspace(-1, 1, N_TEST, dtype=np.float32)
    y_test  = target(x_test).astype(np.float32)

    for ep in range(EPOCHS):
        grad = np.zeros((K, L), dtype=np.float32)

        # Data gradient
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            pred = lut_infer_np(x, lut, K, L)
            res  = pred - y
            # Compute indices
            t  = (x + 1.0) / (2.0 / K)
            k  = int(np.clip(t, 0, K - 1))
            u  = np.clip(t - k, 0.0, 1.0)
            p  = u * (L - 1)
            r0 = int(np.clip(p, 0, L - 2))
            r1 = r0 + 1
            w  = p - r0
            g  = 2.0 * res / N_TRAIN
            grad[k, r0] += g * (1.0 - w)
            grad[k, r1] += g * w

        # λ₂ gradient
        grad += second_diff_grad_np(lut, L)

        # SGD step
        lut -= lr * grad

        # Validation
        val_preds = np.array([lut_infer_np(x, lut, K, L) for x in x_val])
        val_mse   = float(np.mean((val_preds - y_val) ** 2))
        if val_mse < best_val:
            best_val = val_mse
            best_ep  = ep
            best_lut = lut.copy()

    lut = best_lut
    test_preds = np.array([lut_infer_np(x, lut, K, L) for x in x_test])
    test_mse   = float(np.mean((test_preds - y_test) ** 2))
    return test_mse, best_ep

# ── PyTorch training (faster, for quick verification) ─────────────────────────
def train_torch(L, lr):
    import torch, sys
    sys.path.insert(0, '.')
    K = KK
    lut_param = torch.zeros(K, L, requires_grad=True)
    x_train   = torch.linspace(-1, 1, N_TRAIN)
    y_train   = torch.from_numpy(target(x_train.numpy()).astype('float32'))
    x_val     = torch.linspace(-0.995, 0.995, N_VAL)
    y_val     = torch.from_numpy(target(x_val.numpy()).astype('float32'))
    x_test    = torch.linspace(-1, 1, N_TEST)
    y_test    = target(x_test.numpy()).astype('float32')

    opt = torch.optim.SGD([lut_param], lr=lr)
    best_val  = float('inf')
    best_ep   = 0
    best_state = lut_param.data.clone()

    def forward(x, lut, L):
        seg_w = 2.0 / K
        xc = torch.clamp(x, -1.0, 1.0 - 1e-6)
        t  = (xc + 1.0) / seg_w
        k  = torch.clamp(torch.floor(t).long(), 0, K - 1)
        u  = torch.clamp(t - k.float(), 0, 1)
        pos = u * (L - 1)
        r0 = torch.clamp(torch.floor(pos).long(), 0, L - 2)
        r1 = r0 + 1
        w  = pos - r0.float()
        return lut[k, r0] * (1 - w) + lut[k, r1] * w

    def l2_reg(lut):
        d2 = lut[:, 2:] - 2*lut[:, 1:-1] + lut[:, :-2]
        return (d2**2).mean()

    for ep in range(EPOCHS):
        opt.zero_grad()
        pred = forward(x_train, lut_param, L)
        loss = ((pred - y_train)**2).mean() + LAMBDA2 * l2_reg(lut_param)
        loss.backward()
        opt.step()

        with torch.no_grad():
            vp = forward(x_val, lut_param, L)
            vm = float(((vp - y_val)**2).mean())
        if vm < best_val:
            best_val  = vm
            best_ep   = ep
            best_state = lut_param.data.clone()

    lut_param.data.copy_(best_state)
    with torch.no_grad():
        tp = forward(x_test, lut_param, L).numpy()
    test_mse = float(np.mean((tp - target(x_test.numpy()).astype('float32'))**2))
    return test_mse, best_ep


# ── Main ───────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  lut-kan-native hardware validation (PC simulation)")
print(f"  K={KK}  N_TRAIN={N_TRAIN}  EPOCHS={EPOCHS}  LAMBDA2={LAMBDA2}")
print("=" * 55)
print(f"{'Backend:':<14} {'PyTorch SGD' if HAS_TORCH else 'NumPy SGD'}")
print()

results = {}
for L in L_VALUES:
    lr = L / 16.0    # same formula as C: L=8→0.5, L=16→1.0, L=32→2.0
    kxl     = KK * L
    density = N_TRAIN / kxl
    post    = POST_LUT_MSE[L]

    print(f"► K=4, L={L:2d}  K×L={kxl:3d}  density={density:.1f} pts/cell  lr={lr:.1f}")
    print(f"  Rule K×L < N_TRAIN? {'YES ✓' if kxl < N_TRAIN else 'NO ✗ (expect collapse)'}")

    t0 = time.time()
    if HAS_TORCH:
        mse, best_ep = train_torch(L, lr)
    else:
        mse, best_ep = train_numpy(L, lr)
    elapsed = time.time() - t0

    ratio = post / mse
    results[L] = (mse, ratio)
    print(f"  Training time  : {elapsed:.1f} s")
    print(f"  Best epoch     : {best_ep}")
    print(f"  Direct-LUT MSE : {mse:.4e}")
    print(f"  Post-LUT   MSE : {post:.4e}")
    print(f"  Ratio          : {ratio:.1f}×")
    if L == 8:  print(f"  Expected       : ~25×  (sim: 23× at n=500)")
    if L == 16: print(f"  Expected       : ~800× (sim: 936× at n=500)")
    if L == 32: print(f"  Expected       : ~20000× (sim: 21012× at n=500)")
    print()

print("=" * 55)
print("  Flash lut_kan_hw_bench.ino and compare hardware output")
print("  with these numbers. Ratios should agree within ~3×.")
print("=" * 55)
