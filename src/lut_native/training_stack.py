"""
Training loop for LUTKANStack (N-layer LUT-KAN with adaptive norms).

Mirrors training_kan2.py conventions:
  - Adam optimizer, mini-batch SGD
  - Val-based best-model selection (snapshot of LUTs + norm params)
  - Optional λ₁/λ₂ regularization applied to every LUT block
  - Calibration call before training (sets norms from data percentiles)
  - Optional periodic re-calibration if activations drift

Key additions vs KAN2 training:
  1. Two parameter groups: LUTs (default lr) and norm params (norm_lr_scale × lr).
     Norm parameters typically converge faster; a higher LR helps them track
     activation drift without overshooting on the LUT tables.
  2. Snapshot restores both LUT values AND norm params (shift, log_scale),
     because the best model is only meaningful with its paired normalisation.
  3. Coverage stats are optionally tracked in the trace for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .kan_stack import LUTKANStack, coverage_entropy_loss, norm_coverage_loss


# ─────────────────────────────────────────────────────────────────────────────
# Config / Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StackTrainConfig:
    # Regularisation on LUT curvature (same semantics as KAN2TrainConfig)
    lambda_1: float = 0.0          # first-diff penalty weight
    lambda_2: float = 0.0          # second-diff penalty weight

    # Optimiser
    lr: float = 1e-2
    epochs: int = 2000
    batch_size: int = 64

    # Initialisation strategy.
    #
    # cheby_init=True (RECOMMENDED): initialise LUT cells with Chebyshev
    #   polynomial basis functions (T_0, T_1, ..., T_{out_dim-1}) scaled so
    #   that hidden activations span the full downstream LUT domain from step 0.
    #   Fixes the "dead init" failure where std=0.05 noise only activates 4/16
    #   segments, causing the model to predict a constant (MSE ≈ Var(y)) for
    #   all of training.
    # cheby_init=False: use Gaussian noise only (legacy, std=init_noise_std).
    cheby_init:       bool  = True
    cheby_scale:      float = 1.5   # Chebyshev amplitude; 1.5 covers all K segs
    cheby_noise:      float = 0.05  # per-cell noise added after Cheby values

    # Gaussian noise std added to LUT tables (only used when cheby_init=False).
    init_noise_std: float = 0.05

    # RNG
    seed: int = 0

    # Evaluation cadence
    eval_every_epochs: int = 10

    # Calibration: call LUTKANStack.calibrate() once before training on x_train.
    # This sets shift/log_scale in each LUTInterLayerNorm so activations cover
    # the LUT domain.  Should almost always be True.
    calibrate_at_start: bool = True

    # Re-calibration: if > 0, recalibrate every N epochs.
    # Useful when initial calibration drifts due to large LUT updates.
    # Off by default — only enable if coverage diagnostics show degradation.
    recalibrate_every_epochs: int = 0

    # Smooth norm squash (default True — RECOMMENDED).
    # When True, LUTKANStack is created with smooth_norms=True (tanh squash).
    # When False, hard clamp is used (legacy, gradient dies in deep stacks).
    smooth_norms: bool = True

    # Norm training mode.
    #
    # freeze_norms=True (default, RECOMMENDED):
    #   Norm parameters (shift, log_scale) are excluded from the optimizer.
    #   requires_grad is set to False, so no gradient is computed for them,
    #   but gradient still flows THROUGH the norm to earlier LUT parameters.
    #   Norms are periodically re-calibrated from data (see recalibrate_every_epochs).
    #
    # freeze_norms=False (legacy, not recommended):
    #   Norms are in the Adam optimizer at lr * norm_lr_scale.
    #   Risk: optimizer minimises MSE by shrinking scale until all activations
    #   clamp to the domain boundary (gradient through clamp = 0), permanently
    #   freezing norms at poor coverage. Confirmed empirically in H8.
    freeze_norms: bool = True
    norm_lr_scale: float = 2.0   # Only used when freeze_norms=False.

    # Re-calibrate norms every N epochs from x_train activations.
    # When freeze_norms=True (default), recalibration is the ONLY way norms
    # change after startup — it tracks LUT drift at negligible cost.
    # Set to 0 to disable (calibrate_at_start only).
    recalibrate_every_epochs: int = 100

    # Per-block learning rate scales.
    #
    # Gradient imbalance in deep LUT-KAN stacks: block[0] receives ~6000×
    # larger gradients than block[N-1] because gradients amplify through LUT
    # slopes going backward. Training block[0] with full LR causes thrashing
    # (it destabilises the representation that deeper blocks rely on).
    #
    # block_lr_scales: list of multipliers, one per block. Applied on top of
    # base `lr`. Default None = uniform lr across all blocks.
    #
    # Recommended for 3+ block stacks:
    #   block_lr_scales=[0.1, 0.1, 1.0]   (damp all early blocks equally)
    # NOT recommended: geometric [0.01, 0.1, 1.0] — over-damps block[0];
    # empirically [0.1, 0.1, 1.0] gives 12% lower MSE on feynman_2d (H8 exp).
    # or let the helper auto_block_lr_scales() compute from measured gradients.
    #
    # For 1- and 2-block stacks: gradient imbalance is modest; leave as None.
    block_lr_scales: Optional[List[float]] = None

    # EMA norm tracking (the H8b mechanism).
    # When freeze_norms=True and ema_alpha > 0, norm shift/log_scale are
    # updated every batch via exponential moving average of batch statistics.
    # This is continuous recalibration — equivalent to calibrate() every step
    # but batched and smooth. Replaces or supplements recalibrate_every_epochs.
    #
    # EMA update: shift ← (1-α)*shift + α*batch_center
    #             log_scale ← (1-α)*log_scale + α*log(batch_half/domain_half)
    #
    # alpha=0.0 disables EMA (use recalibrate_every_epochs only).
    # alpha=0.05 tracks drift smoothly; alpha=0.2 reacts fast but is noisy.
    # Typical good values: 0.01–0.05.
    ema_alpha: float = 0.01

    # Coverage entropy regulariser (only effective when freeze_norms=False).
    # NOTE: all gradient-based coverage losses are known to be unreliable
    # when norms have collapsed (sigmoid saturation, mixed gradient signs).
    # Use ema_alpha with freeze_norms=True instead.
    # Kept for ablation experiments.
    lambda_cov: float = 0.0

    # If True, log coverage stats (segment uniformity) to trace every eval.
    # Adds a small overhead (one forward on x_val per eval).
    track_coverage: bool = False


@dataclass
class StackTrainResult:
    # Best-epoch model (LUT arrays + norm param dicts)
    best_luts:  List[np.ndarray]
    best_norms: List[dict]

    # Final-epoch model
    final_luts:  List[np.ndarray]
    final_norms: List[dict]

    mse_train_at_best: float
    mse_val_at_best:   float
    mse_test_at_best:  float
    mse_val_final:     float

    best_epoch: int
    trace: Dict           # {"epoch", "mse_train", "mse_val", ["coverage"]}
    n_updates: int


# ─────────────────────────────────────────────────────────────────────────────
# Regularisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _stack_reg(
    model: LUTKANStack,
    lambda_1: float,
    lambda_2: float,
) -> torch.Tensor:
    """
    Sum first-/second-diff penalties over all LUTs in all blocks,
    averaged per-cell so the scale is comparable to single-edge training.
    """
    loss = torch.zeros((), dtype=torch.float32)
    if lambda_1 == 0.0 and lambda_2 == 0.0:
        return loss

    for blk in model.blocks:
        # lut: (in_d, out_d, K, L) → flatten edges to (n_edges, K, L)
        flat = blk.lut.view(-1, blk.K, blk.L)  # (n_edges, K, L)
        if lambda_1 > 0.0:
            d1 = flat[:, :, 1:] - flat[:, :, :-1]
            loss = loss + lambda_1 * (d1 ** 2).mean()
        if lambda_2 > 0.0:
            d2 = flat[:, :, 2:] - 2.0 * flat[:, :, 1:-1] + flat[:, :, :-2]
            loss = loss + lambda_2 * (d2 ** 2).mean()

    # Average over number of blocks so scale is independent of depth
    loss = loss / len(model.blocks)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Per-block LR helper
# ─────────────────────────────────────────────────────────────────────────────

def auto_block_lr_scales(
    model: "LUTKANStack",
    x_sample: "torch.Tensor",
    y_sample: "torch.Tensor",
    n_iters: int = 5,
) -> List[float]:
    """
    Estimate per-block LR scales by measuring gradient magnitudes.

    Runs a few forward/backward passes on x_sample, computes the mean
    absolute gradient per block, then returns scales that normalise all
    blocks to the gradient magnitude of the LAST block (deepest = smallest
    gradient = LR multiplier 1.0, earlier blocks get smaller multipliers).

    This compensates for the gradient imbalance in deep LUT-KAN stacks:
    earlier blocks receive amplified gradients through the LUT slope chain
    and need smaller effective LR to avoid thrashing.

    Parameters
    ----------
    model   : LUTKANStack (must be initialised, calibrated)
    x_sample: (N, in_dim) — a representative batch (e.g. first 256 of x_train)
    y_sample: (N, out_dim)
    n_iters : number of forward/backward passes to average over

    Returns
    -------
    List[float] of length len(model.blocks), normalised so scales[-1] = 1.0.
    Pass directly to StackTrainConfig(block_lr_scales=...).

    Example
    -------
        scales = auto_block_lr_scales(model, xt[:256], yt[:256])
        print(scales)   # e.g. [0.001, 0.01, 1.0] for a 3-block stack
        cfg = StackTrainConfig(block_lr_scales=scales)
    """
    import torch
    n_blocks = len(model.blocks)
    grad_acc = [0.0] * n_blocks

    for _ in range(n_iters):
        for blk in model.blocks:
            if blk.lut.grad is not None:
                blk.lut.grad.zero_()
        loss = ((model(x_sample) - y_sample) ** 2).mean()
        loss.backward()
        for i, blk in enumerate(model.blocks):
            if blk.lut.grad is not None:
                grad_acc[i] += float(blk.lut.grad.abs().mean())
        # Zero gradients after accumulation
        for blk in model.blocks:
            if blk.lut.grad is not None:
                blk.lut.grad.zero_()

    grad_mean = [g / n_iters for g in grad_acc]

    # Scale so that block[-1] gets scale=1.0, earlier blocks get < 1.0
    ref = grad_mean[-1] if grad_mean[-1] > 0 else 1.0
    scales = [ref / max(g, 1e-9) for g in grad_mean]

    # Cap to avoid extreme scales
    max_scale = max(scales)
    scales = [s / max_scale for s in scales]  # normalise so max=1.0

    return scales


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_lut_stack(
    model: LUTKANStack,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: StackTrainConfig,
    patience: int = 0,
) -> StackTrainResult:
    """
    patience: early-stopping patience in epochs (0 = disabled).
    Training stops if val MSE has not improved for `patience` epochs.
    Uses the same eval_every_epochs cadence as the main loop.
    """
    """
    Train a LUTKANStack with adaptive inter-layer normalisation.

    Arrays can be 1D (treated as (N, 1)) or 2D (N, dim).  The model's
    dims[0] and dims[-1] are used to validate shapes.

    Training protocol
    -----------------
    1. Initialise LUT tables: Chebyshev polynomial basis (default) or
       Gaussian noise (legacy). Chebyshev ensures all K segments receive
       gradient from step 0 — critical for multi-layer networks.
    2. Calibrate norms from x_train — sets shift/log_scale so activations at
       step 0 spread uniformly across all K segments.
    The model is expected to already be constructed with smooth_norms=True
    (the default in LUTKANStack) which uses tanh squash instead of hard clamp.
    This is critical for gradient flow in deep stacks.

    3. If cfg.freeze_norms=True (default):
         - Norm parameters are set to requires_grad=False.
         - Optimizer only contains LUT parameters.
         - Gradient still flows THROUGH norms to earlier LUT blocks.
         - If cfg.ema_alpha > 0: norms are updated from batch statistics via
           EMA every mini-batch (continuous tracking of activation drift).
         - If cfg.recalibrate_every_epochs > 0: full-dataset recalibration
           every N epochs (supplements EMA on large distribution shifts).
       If cfg.freeze_norms=False (legacy):
         - Norms and LUTs are both in Adam; gradient-based coverage losses
           (lambda_cov) are unreliable — see H8b findings.
    4. Mini-batch gradient descent, MSE + regularisation.
    5. Val-based best-model snapshot (LUTs + norms).

    Returns StackTrainResult with best and final model states.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    train_lut_stack._no_imp_ctr = 0  # reset early-stopping counter

    def _to2d(a: np.ndarray, width: int) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            if width == 1:
                return a.reshape(-1, 1)
            raise ValueError(f"1D input but model width={width}")
        return a

    in_dim  = model.dims[0]
    out_dim = model.dims[-1]

    xt = torch.from_numpy(_to2d(x_train, in_dim))
    yt = torch.from_numpy(_to2d(y_train, out_dim))
    xv = torch.from_numpy(_to2d(x_val,   in_dim))
    yv = torch.from_numpy(_to2d(y_val,   out_dim))
    xe = torch.from_numpy(_to2d(x_test,  in_dim))
    ye = torch.from_numpy(_to2d(y_test,  out_dim))
    N = xt.shape[0]

    # 1. Initialisation.
    # cheby_init (default): Chebyshev polynomial basis ensures all K LUT
    # segments receive gradient from step 0. Required for multi-layer stacks.
    # Gaussian noise only (cheby_init=False): legacy, activates only 4/16
    # segments with default std=0.05, causing constant-predictor failure.
    if cfg.cheby_init:
        model.cheby_init(scale=cfg.cheby_scale, noise=cfg.cheby_noise)
    else:
        model.add_init_noise(cfg.init_noise_std)

    # 2. Calibration (after noise, so calibration sees step-0 distribution)
    if cfg.calibrate_at_start and len(model.norms) > 0:
        model.calibrate(xt, plo=model.norm_plo, phi=model.norm_phi)

    # 3. Optimizer.
    #
    # freeze_norms=True (default): norm params are excluded from the optimizer
    # and their requires_grad is set to False.  Gradient still flows THROUGH
    # the norm operations to earlier LUT parameters — only the update step is
    # skipped.  This prevents the well-documented norm-collapse failure mode
    # where Adam shrinks scale until all activations clamp to the domain
    # boundary, permanently killing coverage and gradient flow through the norm.
    #
    # freeze_norms=False (legacy): norms are included in Adam at a higher LR.
    # Use only if you need end-to-end differentiability through the norm or are
    # running ablation experiments; expect norm collapse after ~50 epochs.
    # Build per-block LR list.
    # block_lr_scales: explicit list → use directly.
    # None → uniform (all blocks get cfg.lr).
    n_blocks = len(model.blocks)
    if cfg.block_lr_scales is not None:
        if len(cfg.block_lr_scales) != n_blocks:
            raise ValueError(
                f"block_lr_scales has {len(cfg.block_lr_scales)} entries "
                f"but model has {n_blocks} blocks"
            )
        block_lrs = [cfg.lr * s for s in cfg.block_lr_scales]
    else:
        block_lrs = [cfg.lr] * n_blocks

    if cfg.freeze_norms and len(model.norms) > 0:
        for nm in model.norms:
            nm.shift.requires_grad_(False)
            nm.log_scale.requires_grad_(False)
        param_groups = [
            {"params": list(model.blocks[i].parameters()), "lr": block_lrs[i]}
            for i in range(n_blocks)
        ]
        optim = torch.optim.Adam(param_groups)
    else:
        norm_params = [p for nm in model.norms for p in nm.parameters()]
        param_groups = (
            [
                {"params": list(model.blocks[i].parameters()), "lr": block_lrs[i]}
                for i in range(n_blocks)
            ] + [
                {"params": norm_params, "lr": cfg.lr * cfg.norm_lr_scale},
            ]
        )
        optim = torch.optim.Adam(param_groups)

    # Initial val MSE
    with torch.no_grad():
        mse_val_init = ((model(xv) - yv) ** 2).mean().item()

    best_val   = mse_val_init
    best_luts  = model.snapshot_luts()
    best_norms = model.snapshot_norms()
    best_epoch = 0
    n_updates  = 0

    trace: Dict = {
        "epoch":     [0],
        "mse_train": [None],
        "mse_val":   [mse_val_init],
    }
    if cfg.track_coverage:
        trace["coverage_uniformity"] = [None]

    for ep in range(cfg.epochs):
        # Periodic re-calibration.
        # With freeze_norms=True: recalibration is cheap (no optimizer state),
        # tracks activation drift as LUT values change.
        # With freeze_norms=False: recalibration fights the optimizer — use
        # only for diagnostic purposes, not as a fix for norm collapse.
        if (
            cfg.recalibrate_every_epochs > 0
            and len(model.norms) > 0
            and ep > 0
            and ep % cfg.recalibrate_every_epochs == 0
        ):
            with torch.no_grad():
                model.calibrate(xt, plo=model.norm_plo, phi=model.norm_phi)
            # When norms are trainable (freeze_norms=False), also reset Adam
            # state for norm params — otherwise stale momentum immediately
            # undoes the calibration in the very next optimizer step.
            if not cfg.freeze_norms:
                norm_param_set = {p for nm in model.norms for p in nm.parameters()}
                for group in optim.param_groups:
                    for p in group["params"]:
                        if p in norm_param_set and p in optim.state:
                            optim.state[p] = {}

        # Mini-batch pass
        perm = torch.randperm(N)
        for s in range(0, N, cfg.batch_size):
            idx = perm[s : s + cfg.batch_size]
            xb, yb = xt[idx], yt[idx]

            # EMA norm tracking: update shift/log_scale from batch stats
            # before the gradient step, so each batch sees accurate domain.
            if cfg.freeze_norms and cfg.ema_alpha > 0.0 and len(model.norms) > 0:
                with torch.no_grad():
                    z_fwd = xb.float()
                    for i, blk in enumerate(model.blocks[:-1]):
                        z_fwd = blk(z_fwd)
                        model.norms[i].ema_update(
                            z_fwd,
                            alpha=cfg.ema_alpha,
                            plo=model.norm_plo,
                            phi=model.norm_phi,
                        )
                        z_fwd = model.norms[i](z_fwd)

            optim.zero_grad()
            y_pred    = model(xb)
            data_loss = ((y_pred - yb) ** 2).mean()
            reg       = _stack_reg(model, cfg.lambda_1, cfg.lambda_2)
            # Coverage loss: only active when freeze_norms=False.
            # Uses norm_coverage_loss (pre-clamp) — gradient is non-zero even
            # after collapse, correctly pushing scale up to fill the LUT domain.
            cov_loss  = torch.zeros(())
            if cfg.lambda_cov > 0.0 and not cfg.freeze_norms and len(model.norms) > 0:
                z_fwd = xb.float()
                for i, blk in enumerate(model.blocks[:-1]):
                    z_fwd = blk(z_fwd)
                    cov_loss = cov_loss + norm_coverage_loss(z_fwd, model.norms[i], model.K)
                    z_fwd = model.norms[i](z_fwd)   # pass normalised value to next block
                cov_loss = cov_loss / max(1, len(model.norms))
            (data_loss + reg + cfg.lambda_cov * cov_loss).backward()
            optim.step()
            n_updates += 1

        # Evaluation
        if (ep + 1) % cfg.eval_every_epochs == 0 or (ep + 1) == cfg.epochs:
            with torch.no_grad():
                mse_t = ((model(xt) - yt) ** 2).mean().item()
                mse_v = ((model(xv) - yv) ** 2).mean().item()
            trace["epoch"].append(ep + 1)
            trace["mse_train"].append(mse_t)
            trace["mse_val"].append(mse_v)

            # Coverage tracking (optional)
            if cfg.track_coverage and len(model.norms) > 0:
                with torch.no_grad():
                    # Measure uniformity of the first inter-layer norm
                    z0 = model.blocks[0](xv)
                    cov = model.norms[0].coverage_stats(z0, model.K)
                trace["coverage_uniformity"].append(
                    round(cov["active_segment_fraction"], 4)
                )

            if mse_v < best_val:
                best_val      = mse_v
                best_luts     = model.snapshot_luts()
                best_norms    = model.snapshot_norms()
                best_epoch    = ep + 1
                epochs_no_imp = 0
            else:
                epochs_no_imp = getattr(train_lut_stack, "_no_imp_ctr", 0)

            # Early stopping
            if patience > 0:
                if not hasattr(train_lut_stack, "_no_imp_ctr"):
                    train_lut_stack._no_imp_ctr = 0
                if mse_v >= best_val + 1e-9:
                    train_lut_stack._no_imp_ctr += cfg.eval_every_epochs
                else:
                    train_lut_stack._no_imp_ctr = 0
                if train_lut_stack._no_imp_ctr >= patience:
                    break

    # Final-epoch snapshot
    final_luts  = model.snapshot_luts()
    final_norms = model.snapshot_norms()

    # Restore requires_grad on norm params so the returned model can be
    # fine-tuned or inspected without surprising frozen state.
    if cfg.freeze_norms:
        for nm in model.norms:
            nm.shift.requires_grad_(True)
            nm.log_scale.requires_grad_(True)

    # Evaluate best model: load snapshot, score, then keep as final state.
    # After this function returns, the model is in its best-epoch state.
    model.load_snapshot(best_luts, best_norms)
    with torch.no_grad():
        mse_train_best = ((model(xt) - yt) ** 2).mean().item()
        mse_val_best   = ((model(xv) - yv) ** 2).mean().item()
        mse_test_best  = ((model(xe) - ye) ** 2).mean().item()

    # Score the final-epoch weights without touching the model state.
    # We reconstruct a temporary snapshot load to get the score only.
    import copy
    tmp = copy.deepcopy(model)
    tmp.load_snapshot(final_luts, final_norms)
    with torch.no_grad():
        mse_val_final = ((tmp(xv) - yv) ** 2).mean().item()
    del tmp

    return StackTrainResult(
        best_luts=best_luts,
        best_norms=best_norms,
        final_luts=final_luts,
        final_norms=final_norms,
        mse_train_at_best=float(mse_train_best),
        mse_val_at_best=float(mse_val_best),
        mse_test_at_best=float(mse_test_best),
        mse_val_final=float(mse_val_final),
        best_epoch=int(best_epoch),
        trace=trace,
        n_updates=n_updates,
    )
