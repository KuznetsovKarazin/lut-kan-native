"""
N-layer LUT-KAN stack with adaptive inter-layer normalization.

Architecture: [d0 -> d1 -> d2 -> ... -> dN]

Each transition dᵢ -> dᵢ₊₁ consists of:
  1. LUTBlock: one LUT edge per (src, dst) pair, vectorised as (dᵢ, dᵢ₊₁, K, L).
               z = sum_i lut[i, j](xᵢ)   for each output j.
  2. LUTInterLayerNorm (between all consecutive LUTBlocks, not after the last).
               a = clamp((z - shift) / scale, x_min, x_max - eps)

The key design decision is LUTInterLayerNorm: it replaces the hardcoded
tanh squash of LUTKAN2Layer with a *learnable* per-channel affine that is
*calibrated* from data so the activation distribution covers [x_min, x_max]
as uniformly as possible.  This directly fixes the dead-cell problem that
prevents naive multi-level stacking.

Why not just tanh?
  tanh(z) squashes unbounded z into (-1, 1) but the resulting distribution
  is Gaussian-like, heavily concentrated near zero.  Central LUT cells are
  overloaded; the boundary cells (k=0 and k=K-1) receive almost no gradient.
  The norm below instead linearly stretches the p_lo..p_hi percentile range
  of the empirical distribution to exactly fill [x_min, x_max), so all K
  segments receive roughly equal expected traffic.

Why learnable (shift, log_scale) rather than fixed percentile clips?
  Fixed clips must be recomputed after every weight update; learnable
  parameters can co-adapt with the LUTs during backprop.  The calibration
  call sets a good starting point; the optimizer is then free to refine.

Why log_scale and not scale directly?
  scale = exp(log_scale) is always positive and has a smooth landscape — the
  optimizer can safely push log_scale toward -∞ without numeric catastrophe.

Calibration protocol (call once before training, optionally re-call
periodically if activations drift far):

    model = LUTKANStack(dims=[2, 8, 8, 1], K=16, L=32)
    model.calibrate(x_calib_tensor)        # sets all norms from data
    result = train_lut_stack(model, ...)   # optimizer refines from there

Deployment / numpy evaluation:
    stack_forward_numpy(x, model.export_luts(), model.export_norms())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .core import lut_forward_numpy


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: numpy LUT forward matching LUTBlock._edge_forward_bulk
# ─────────────────────────────────────────────────────────────────────────────

def _lut_fwd_compat(
    x: np.ndarray,
    lut: np.ndarray,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    """
    NumPy LUT forward matching *LUTBlock._edge_forward_bulk* exactly.

    Uses ``x_max - float32(1e-7)`` as the upper clip boundary, identical to
    the pytorch implementation, rather than ``np.nextafter`` which is used by
    ``lut_forward_numpy`` (the C-runtime-compatible reference in core.py).

    This function exists solely for ``stack_forward_numpy`` parity.
    For post-training evaluation against the main lut-kan C runtime use
    ``lut_forward_numpy`` from core.py which matches the C kernel exactly.
    """
    K, L = lut.shape
    x = np.asarray(x, dtype=np.float32)
    hi = np.float32(x_max) - np.float32(1e-7)
    x_clip = np.clip(x, np.float32(x_min), hi)
    seg_width = np.float32((x_max - x_min) / K)
    t = (x_clip - np.float32(x_min)) / seg_width
    k = np.clip(np.floor(t).astype(np.int32), 0, K - 1)
    u = np.clip(t - k.astype(np.float32), np.float32(0.0), np.float32(1.0))
    pos = u * np.float32(L - 1)
    r0 = np.clip(np.floor(pos).astype(np.int32), 0, L - 1)
    r1 = np.clip(r0 + 1, 0, L - 1)
    w = pos - r0.astype(np.float32)
    return (lut[k, r0] * (np.float32(1.0) - w) + lut[k, r1] * w).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive inter-layer normalization
# ─────────────────────────────────────────────────────────────────────────────

class LUTInterLayerNorm(nn.Module):
    """
    Learnable per-channel affine normalization between LUT-KAN layers.

    Maps inter-layer activations z into [x_min, x_max) so that the empirical
    distribution covers as many LUT cells as possible.

    Two squash modes (controlled by ``smooth`` parameter):

      smooth=True (DEFAULT, recommended):
        u = (z - shift) / scale
        a = tanh(u) * domain_half + center
        Maps ℝ → (x_min, x_max). Differentiable everywhere — no zero-gradient
        boundary. Required for gradient flow in deep (≥3 block) stacks.

      smooth=False (legacy):
        a = clamp((z - shift) / scale, x_min, x_max - eps)
        Has zero gradient where |u| > 1. Causes 3-orders-of-magnitude gradient
        attenuation in deep stacks (empirically confirmed, H8 experiments).

    scale = exp(log_scale) — always > 0.
    After calibrate(), shift ≈ p50(z) and scale is set so that p5..p95 of z
    maps to ≈ ±tanh⁻¹(0.9) in u-space, giving good domain coverage.

    Parameters
    ----------
    dim : int
        Number of channels (= output width of the preceding LUTBlock).
    x_min, x_max : float
        Domain of the next layer's LUT edges.  Defaults to [-1, 1].
    """

    def __init__(
        self,
        dim: int,
        x_min: float = -1.0,
        x_max: float = 1.0,
        smooth: bool = True,
    ):
        """
        Parameters
        ----------
        smooth : bool (default True — RECOMMENDED)
            If True, use ``tanh`` squash instead of hard ``clamp``.

            forward:  a = tanh(u) * domain_half + center
            where     u = (z - shift) / scale

            tanh maps ℝ → (-1, 1) and is differentiable everywhere.
            Hard clamp has zero gradient at |u| > 1, which causes the
            3-orders-of-magnitude gradient attenuation observed in deep
            stacks (confirmed in H8 experiments).

            With smooth=True, calibrate() sets scale so that the [p5,p95]
            mass of z maps to tanh⁻¹(0.9) ≈ ±1.47 in u-space, giving
            uniform coverage of the domain with smooth gradient through
            the tails.

            If False (legacy): hard clamp to [x_min, x_max - eps].
        """
        super().__init__()
        if x_min >= x_max:
            raise ValueError(f"require x_min < x_max, got ({x_min}, {x_max})")
        self.dim    = int(dim)
        self.x_min  = float(x_min)
        self.x_max  = float(x_max)
        self.smooth = bool(smooth)

        # Cached constants (smooth mode)
        self._center      = (x_max + x_min) / 2.0
        self._domain_half = (x_max - x_min) / 2.0

        # Learnable parameters — one scalar per hidden channel.
        # Init = identity: shift=0, scale=exp(0)=1.
        self.shift     = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.log_scale = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

        # Calibration flag (informational only; forward works without it).
        self.register_buffer("_calibrated", torch.tensor(False))

    # ---- property --------------------------------------------------------

    @property
    def scale(self) -> torch.Tensor:
        """Positive scale, always > 0."""
        return self.log_scale.exp().clamp(min=1e-6)

    # ---- calibration -----------------------------------------------------

    @torch.no_grad()
    def calibrate(
        self,
        z: torch.Tensor,
        plo: float = 5.0,
        phi: float = 95.0,
    ) -> dict:
        """
        One-shot initialisation from data percentiles.

        Sets shift and log_scale so that the [plo, phi] percentile range of
        the empirical distribution of z maps linearly to [x_min, x_max].

        Parameters
        ----------
        z : torch.Tensor, shape (N, dim)
            Activations produced by the preceding LUTBlock on calibration data.
        plo, phi : float
            Percentiles (0-100) that will be mapped to x_min, x_max.
            Default 5-95 means 90% of the distribution fills the domain.

        Returns
        -------
        dict with calibration stats for logging / inspection.
        """
        if z.dim() != 2 or z.shape[1] != self.dim:
            raise ValueError(
                f"Expected z of shape (N, {self.dim}), got {tuple(z.shape)}"
            )
        lo = torch.quantile(z.float(), plo / 100.0, dim=0)   # (dim,)
        hi = torch.quantile(z.float(), phi / 100.0, dim=0)   # (dim,)

        # Center of the percentile range → shift
        center = (lo + hi) / 2.0

        # Half-width of the percentile range → maps to half the domain
        half_data   = (hi - lo) / 2.0
        half_domain = (self.x_max - self.x_min) / 2.0

        # Protect against constant channels
        half_data = half_data.clamp(min=1e-4)

        if self.smooth:
            # With tanh: want p95 of u = atanh(0.9) ≈ 1.472
            # so that tanh(p95_u) ≈ 0.9 → covers 90% of domain half
            import math
            target_u = math.atanh(0.9)   # ≈ 1.472
            new_scale = half_data / target_u
        else:
            new_scale = half_data / half_domain
        self.shift.copy_(center)
        self.log_scale.copy_(new_scale.log())
        self._calibrated.fill_(True)

        return {
            "shift": center.cpu().numpy().copy(),
            "scale": new_scale.cpu().numpy().copy(),
            "z_p_lo": lo.cpu().numpy().copy(),
            "z_p_hi": hi.cpu().numpy().copy(),
            "z_mean": z.mean(dim=0).cpu().numpy().copy(),
            "z_std":  z.std(dim=0).cpu().numpy().copy(),
        }

    # ---- EMA tracking ---------------------------------------------------

    @torch.no_grad()
    def ema_update(
        self,
        z: torch.Tensor,
        alpha: float = 0.05,
        plo: float = 5.0,
        phi: float = 95.0,
    ) -> None:
        """
        Update shift and log_scale toward current batch statistics via EMA.

        Called every training batch (or every N batches) when freeze_norms=True.
        Tracks activation drift as LUT values change without any gradient.

        Why EMA and not a coverage loss?
        ─────────────────────────────────
        All gradient-based coverage approaches fail when norm has collapsed:
          - soft histogram: sigmoid gates saturate to 0 outside domain.
          - out-of-domain penalty: mixed-sign gradients due to shift/scale
            coupling when activations are asymmetrically distributed.
          - quantile matching: torch.quantile gradient is sparse (2 samples
            per channel) and unreliable at extreme u values.

        EMA operates directly on raw z — always non-zero signal, no saturation,
        converges to the same target as calibrate() but tracks continuously.

        Parameters
        ----------
        z     : (N, dim) raw output of the preceding LUTBlock (pre-norm).
        alpha : EMA smoothing factor [0.01, 0.1]. Default 0.05.
        plo, phi : percentiles defining the target domain coverage.
        """
        lo     = torch.quantile(z.float(), plo / 100.0, dim=0)
        hi_q   = torch.quantile(z.float(), phi / 100.0, dim=0)
        center = (lo + hi_q) / 2.0
        half_data = ((hi_q - lo) / 2.0).clamp(min=1e-4)
        half_domain = (self.x_max - self.x_min) / 2.0
        target_log_scale = (half_data / half_domain).log()
        # EMA: new = (1-alpha)*old + alpha*target
        self.shift.lerp_(center, alpha)
        self.log_scale.lerp_(target_log_scale, alpha)

    # ---- forward ---------------------------------------------------------

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Normalize z.

        smooth=True (default):
            u = (z - shift) / scale
            a = tanh(u) * domain_half + center
            Output ∈ (x_min, x_max), differentiable everywhere.

        smooth=False (legacy, hard clamp):
            a = clamp((z - shift) / scale, x_min, x_max - eps)
            Zero gradient at |u| > 1.

        Args:
            z : (N, dim)
        Returns:
            (N, dim)
        """
        u = (z - self.shift) / self.scale
        if self.smooth:
            return torch.tanh(u) * self._domain_half + self._center
        else:
            hi = self.x_max - 1e-7
            return torch.clamp(u, self.x_min, hi)

    # ---- numpy export for deployment ------------------------------------

    def export_numpy(self) -> dict:
        """Export parameters as numpy arrays for deployment / evaluation."""
        return {
            "shift":     self.shift.detach().cpu().numpy().copy(),
            "scale":     self.scale.detach().cpu().numpy().copy(),
            "x_min":     self.x_min,
            "x_max":     self.x_max,
            "smooth":    self.smooth,
        }

    # ---- coverage diagnostics -------------------------------------------

    @torch.no_grad()
    def coverage_stats(self, z: torch.Tensor, K: int) -> dict:
        """
        Measure how uniformly z (pre-normalisation) would cover K segments.

        Returns per-channel and aggregate stats.
        """
        a = self.forward(z).cpu().numpy()          # (N, dim)
        N = a.shape[0]
        seg_counts = np.zeros((self.dim, K), dtype=np.float64)
        for d in range(self.dim):
            t = (np.clip(a[:, d], self.x_min, self.x_max - 1e-9) - self.x_min) \
                / (self.x_max - self.x_min) * K
            k = np.clip(t.astype(int), 0, K - 1)
            for ki in range(K):
                seg_counts[d, ki] = (k == ki).sum()

        # uniformity: how flat is the segment distribution?
        # perfect = N/K per segment per channel
        expected = N / K
        uniformity = 1.0 - float(np.abs(seg_counts - expected).mean() / (expected + 1e-9))
        active_frac = float((seg_counts > 0).mean())
        return {
            "uniformity": max(0.0, uniformity),
            "active_segment_fraction": active_frac,
            "seg_counts_mean_per_channel": seg_counts.mean(axis=1).tolist(),
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, x_min={self.x_min}, x_max={self.x_max}, "
            f"smooth={self.smooth}, calibrated={bool(self._calibrated.item())}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable coverage entropy loss (for trainable-norm mode)
# ─────────────────────────────────────────────────────────────────────────────

def _soft_histogram_entropy(
    v: torch.Tensor,
    K: int,
    x_min: float,
    x_max: float,
    temperature: float,
) -> torch.Tensor:
    """
    Compute soft-histogram entropy of v (N, dim) over K segments in [x_min, x_max].

    H = -sum_k p_k log(p_k)   (higher = more uniform = better coverage)
    Returns -H (scalar, minimise to maximise coverage).

    Segments cover [x_min, x_max] uniformly.  The soft assignment uses
    sigmoid gates — differentiable everywhere including outside [x_min, x_max].
    """
    eps = 1e-8
    edges  = torch.linspace(x_min, x_max, K + 1, dtype=v.dtype, device=v.device)
    lo = edges[:-1].view(1, 1, K)   # (1, 1, K)
    hi = edges[1:].view(1, 1, K)    # (1, 1, K)
    v_exp = v.unsqueeze(-1)          # (N, dim, 1)

    left_gate  = torch.sigmoid((v_exp - lo) * temperature)
    right_gate = torch.sigmoid((hi - v_exp) * temperature)
    soft       = left_gate * right_gate                          # (N, dim, K)
    p          = soft / (soft.sum(dim=-1, keepdim=True) + eps)  # (N, dim, K)
    p_marg     = p.mean(dim=0)                                   # (dim, K)
    H          = -(p_marg * (p_marg + eps).log()).sum(dim=-1)    # (dim,)
    return -H.mean()


def norm_coverage_loss(
    z: torch.Tensor,
    norm: "LUTInterLayerNorm",
    K: int,
    temperature: float = 50.0,
) -> torch.Tensor:
    """
    Differentiable coverage loss for a LUTInterLayerNorm.

    **This is the correct API for H8b experiments.**

    Operates on the PRE-CLAMP normalised value ``u = (z - shift) / scale``
    rather than the post-clamp output ``a = clamp(u, x_min, x_max)``.

    Why pre-clamp matters
    ---------------------
    After norm collapse, ``scale`` shrinks until nearly all activations land
    outside ``[x_min, x_max]``, so ``a = ±1`` for ~90% of samples.  The
    gradient of any loss through the hard ``clamp`` is zero for clamped
    values — the loss cannot pull ``shift`` or ``scale`` back toward a
    sensible state.

    ``u = (z - shift) / scale`` is always differentiable w.r.t. ``shift``
    and ``log_scale`` regardless of clamping:
        ∂u/∂shift     = -1/scale       (non-zero everywhere)
        ∂u/∂log_scale = -u             (non-zero unless u=0)

    The soft histogram on ``u`` uses the target domain ``[x_min, x_max]`` as
    the histogram range (same as the LUT domain).  When ``scale`` is too
    small, ``u`` is spread over a range much wider than ``[x_min, x_max]``
    and the entropy across the K target segments is low.  Minimising
    ``-H(u)`` pushes ``scale`` to grow so that ``u`` fills the domain
    uniformly.

    Usage
    -----
        # In training loop (freeze_norms=False required):
        loss = mse_loss + lambda_cov * norm_coverage_loss(z_block, norm, K)

    Args:
        z    : (N, dim)  raw output of the preceding LUTBlock (pre-norm).
        norm : LUTInterLayerNorm instance whose shift/log_scale to optimise.
        K    : number of LUT segments in the NEXT block.
        temperature : sigmoid gate sharpness.  50 is reasonable for K=16,
                      domain [-1,1]; scale ∝ 1/(segment_width*2).

    Returns:
        Scalar, negative entropy (minimise → maximise uniformity).
    """
    u = (z - norm.shift) / norm.scale   # (N, dim), pre-clamp, differentiable
    return _soft_histogram_entropy(u, K, norm.x_min, norm.x_max, temperature)


def coverage_entropy_loss(
    a: torch.Tensor,
    K: int,
    x_min: float = -1.0,
    x_max: float = 1.0,
    temperature: float = 50.0,
) -> torch.Tensor:
    """
    Soft-histogram entropy loss on POST-CLAMP activations ``a``.

    .. deprecated::
        Use :func:`norm_coverage_loss` instead.  This function operates on
        the clamped output of ``LUTInterLayerNorm.forward()`` and produces
        **zero or perverse gradients** w.r.t. ``shift`` / ``log_scale``
        whenever the norm has collapsed (scale too small → all activations
        clamped to ±1 → gradient through clamp = 0).

        Kept for backward compatibility and ablation experiments only.

    Args:
        a          : (N, dim) post-clamp activations.
        K          : number of LUT segments.
        x_min/max  : domain.
        temperature: sigmoid gate sharpness.

    Returns:
        Scalar negative entropy (minimise → maximise uniformity).
    """
    return _soft_histogram_entropy(a, K, x_min, x_max, temperature)


# ─────────────────────────────────────────────────────────────────────────────
# Single LUT layer (one block in the stack)
# ─────────────────────────────────────────────────────────────────────────────

class LUTBlock(nn.Module):
    """
    Single LUT layer: [in_dim -> out_dim] via (in_dim × out_dim) LUT edges.

    For each output j:
        z_j = sum_i lut[i, j](x_i)

    All edges share the same K, L, x_min, x_max.  Each edge has its own
    (K, L) table stored in a single fused parameter lut of shape
    (in_dim, out_dim, K, L).

    This is the same maths as LUTKAN2Layer._edge_forward_bulk; factored out
    so the stack can compose arbitrary numbers of them.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        K: int,
        L: int,
        x_min: float = -1.0,
        x_max: float = 1.0,
    ):
        super().__init__()
        if K < 1 or L < 2:
            raise ValueError("Require K >= 1 and L >= 2")
        self.in_dim  = int(in_dim)
        self.out_dim = int(out_dim)
        self.K = int(K)
        self.L = int(L)
        self.x_min = float(x_min)
        self.x_max = float(x_max)

        self.lut = nn.Parameter(
            torch.zeros(in_dim, out_dim, K, L, dtype=torch.float32)
        )
        self.register_buffer("_x_min_t",      torch.tensor(x_min, dtype=torch.float32))
        self.register_buffer("_x_max_t",      torch.tensor(x_max, dtype=torch.float32))
        self.register_buffer("_seg_width_t",  torch.tensor((x_max - x_min) / K,
                                                            dtype=torch.float32))

    # ---- initialisation --------------------------------------------------

    @torch.no_grad()
    def init_from_array(self, luts: np.ndarray) -> None:
        """luts: (in_dim, out_dim, K, L) float32 array."""
        expected = (self.in_dim, self.out_dim, self.K, self.L)
        if luts.shape != expected:
            raise ValueError(f"Expected shape {expected}, got {luts.shape}")
        self.lut.copy_(torch.from_numpy(luts.astype(np.float32)))

    @torch.no_grad()
    def add_init_noise(self, std: float) -> None:
        if std > 0:
            self.lut.add_(torch.randn_like(self.lut) * std)

    @torch.no_grad()
    def cheby_init(self, scale: float = 1.5, noise: float = 0.05) -> None:
        """
        Initialise LUT cells with Chebyshev polynomial basis functions.

        Why this is needed
        ------------------
        With zero + small-noise init (std=0.05), each LUT edge outputs ~0 for
        every input. In a two-layer network this means the second block always
        receives near-constant hidden states — effectively making the model a
        constant predictor (MSE ≈ Var(y)).

        This sets lut[si, di, k, r] = T_{di}(x_val) * scale / in_dim where
        x_val is the grid point at cell (k, r) and degree = di % out_dim.
        This ensures:
          1. Non-zero d(lut_output)/dx from step 0 → gradient flows everywhere.
          2. Diverse hidden representations per output channel.
          3. Hidden activations span the full downstream LUT domain immediately.

        Parameters
        ----------
        scale : float  — overall Chebyshev amplitude / n_src.
                         scale=1.5 gives z_std≈0.9, covering all K segments.
        noise : float  — additional Gaussian noise to break per-channel ties.
        """
        seg_w = (self.x_max - self.x_min) / self.K
        for si in range(self.in_dim):
            for di in range(self.out_dim):
                degree = di % max(1, self.out_dim)
                for k in range(self.K):
                    for r in range(self.L):
                        x_val = self.x_min + (k + r / (self.L - 1)) * seg_w
                        x_n   = 2.0 * (x_val - self.x_min) / (self.x_max - self.x_min) - 1.0
                        x_n   = float(np.clip(x_n, -1 + 1e-6, 1 - 1e-6))
                        t_k   = float(np.cos(degree * np.arccos(x_n)))
                        self.lut[si, di, k, r] = t_k * scale / self.in_dim
        if noise > 0:
            self.lut.add_(torch.randn_like(self.lut) * noise)

    # ---- forward ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_dim). Returns (N, out_dim).
        """
        return self._edge_forward_bulk(
            x, self.lut, self._x_min_t, self._x_max_t, self._seg_width_t
        )

    def _edge_forward_bulk(
        self,
        x: torch.Tensor,           # (N, n_src)
        lut_block: torch.Tensor,   # (n_src, n_dst, K, L)
        x_min_t: torch.Tensor,
        x_max_t: torch.Tensor,
        seg_width_t: torch.Tensor,
    ) -> torch.Tensor:
        """Fused vectorised forward, identical in maths to LUTKAN2Layer."""
        N = x.shape[0]
        n_src, n_dst, K, L = lut_block.shape
        hi = x_max_t - torch.tensor(1e-7, dtype=x.dtype, device=x.device)
        x_clip = torch.clamp(x, x_min_t, hi)
        t  = (x_clip - x_min_t) / seg_width_t
        k  = torch.clamp(torch.floor(t).long(), 0, K - 1)
        u  = torch.clamp(t - k.to(t.dtype), 0.0, 1.0)
        pos = u * (L - 1)
        r0 = torch.clamp(torch.floor(pos).long(), 0, L - 1)
        r1 = torch.clamp(r0 + 1, 0, L - 1)
        w  = pos - r0.to(pos.dtype)
        src_idx = torch.arange(n_src, device=x.device).view(1, n_src).expand(N, n_src)
        v0 = lut_block[src_idx, :, k, r0]   # (N, n_src, n_dst)
        v1 = lut_block[src_idx, :, k, r1]
        w_ = w.unsqueeze(-1)
        return (v0 * (1.0 - w_) + v1 * w_).sum(dim=1)   # (N, n_dst)

    # ---- utility ----------------------------------------------------------

    def extra_repr(self) -> str:
        n = self.in_dim * self.out_dim * self.K * self.L
        return (
            f"{self.in_dim}->{self.out_dim}, K={self.K}, L={self.L}, "
            f"x_min={self.x_min}, x_max={self.x_max}, n_params={n}"
        )

    def n_lut_params(self) -> int:
        return self.in_dim * self.out_dim * self.K * self.L

    def memory_bytes_uint8(self) -> int:
        """Bytes if all LUTs were uint8-quantized (K*L + 4*K per edge)."""
        n_edges = self.in_dim * self.out_dim
        return n_edges * (self.K * self.L + 4 * self.K)


# ─────────────────────────────────────────────────────────────────────────────
# N-layer stack
# ─────────────────────────────────────────────────────────────────────────────

class LUTKANStack(nn.Module):
    """
    Arbitrary-depth LUT-KAN stack with adaptive inter-layer normalisation.

    Topology is given by `dims`, e.g. dims=[2, 8, 8, 1] gives:
        LUTBlock(2 -> 8) → LUTInterLayerNorm(8) → LUTBlock(8 -> 8)
        → LUTInterLayerNorm(8) → LUTBlock(8 -> 1)

    There are len(dims)-1 LUTBlocks and len(dims)-2 norms (no norm after the
    last block — its output is the final prediction).

    Parameters
    ----------
    dims : list[int]
        Width of each layer, including input and output.  Minimum length 2.
    K, L : int
        Segments and entries per LUT edge.  Shared across all blocks.
    x_min, x_max : float
        Input domain for the first block.
    norm_plo, norm_phi : float
        Percentiles used during calibration (default 5-95: 90% of the
        activation distribution fills the inter-layer domain).
    inter_x_min, inter_x_max : float
        Domain of inter-layer LUT edges (after each norm).  Defaults to
        same as x_min/x_max so all edges are bit-compatible.
    """

    def __init__(
        self,
        dims: List[int],
        K: int,
        L: int,
        x_min: float = -1.0,
        x_max: float = 1.0,
        norm_plo: float = 5.0,
        norm_phi: float = 95.0,
        inter_x_min: float = -1.0,
        inter_x_max: float = 1.0,
        smooth_norms: bool = True,
    ):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must have at least 2 entries (in, out)")
        self.dims   = list(dims)
        self.K      = int(K)
        self.L      = int(L)
        self.x_min  = float(x_min)
        self.x_max  = float(x_max)
        self.norm_plo = float(norm_plo)
        self.norm_phi = float(norm_phi)
        self.inter_x_min = float(inter_x_min)
        self.inter_x_max = float(inter_x_max)
        self.smooth_norms = bool(smooth_norms)

        # Build blocks and norms.
        # Block i maps dims[i] -> dims[i+1].
        # Norm i normalises the output of block i before it enters block i+1.
        n_blocks = len(dims) - 1
        blocks: List[nn.Module] = []
        norms:  List[nn.Module] = []

        for i in range(n_blocks):
            in_d  = dims[i]
            out_d = dims[i + 1]
            # First block uses the user-supplied input domain.
            # Subsequent blocks take normalised activations in [inter_x_min, inter_x_max].
            blk_x_min = x_min  if i == 0 else inter_x_min
            blk_x_max = x_max  if i == 0 else inter_x_max
            blocks.append(LUTBlock(in_d, out_d, K, L, blk_x_min, blk_x_max))
            # Norm after every block except the last.
            if i < n_blocks - 1:
                norms.append(LUTInterLayerNorm(out_d, inter_x_min, inter_x_max,
                                               smooth=smooth_norms))

        self.blocks = nn.ModuleList(blocks)
        self.norms  = nn.ModuleList(norms)

    # ---- initialisation --------------------------------------------------

    @torch.no_grad()
    def add_init_noise(self, std: float) -> None:
        """Add Gaussian noise to all LUT blocks (for symmetry-breaking at init)."""
        for blk in self.blocks:
            blk.add_init_noise(std)

    @torch.no_grad()
    def cheby_init(self, scale: float = 1.5, noise: float = 0.05) -> None:
        """
        Apply Chebyshev polynomial init to ALL LUT blocks in the stack.

        All blocks benefit: block 0 spans the input domain richly, and
        subsequent blocks span the inter-layer domain ([-1,1] after norm)
        with diverse polynomial shapes. Call BEFORE calibrate().

        Parameters
        ----------
        scale : float  — Chebyshev amplitude. 1.5 recommended.
        noise : float  — per-cell Gaussian noise for diversity. 0.05 recommended.
        """
        for blk in self.blocks:
            blk.cheby_init(scale=scale, noise=noise)

    # ---- calibration -----------------------------------------------------

    @torch.no_grad()
    def calibrate(
        self,
        x_calib: torch.Tensor,
        plo: Optional[float] = None,
        phi: Optional[float] = None,
        verbose: bool = False,
    ) -> List[dict]:
        """
        Propagate calibration data through the stack and set each
        LUTInterLayerNorm from the empirical distribution of its input.

        Must be called AFTER initialising LUT values (e.g. adding init noise
        or loading polynomial initialisations) so the calibration reflects
        the actual activation distribution at step 0 of training.

        Parameters
        ----------
        x_calib : Tensor, (N, dims[0])
            Calibration examples (typically x_train or a representative subset).
        plo, phi : float or None
            Percentiles; defaults to self.norm_plo / self.norm_phi.
        verbose : bool
            If True, print per-norm stats.

        Returns
        -------
        List of calibration stat dicts, one per norm.
        """
        plo = plo if plo is not None else self.norm_plo
        phi = phi if phi is not None else self.norm_phi
        if x_calib.dim() == 1:
            x_calib = x_calib.unsqueeze(-1)

        stats_list = []
        z = x_calib.float()
        for i, blk in enumerate(self.blocks[:-1]):    # all except last
            z = blk(z)                                 # (N, dims[i+1])
            norm = self.norms[i]
            stats = norm.calibrate(z, plo=plo, phi=phi)
            stats_list.append(stats)
            z = norm(z)                                # feed calibrated value forward
            if verbose:
                print(
                    f"  norm[{i}]  scale min/max: "
                    f"{stats['scale'].min():.3f} / {stats['scale'].max():.3f}  "
                    f"shift min/max: "
                    f"{stats['shift'].min():.3f} / {stats['shift'].max():.3f}"
                )
        return stats_list

    # ---- forward ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, dims[0]) or (N,) for dims[0]==1.
        Returns: (N, dims[-1]).
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        z = x.float()
        for i, blk in enumerate(self.blocks):
            z = blk(z)
            if i < len(self.norms):
                z = self.norms[i](z)
        return z

    # ---- export for deployment ------------------------------------------

    def export_luts(self) -> List[np.ndarray]:
        """Return list of LUT arrays, one per block, shape (in_d, out_d, K, L)."""
        return [blk.lut.detach().cpu().numpy().copy() for blk in self.blocks]

    def export_norms(self) -> List[dict]:
        """Return list of norm parameter dicts (shift, scale, x_min, x_max)."""
        return [n.export_numpy() for n in self.norms]

    # ---- state snapshot for best-model tracking -------------------------

    def snapshot_luts(self) -> List[np.ndarray]:
        return self.export_luts()

    def snapshot_norms(self) -> List[dict]:
        return self.export_norms()

    def load_snapshot(
        self, luts: List[np.ndarray], norms: List[dict]
    ) -> None:
        """Restore LUT values and norm parameters from a snapshot."""
        with torch.no_grad():
            for blk, arr in zip(self.blocks, luts):
                blk.lut.copy_(torch.from_numpy(arr))
            for norm, d in zip(self.norms, norms):
                norm.shift.copy_(torch.from_numpy(d["shift"]))
                # d["scale"] is the actual scale, not log_scale
                norm.log_scale.copy_(
                    torch.from_numpy(
                        np.log(np.clip(d["scale"], 1e-6, None)).astype(np.float32)
                    )
                )

    # ---- utility ---------------------------------------------------------

    def n_lut_params(self) -> int:
        return sum(blk.n_lut_params() for blk in self.blocks)

    def n_norm_params(self) -> int:
        return sum(2 * n.dim for n in self.norms)   # shift + log_scale per channel

    def total_params(self) -> int:
        return self.n_lut_params() + self.n_norm_params()

    def memory_bytes_uint8(self) -> int:
        return sum(blk.memory_bytes_uint8() for blk in self.blocks)

    def extra_repr(self) -> str:
        dims_str = "->".join(str(d) for d in self.dims)
        return (
            f"dims=[{dims_str}], K={self.K}, L={self.L}, "
            f"n_lut_params={self.n_lut_params()}, "
            f"n_norm_params={self.n_norm_params()}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# NumPy reference forward for deployment / post-training evaluation
# ─────────────────────────────────────────────────────────────────────────────

def stack_forward_numpy(
    x: np.ndarray,
    luts: List[np.ndarray],
    norms: Optional[List[dict]] = None,
) -> np.ndarray:
    """
    NumPy reference forward matching LUTKANStack.forward bit-for-bit.

    Parameters
    ----------
    x : (N, dims[0]) or (N,) for dims[0]==1.
    luts : list of (in_d, out_d, K, L) arrays, one per LUTBlock.
    norms : list of norm dicts (from LUTInterLayerNorm.export_numpy()),
            one per norm (= len(luts) - 1).  If None, no norms are applied
            (useful for single-layer stacks).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n_blocks = len(luts)
    if norms is None:
        norms = []

    z = x
    for i, lut_block in enumerate(luts):
        in_d, out_d, K, L = lut_block.shape
        N = z.shape[0]
        if i == 0:
            x_min, x_max = -1.0, 1.0
        else:
            x_min = norms[i - 1]["x_min"]
            x_max = norms[i - 1]["x_max"]
        z_next = np.zeros((N, out_d), dtype=np.float32)
        for src in range(in_d):
            for dst in range(out_d):
                z_next[:, dst] += _lut_fwd_compat(
                    z[:, src], lut_block[src, dst], x_min=x_min, x_max=x_max
                )
        z = z_next
        if i < len(norms):
            nd = norms[i]
            shift = nd["shift"].astype(np.float32).reshape(1, -1)
            scale = nd["scale"].astype(np.float32).reshape(1, -1)
            u = (z - shift) / scale
            if nd.get("smooth", True):
                domain_half = np.float32((nd["x_max"] - nd["x_min"]) / 2.0)
                center      = np.float32((nd["x_max"] + nd["x_min"]) / 2.0)
                z = (np.tanh(u) * domain_half + center).astype(np.float32)
            else:
                hi = np.float32(nd["x_max"]) - np.float32(1e-7)
                z = np.clip(u, np.float32(nd["x_min"]), hi)
    return z
