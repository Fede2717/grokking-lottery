"""
src/metrics.py — Advanced Metrics for Grokking × LTH Research
==============================================================

Implements all metrics categorised in the theoretical design document:

CV-impact metrics
-----------------
    compute_hessian_top_eigenvalue   — sharpness via power iteration
    compute_effective_rank           — intrinsic dimensionality of weight mats
    compute_gsnr                     — gradient signal-to-noise ratio per layer

LinkedIn / visual metrics
--------------------------
    compute_weight_norms             — global & per-layer L1/L2 norms
    compute_fourier_features         — DFT of embedding matrix at checkpoints
    compute_sparsity_from_masks      — current sparsity ratio

Scientific / rigorous metrics
------------------------------
    compute_grokking_metrics         — S_G, G_gap, weight norm at grokking
    compute_generalization_gap       — train_acc - val_acc over time

References
----------
    Nanda   et al. (2023) arXiv:2301.05217  [Fourier features]
    Frankle et al. (2020) arXiv:1912.05671  [IMP stability]
    Ghorbani et al. (2019) arXiv:1901.10159 [Hessian eigenvalues]
    Liu     et al. (2023)                   [weight decay → grokking]
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from src.model import GrokTransformer


# ===========================================================================
# Weight Norm Metrics
# ===========================================================================

def compute_weight_norms(
    model: nn.Module,
    masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    """
    Compute global and per-layer L1/L2 norms of *active* weights.

    If masks are provided, only unmasked (active) weights contribute.

    Returns
    -------
    dict with keys:
        "global_l2"       : float
        "global_l1"       : float
        "per_layer"       : dict[str → {"l2": float, "l1": float}]
    """
    global_sq   = 0.0
    global_abs  = 0.0
    per_layer: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            w = param.detach().float()

            if masks is not None and name in masks:
                w = w * masks[name].float()

            l2 = float(w.pow(2).sum().sqrt())
            l1 = float(w.abs().sum())

            per_layer[name] = {"l2": l2, "l1": l1}
            global_sq  += w.pow(2).sum().item()
            global_abs += w.abs().sum().item()

    return {
        "global_l2" : math.sqrt(global_sq),
        "global_l1" : global_abs,
        "per_layer" : per_layer,
    }


# ===========================================================================
# Sparsity
# ===========================================================================

def compute_sparsity_from_masks(
    masks: dict[str, torch.Tensor],
) -> float:
    """
    Return the fraction of weights that are pruned (set to 0).

    sparsity = 1 - (# unmasked weights) / (# total masked weights)
    """
    total   = sum(m.numel() for m in masks.values())
    active  = sum(m.sum().item() for m in masks.values())
    return 1.0 - (active / total) if total > 0 else 0.0


def compute_model_sparsity(model: nn.Module) -> float:
    """
    Empirical sparsity: fraction of parameters that are exactly zero.
    Useful for measuring effective sparsity after mask application.
    """
    total = 0
    zeros = 0
    with torch.no_grad():
        for param in model.parameters():
            total += param.numel()
            zeros += (param == 0).sum().item()
    return zeros / total if total > 0 else 0.0


# ===========================================================================
# Fourier Feature Analysis   (Light version — checkpoints only)
# ===========================================================================

def compute_fourier_features(
    embedding_weights: torch.Tensor,   # (vocab_size, d_model)
    p: int,
    top_k: int = 5,
) -> dict:
    """
    Analyse the Fourier structure of number-token embeddings.

    Grokking on modular arithmetic is mechanistically explained by the
    model learning specific frequencies in the embedding space (Nanda 2023).
    This function computes the DFT over the p number tokens and identifies
    dominant frequencies — a direct readout of whether the network has
    discovered the generalising algorithm.

    Algorithm
    ---------
    1. Extract rows 0..p-1  from the embedding matrix  → W  (p, d_model)
    2. Compute rfft over the token axis (axis=0)        → W_fft  (p//2+1, d_model)
    3. Mean power spectrum across embedding dims        → mean_power  (p//2+1,)
    4. Report: top-k dominant frequencies, concentration ratio

    Parameters
    ----------
    embedding_weights : Full embedding matrix  (vocab_size, d_model).
    p                 : Modulus (number of number tokens to consider).
    top_k             : How many dominant frequencies to report.

    Returns
    -------
    dict with:
        "power_spectrum"   : list[float]   — mean power per frequency
        "frequencies"      : list[int]     — frequency indices (0 .. p//2)
        "top_frequencies"  : list[int]     — indices of top-k dominant freqs
        "top_powers"       : list[float]   — power at top-k frequencies
        "concentration"    : float         — power-in-top1 / total power
        "entropy"          : float         — spectral entropy (lower = more structured)
    """
    W = embedding_weights[:p, :].detach().cpu().float().numpy()  # (p, d_model)

    # DFT over token axis
    W_fft   = np.fft.rfft(W, axis=0)              # (p//2+1, d_model)
    power   = (np.abs(W_fft) ** 2)                # power spectrum per (freq, dim)
    mean_pw = power.mean(axis=1)                   # average over embedding dims

    # Dominant frequencies
    order      = np.argsort(mean_pw)[::-1]
    top_freqs  = order[:top_k].tolist()
    top_powers = mean_pw[top_freqs].tolist()

    # Concentration: fraction of total power in the single dominant frequency
    total_pw      = float(mean_pw.sum())
    concentration = float(mean_pw[top_freqs[0]] / total_pw) if total_pw > 0 else 0.0

    # Spectral entropy: H = -sum(p * log(p)) over normalised spectrum
    p_norm  = mean_pw / (total_pw + 1e-12)
    entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-12)))

    return {
        "power_spectrum"  : mean_pw.tolist(),
        "frequencies"     : list(range(len(mean_pw))),
        "top_frequencies" : top_freqs,
        "top_powers"      : top_powers,
        "concentration"   : concentration,
        "entropy"         : entropy,
    }


# ===========================================================================
# Effective Rank
# ===========================================================================

def compute_effective_rank(weight_matrix: torch.Tensor) -> float:
    """
    Compute the effective rank of a 2-D weight matrix.

    Effective rank (Roy & Vetterli, 2007) = exp(H(σ)), where H is the
    Shannon entropy of the normalised singular values.  A low effective rank
    signals that the weight matrix has collapsed to a low-dimensional
    subspace — a proxy for implicit regularisation and structured learning.

    Parameters
    ----------
    weight_matrix : 2-D FloatTensor  (out_features, in_features)

    Returns
    -------
    Effective rank ∈ [1, min(out, in)]
    """
    W  = weight_matrix.detach().float()
    sv = torch.linalg.svdvals(W)
    sv = sv[sv > 1e-10]                 # discard numerical zeros
    sv = sv / sv.sum()                  # normalise to probability vector
    H  = -(sv * sv.log()).sum()         # Shannon entropy of singular values
    return float(H.exp())


def compute_all_effective_ranks(model: nn.Module) -> dict[str, float]:
    """Return effective rank for every 2-D weight matrix in the model."""
    ranks: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.dim() == 2:
            ranks[name] = compute_effective_rank(param)
    return ranks


# ===========================================================================
# Gradient Signal-to-Noise Ratio (GSNR)
# ===========================================================================

def compute_gsnr(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_batches: int = 8,
) -> dict[str, float]:
    """
    Compute per-layer Gradient Signal-to-Noise Ratio.

    GSNR(layer) = ||E[g]||² / E[||g - E[g]||²]

    where g is the gradient of that layer's parameters over a single batch.
    A high GSNR means gradients are consistent across batches (clean signal);
    a low GSNR means noisy, conflicting gradients (hard to optimise).

    Parameters
    ----------
    n_batches : Number of mini-batches used to estimate expectation.

    Returns
    -------
    dict[layer_name → GSNR]
    """
    model.eval()
    grad_accum: dict[str, list[torch.Tensor]] = {}

    loader_iter = iter(loader)
    for _ in range(n_batches):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)

        model.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                g = param.grad.detach().clone()
                grad_accum.setdefault(name, []).append(g)

    gsnr: dict[str, float] = {}
    for name, grads in grad_accum.items():
        stacked   = torch.stack(grads, dim=0)   # (n_batches, *param_shape)
        mean_g    = stacked.mean(0)
        var_g     = stacked.var(0)
        signal    = mean_g.pow(2).sum().item()
        noise     = var_g.sum().item() + 1e-12
        gsnr[name] = signal / noise

    model.train()
    return gsnr


# ===========================================================================
# Hessian Top Eigenvalue  (power iteration)
# ===========================================================================

def compute_hessian_top_eigenvalue(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_steps: int = 20,
    n_batches: int = 4,
) -> float:
    """
    Estimate the top Hessian eigenvalue via power iteration.

    Measures loss-landscape *sharpness*.  Sharp minima (high λ_max)
    correlate with poor generalisation (Keskar et al., 2017).
    Grokking is associated with a transition to flatter regions.

    This is a Hessian-vector product approach:
        Hv ≈ (g(w + εv) - g(w - εv)) / (2ε)
    using finite differences on gradients (no second-order autograd).

    Parameters
    ----------
    n_steps   : Power iteration steps (20 sufficient for top eigenvalue).
    n_batches : Batches averaged per gradient evaluation.

    Returns
    -------
    Estimated top Hessian eigenvalue (float).
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    def get_gradient(perturbed_params=None) -> list[torch.Tensor]:
        """Compute ∇L over n_batches, optionally with perturbed params."""
        model.zero_grad()
        loader_iter = iter(loader)
        total_loss  = torch.tensor(0.0, device=device)
        count = 0
        for _ in range(n_batches):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                break
            x, y = x.to(device), y.to(device)
            total_loss = total_loss + criterion(model(x), y)
            count += 1
        if count > 0:
            (total_loss / count).backward()
        return [p.grad.detach().clone() if p.grad is not None
                else torch.zeros_like(p)
                for p in params]

    # Initialise random unit vector v
    v   = [torch.randn_like(p) for p in params]
    v_n = sum(t.pow(2).sum() for t in v).sqrt()
    v   = [t / v_n for t in v]

    eigenvalue = 0.0
    eps        = 1e-3

    for _ in range(n_steps):
        # Perturb +εv
        with torch.no_grad():
            for p, vi in zip(params, v):
                p.data.add_(vi, alpha=eps)
        g_plus = get_gradient()

        # Perturb -2εv  (undo +εv and apply -εv)
        with torch.no_grad():
            for p, vi in zip(params, v):
                p.data.sub_(vi, alpha=2 * eps)
        g_minus = get_gradient()

        # Restore
        with torch.no_grad():
            for p, vi in zip(params, v):
                p.data.add_(vi, alpha=eps)

        # Hv ≈ (g+ - g-) / (2ε)
        Hv = [(gp - gm) / (2 * eps) for gp, gm in zip(g_plus, g_minus)]

        # Rayleigh quotient: λ = v^T H v / v^T v = v^T Hv  (v is unit)
        eigenvalue = float(sum((vi * Hvi).sum() for vi, Hvi in zip(v, Hv)))

        # Normalise Hv to get next v
        Hv_n = sum(t.pow(2).sum() for t in Hv).sqrt() + 1e-12
        v    = [t / Hv_n for t in Hv]

    model.train()
    return abs(eigenvalue)


# ===========================================================================
# Grokking-specific metrics
# ===========================================================================

def compute_grokking_metrics(history_dict: dict) -> dict:
    """
    Derive grokking-specific summary statistics from a TrainingHistory dict.

    Parameters
    ----------
    history_dict : Output of TrainingHistory.to_dict()

    Returns
    -------
    dict with:
        "grokked"             : bool
        "memorization_step"   : int  (-1 if never)
        "grokking_step"       : int  (-1 if never)
        "grokking_gap"        : int  (S_G - S_mem, -1 if incomplete)
        "weight_l2_at_grok"   : float  (global L2 norm at grokking step)
        "weight_l2_at_mem"    : float
        "l2_ratio"            : float  (l2_at_grok / l2_at_mem)
        "max_gen_gap"         : float  (max train_acc - val_acc)
        "final_val_acc"       : float
        "final_train_acc"     : float
    """
    steps   = history_dict["steps"]
    val_acc = history_dict["val_acc"]
    tr_acc  = history_dict["train_acc"]
    l2      = history_dict["weight_l2"]

    S_mem   = history_dict["memorization_step"]
    S_G     = history_dict["grokking_step"]

    def _val_at_step(s: int, field: list) -> float:
        if s < 0 or len(steps) == 0:
            return float("nan")
        # Find nearest logged step
        diffs  = [abs(st - s) for st in steps]
        idx    = int(np.argmin(diffs))
        return float(field[idx])

    l2_at_grok = _val_at_step(S_G,   l2)
    l2_at_mem  = _val_at_step(S_mem, l2)
    l2_ratio   = (l2_at_grok / l2_at_mem) if (l2_at_mem > 0) else float("nan")

    gen_gap = [ta - va for ta, va in zip(tr_acc, val_acc)]

    return {
        "grokked"           : history_dict.get("grokked", False),
        "memorization_step" : S_mem,
        "grokking_step"     : S_G,
        "grokking_gap"      : history_dict.get("grokking_gap", -1),
        "weight_l2_at_grok" : l2_at_grok,
        "weight_l2_at_mem"  : l2_at_mem,
        "l2_ratio"          : l2_ratio,
        "max_gen_gap"       : float(max(gen_gap)) if gen_gap else float("nan"),
        "final_val_acc"     : float(val_acc[-1]) if val_acc else float("nan"),
        "final_train_acc"   : float(tr_acc[-1])  if tr_acc  else float("nan"),
    }
