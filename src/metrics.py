"""Metrics used by the training and analysis code.

The module provides weight norms, mask sparsity, embedding Fourier summaries,
effective rank, gradient signal-to-noise ratio, grokking event summaries, and a
top-Hessian-eigenvalue estimate based on autograd Hessian-vector products.
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


# Weight Norm Metrics

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


# Sparsity

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


# Fourier features

def compute_fourier_features(
    embedding_weights: torch.Tensor,   # (vocab_size, d_model)
    p: int,
    top_k: int = 5,
) -> dict:
    """
    Summarize the Fourier structure of number-token embeddings.

    The function computes a DFT over the first ``p`` embedding rows and reports
    the mean power spectrum, dominant frequencies, top-frequency concentration,
    and spectral entropy. These are descriptive embedding statistics.

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
        "power_spectrum"   : mean power per frequency
        "frequencies"      : frequency indices (0 .. p//2)
        "top_frequencies"  : indices of top-k dominant frequencies
        "top_powers"       : power at the top-k frequencies
        "concentration"    : power at the top frequency divided by total power
        "entropy"          : entropy of the normalized mean power spectrum
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


# Effective Rank

def compute_effective_rank(weight_matrix: torch.Tensor) -> float:
    """
    Compute the effective rank of a 2-D weight matrix.

    Effective rank (Roy & Vetterli, 2007) = exp(H(σ)), where H is the
    Shannon entropy of the normalised singular values.  A low effective rank
    indicates that the singular-value distribution is concentrated in fewer
    dimensions.

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


# Gradient Signal-to-Noise Ratio (GSNR)

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


# Hessian Top Eigenvalue  (power iteration)

def compute_hessian_top_eigenvalue(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    masks: dict[str, torch.Tensor] | None = None,
    n_steps: int = 20,
    n_batches: int = 4,
) -> float:
    """
    Estimate the largest Hessian eigenvalue using Hessian-vector products.

    Method
    ------
    Uses autograd double backward rather than finite differences::

        g  = ∇_θ L                       (create_graph=True)
        Hv = ∇_θ (gᵀ v)                   (torch.autograd.grad of the inner product)

    It power-iterates ``v ← Hv / ‖Hv‖`` and reports the Rayleigh quotient
    ``vᵀ H v``.

    When masks are provided, the probe vector and each HVP are zeroed in pruned
    dimensions, so power iteration remains in the active subspace. The metric is
    disabled by default because it is computationally expensive.

    Parameters
    ----------
    masks     : Optional {param_name → 0/1 tensor}; ``v`` is confined to mask==1.
    n_steps   : Power-iteration steps.
    n_batches : Batches summed into the loss whose Hessian is probed.

    Returns
    -------
    Estimated top Hessian eigenvalue (float).
    """
    was_training = model.training
    model.eval()

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]

    def active_mask(name: str, ref: torch.Tensor) -> torch.Tensor:
        if masks is not None and name in masks:
            return masks[name].to(device=ref.device, dtype=ref.dtype)
        return torch.ones_like(ref)

    pmasks = [active_mask(n, p) for n, p in named]

    # Build a single differentiable loss over n_batches, then its gradient graph.
    loader_iter = iter(loader)
    total_loss = None
    count = 0
    for _ in range(n_batches):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        batch_loss = criterion(model(x), y)
        total_loss = batch_loss if total_loss is None else total_loss + batch_loss
        count += 1
    if count == 0:
        if was_training:
            model.train()
        return 0.0
    loss = total_loss / count
    grads = torch.autograd.grad(loss, params, create_graph=True)

    def _restrict(vs):
        return [v * m for v, m in zip(vs, pmasks)]

    def _norm(vs):
        return float(sum(v.pow(2).sum() for v in vs)) ** 0.5

    # Random unit probe vector confined to the active subspace.
    v = _restrict([torch.randn_like(p) for p in params])
    nrm = _norm(v) + 1e-12
    v = [vi / nrm for vi in v]

    eigenvalue = 0.0
    for _ in range(n_steps):
        gv = sum((g * vi).sum() for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(gv, params, retain_graph=True)
        Hv = _restrict([hv.detach() for hv in Hv])
        eigenvalue = float(sum((vi * hv).sum() for vi, hv in zip(v, Hv)))  # Rayleigh (v unit)
        nrm = _norm(Hv) + 1e-12
        v = [hv / nrm for hv in Hv]

    if was_training:
        model.train()
    return abs(eigenvalue)


# Grokking-specific metrics

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
    # Weight norms live on their own (sparser) step axis; fall back to `steps`
    # for older histories that did not record metric_steps.
    l2_steps = history_dict.get("metric_steps") or steps

    S_mem   = history_dict["memorization_step"]
    S_G     = history_dict["grokking_step"]

    def _val_at_step(s: int, field: list, axis: list) -> float:
        if s < 0 or len(field) == 0 or len(axis) == 0:
            return float("nan")
        # Find nearest logged step on the relevant axis.
        idx = int(np.argmin([abs(st - s) for st in axis[: len(field)]]))
        return float(field[idx])

    l2_at_grok = _val_at_step(S_G,   l2, l2_steps)
    l2_at_mem  = _val_at_step(S_mem, l2, l2_steps)
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
