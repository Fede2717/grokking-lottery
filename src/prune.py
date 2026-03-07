"""
src/prune.py — Pruning Engine v2 (Disk-Based Weight Rewinding)
==============================================================

Key change from v1
------------------
    rewind_weights() now loads from a DISK path (produced by
    save_init_checkpoint in train.py) rather than from an in-memory dict.

    Rationale: IMP involves multiple Python processes (parallel seeds),
    Kaggle kernel restarts, and long training loops.  An in-memory dict
    can silently diverge — a disk file is the single source of truth.

    Every call to rewind_weights() performs:
        1. Load state_dict from path.
        2. Copy into model parameters (strict=True by default).
        3. Apply the current binary mask (zero out pruned weights).

API changes
-----------
    rewind_weights(model, init_ckpt_path: str|Path, masks)
        — was rewind_weights(model, initial_state: dict, masks)
    run_imp(... init_ckpt_path: str|Path ...)
        — was run_imp(... initial_state: dict ...)
    one_shot_prune(... init_ckpt_path: str|Path ...)
        — was one_shot_prune(... initial_state: dict ...)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from src.train import (
    load_checkpoint_from_disk,
    TrainingHistory,
)


# ===========================================================================
# Mask utilities
# ===========================================================================

def make_empty_masks(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return all-ones mask dict for every prunable parameter."""
    return {
        name: torch.ones_like(param)
        for name, param in model.get_prunable_named_parameters().items()
    }


def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    """Zero out all masked weights in-place. Call after every optimizer.step()."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.data.mul_(masks[name])


def compute_sparsity(masks: dict[str, torch.Tensor]) -> float:
    """Return fraction of pruned (mask==0) weights."""
    total  = sum(m.numel() for m in masks.values())
    active = sum(m.sum().item() for m in masks.values())
    return 1.0 - active / total if total > 0 else 0.0


def rewind_weights(
    model:           nn.Module,
    init_ckpt_path:  str | Path,
    masks:           dict[str, torch.Tensor],
    strict:          bool = True,
) -> None:
    """
    Rewind model weights to the DISK checkpoint at init_ckpt_path, then
    apply the current pruning mask.

    Parameters
    ----------
    model          : Model whose weights are reset in-place.
    init_ckpt_path : Path to the init_weights.pt file written by
                     save_init_checkpoint() BEFORE any training.
    masks          : Current binary mask (applied after weight restore).
    strict         : Passed to model.load_state_dict(). Default True.

    This is the critical step that distinguishes LTH from fine-tuning:
    we restore the architecture (mask) but not the learned weights.
    """
    ckpt = load_checkpoint_from_disk(init_ckpt_path)
    model.load_state_dict(ckpt["state_dict"], strict=strict)
    apply_masks(model, masks)


# ===========================================================================
# Global Magnitude Pruning
# ===========================================================================

def _global_threshold(
    model:           nn.Module,
    masks:           dict[str, torch.Tensor],
    target_sparsity: float,
) -> float:
    """
    Compute the magnitude threshold that achieves *target_sparsity* globally.
    Only active weights (mask==1) are considered.
    """
    active_mags: list[torch.Tensor] = []
    prunable = model.get_prunable_named_parameters()

    with torch.no_grad():
        for name, param in prunable.items():
            if name in masks:
                active = param.detach().abs()[masks[name].bool()]
            else:
                active = param.detach().abs().flatten()
            active_mags.append(active.cpu())

    all_mags = torch.cat(active_mags)
    if all_mags.numel() == 0:
        return 0.0

    total   = sum(m.numel() for m in masks.values())
    n_keep  = max(1, int(total * (1.0 - target_sparsity)))
    n_prune = all_mags.numel() - n_keep
    if n_prune <= 0:
        return 0.0

    sorted_mags, _ = torch.sort(all_mags)
    return float(sorted_mags[n_prune - 1])


def apply_global_magnitude_pruning(
    model:           nn.Module,
    masks:           dict[str, torch.Tensor],
    target_sparsity: float,
) -> dict[str, torch.Tensor]:
    """
    Update masks so that global sparsity == target_sparsity.
    Masks are monotone — never re-activated.
    """
    threshold = _global_threshold(model, masks, target_sparsity)
    prunable  = model.get_prunable_named_parameters()

    with torch.no_grad():
        for name, param in prunable.items():
            if name not in masks:
                continue
            to_prune      = (param.detach().abs() <= threshold)
            masks[name]   = masks[name] * (~to_prune).float()

    apply_masks(model, masks)
    return masks


# ===========================================================================
# One-shot Pruning  (ablation baseline)
# ===========================================================================

def one_shot_prune(
    model:           nn.Module,
    init_ckpt_path:  str | Path,
    target_sparsity: float,
) -> dict[str, torch.Tensor]:
    """
    One-shot magnitude pruning.

    1. Rank weights by magnitude (uses current model weights).
    2. Build global mask at target_sparsity.
    3. Rewind to init_ckpt_path and apply mask.

    Parameters
    ----------
    model           : Trained model (weights used to rank magnitudes).
    init_ckpt_path  : Path to step-0 checkpoint for rewinding.
    target_sparsity : Desired global sparsity in (0, 1).

    Returns
    -------
    Binary mask dict.
    """
    masks = make_empty_masks(model)
    masks = apply_global_magnitude_pruning(model, masks, target_sparsity)
    rewind_weights(model, init_ckpt_path, masks)
    return masks


# ===========================================================================
# Iterative Magnitude Pruning (IMP)
# ===========================================================================

@dataclass
class IMPResult:
    """Per-round diagnostics from an IMP run."""
    round_sparsities: list[float] = field(default_factory=list)
    round_val_accs:   list[float] = field(default_factory=list)
    final_masks:      dict        = field(default_factory=dict)
    final_sparsity:   float       = 0.0


def run_imp(
    model:                nn.Module,
    init_ckpt_path:       str | Path,
    trainer,
    target_sparsity:      float,
    prune_rate_per_round: float = 0.20,
    steps_per_round:      int   = 2_000,
) -> IMPResult:
    """
    Iterative Magnitude Pruning with disk-based weight rewinding.

    Algorithm (Frankle & Carlin, 2019)
    ------------------------------------
    k_rounds = ceil(log(1-s*) / log(1-p))  where s*=target, p=prune_rate

    For round k in 1..k_rounds:
        1. Train sparse net from rewound weights for T steps.
        2. Prune bottom p% of remaining active weights globally.
        3. Rewind: load init_ckpt_path, apply cumulative mask.

    Parameters
    ----------
    model                : Model (modified in-place).
    init_ckpt_path       : Path to init_weights.pt (from save_init_checkpoint).
                           MUST exist before this function is called.
    trainer              : Trainer instance — its .train() method is called.
    target_sparsity      : Final global sparsity target in [0, 1).
    prune_rate_per_round : Fraction of currently-active weights pruned per round.
    steps_per_round      : Gradient steps for short training between rounds.
    """
    init_ckpt_path = Path(init_ckpt_path)

    if not init_ckpt_path.exists():
        raise FileNotFoundError(
            f"IMP cannot proceed: init checkpoint not found at {init_ckpt_path}.\n"
            "Call save_init_checkpoint(model, run_dir) BEFORE run_imp()."
        )

    if target_sparsity <= 0.0:
        masks = make_empty_masks(model)
        return IMPResult(
            round_sparsities=[0.0], round_val_accs=[],
            final_masks=masks,     final_sparsity=0.0,
        )

    k_rounds = math.ceil(
        math.log(1.0 - target_sparsity) / math.log(1.0 - prune_rate_per_round)
    )

    print(
        f"\n{'='*60}\n"
        f"IMP:  target={target_sparsity:.1%}  rate={prune_rate_per_round:.0%}/round  "
        f"rounds={k_rounds}  ckpt={init_ckpt_path.name}\n"
        f"{'='*60}"
    )

    masks  = make_empty_masks(model)
    result = IMPResult()

    for rnd in range(k_rounds):
        round_target = min(
            1.0 - (1.0 - prune_rate_per_round) ** (rnd + 1),
            target_sparsity,
        )
        current_sp = compute_sparsity(masks)
        print(
            f"  Round {rnd+1:2d}/{k_rounds} | "
            f"sparsity: {current_sp:.1%} → {round_target:.1%}"
        )

        # Short training
        history = trainer.train(
            n_steps=steps_per_round, masks=masks,
            save_checkpoints=False, verbose=False,
        )
        val_acc = history.val_acc[-1] if history.val_acc else 0.0
        result.round_val_accs.append(val_acc)

        # Prune
        masks = apply_global_magnitude_pruning(model, masks, round_target)
        actual_sp = compute_sparsity(masks)
        result.round_sparsities.append(actual_sp)

        # Rewind from disk
        rewind_weights(model, init_ckpt_path, masks)

        print(f"          → sparsity={actual_sp:.1%}  val_acc={val_acc:.3f}")

    result.final_masks    = masks
    result.final_sparsity = compute_sparsity(masks)
    print(f"\nIMP complete.  Final sparsity = {result.final_sparsity:.1%}\n")
    return result
