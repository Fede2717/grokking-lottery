"""
src/train.py — Training Engine v2 (W&B + Strict Disk Checkpointing)
====================================================================

Changes from v1
---------------
    W&B integration     : all metrics logged at every eval step.
                          Heavy metrics (Hessian λ_max) on a sparse
                          schedule to avoid T4 overhead.
    Disk checkpointing  : model weights saved to disk at step=0 BEFORE
                          any gradient update. IMP rewinding loads from
                          this file — never from an in-memory dict.
    run_dir             : each Trainer has a dedicated output directory;
                          all .pt files and history.json land there.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.metrics import (
    compute_weight_norms,
    compute_sparsity_from_masks,
    compute_fourier_features,
    compute_hessian_top_eigenvalue,
)

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ===========================================================================
# Disk Checkpoint Helpers  (Requirement 5)
# ===========================================================================

def save_init_checkpoint(
    model:    nn.Module,
    run_dir:  str | Path,
    filename: str = "init_weights.pt",
) -> Path:
    """
    Save exact step-0 model weights to disk BEFORE any gradient update.

    This is the canonical source of truth for LTH weight rewinding.
    Loading from disk (not an in-memory dict) prevents silent state
    corruption across Python processes and Kaggle kernel restarts.

    CALL THIS BEFORE optimizer.step() IS EVER INVOKED.

    Returns the absolute path to the saved file.
    """
    run_dir   = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / filename

    torch.save(
        {
            "step"       : 0,
            "event"      : "init",
            "state_dict" : copy.deepcopy(model.state_dict()),
        },
        ckpt_path,
    )
    return ckpt_path.resolve()


def save_checkpoint(
    model:    nn.Module,
    run_dir:  str | Path,
    step:     int,
    event:    str,
    metadata: dict | None = None,
) -> Path:
    """Save a named mid-training checkpoint to disk."""
    run_dir   = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    filename  = f"ckpt_step{step:07d}_{event}.pt"
    ckpt_path = run_dir / filename

    payload = {
        "step"       : step,
        "event"      : event,
        "state_dict" : copy.deepcopy(model.state_dict()),
    }
    if metadata:
        payload["metadata"] = metadata

    torch.save(payload, ckpt_path)
    return ckpt_path.resolve()


def load_checkpoint_from_disk(path: str | Path) -> dict:
    """
    Load a checkpoint saved by save_init_checkpoint or save_checkpoint.
    Raises FileNotFoundError with a helpful message if missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Ensure save_init_checkpoint() was called before IMP and that "
            "run_dir is consistent across IMP rounds."
        )
    return torch.load(path, map_location="cpu")


# ===========================================================================
# Training History
# ===========================================================================

@dataclass
class TrainingHistory:
    """
    Immutable result container for one training run.

    v2 change: stores checkpoint *paths* (str) not state dicts.
    Load weights with load_checkpoint_from_disk(path).
    """

    steps:        list[int]   = field(default_factory=list)
    train_loss:   list[float] = field(default_factory=list)
    train_acc:    list[float] = field(default_factory=list)
    val_loss:     list[float] = field(default_factory=list)
    val_acc:      list[float] = field(default_factory=list)
    weight_l2:    list[float] = field(default_factory=list)
    weight_l1:    list[float] = field(default_factory=list)
    grad_norm:    list[float] = field(default_factory=list)
    sparsity:     list[float] = field(default_factory=list)

    memorization_step: int  = -1
    grokking_step:     int  = -1

    # Paths (str) to disk checkpoints — load with load_checkpoint_from_disk
    checkpoint_paths: dict  = field(default_factory=dict)

    fourier_data:    dict   = field(default_factory=dict)
    config_summary:  dict   = field(default_factory=dict)

    @property
    def grokking_gap(self) -> int:
        if self.memorization_step < 0 or self.grokking_step < 0:
            return -1
        return self.grokking_step - self.memorization_step

    @property
    def grokked(self) -> bool:
        return self.grokking_step >= 0

    @property
    def memorized(self) -> bool:
        return self.memorization_step >= 0

    def to_dict(self) -> dict:
        return {
            "steps"            : self.steps,
            "train_loss"       : self.train_loss,
            "train_acc"        : self.train_acc,
            "val_loss"         : self.val_loss,
            "val_acc"          : self.val_acc,
            "weight_l2"        : self.weight_l2,
            "weight_l1"        : self.weight_l1,
            "grad_norm"        : self.grad_norm,
            "sparsity"         : self.sparsity,
            "memorization_step": self.memorization_step,
            "grokking_step"    : self.grokking_step,
            "grokking_gap"     : self.grokking_gap,
            "grokked"          : self.grokked,
            "memorized"        : self.memorized,
            "checkpoint_paths" : self.checkpoint_paths,
            "fourier_data"     : self.fourier_data,
            "config_summary"   : self.config_summary,
        }

    def save_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ===========================================================================
# W&B helper
# ===========================================================================

def init_wandb(cfg_dict: dict, run_name: str, group: str | None = None) -> bool:
    """Initialise a W&B run from a flat/nested config dict."""
    if not _WANDB_AVAILABLE:
        print("  [W&B] Not installed — skipping. Run: pip install wandb")
        return False

    wb = cfg_dict.get("wandb", {})
    if not wb.get("enabled", False):
        return False

    _wandb.init(
        project = wb.get("project", "grokking-lottery"),
        entity  = wb.get("entity",  None),
        name    = run_name,
        group   = group or cfg_dict.get("experiment", {}).get("name", "default"),
        tags    = list(wb.get("tags", [])),
        config  = cfg_dict,
        resume  = "allow",
    )
    return True


# ===========================================================================
# Trainer
# ===========================================================================

class Trainer:
    """
    Step-based training engine with grokking detection, W&B logging,
    and disk-based checkpoint management.
    """

    def __init__(
        self,
        model:           nn.Module,
        train_loader:    DataLoader,
        val_loader:      DataLoader,
        optimizer:       optim.Optimizer,
        device:          torch.device,
        run_dir:         str | Path,
        p:               int   = 97,
        log_every:       int   = 100,
        grok_threshold:  float = 0.95,
        mem_threshold:   float = 0.99,
        grok_window:     int   = 3,
        compute_fourier: bool  = True,
        use_amp:         bool  = True,
        wandb_cfg:       dict  | None = None,
        max_grad_norm:   float = 1.0,
    ) -> None:
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.optimizer      = optimizer
        self.device         = device
        self.run_dir        = Path(run_dir)
        self.p              = p
        self.log_every      = log_every
        self.grok_threshold = grok_threshold
        self.mem_threshold  = mem_threshold
        self.grok_window    = grok_window
        self.compute_fourier= compute_fourier
        self.max_grad_norm  = max_grad_norm

        self.use_amp = use_amp and device.type == "cuda"
        self.scaler  = GradScaler() if self.use_amp else None
        self.criterion = nn.CrossEntropyLoss()

        wc = wandb_cfg or {}
        self._wb_enabled       = wc.get("enabled", False) and _WANDB_AVAILABLE
        self._wb_log_every     = wc.get("log_every",      log_every)
        self._wb_hess_every    = wc.get("hessian_every",  5000)
        self._wb_fourier_every = wc.get("fourier_every",  1000)

        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with autocast(enabled=self.use_amp):
                logits = self.model(x)
                loss   = self.criterion(logits, y)
            total_loss    += loss.item() * y.size(0)
            total_correct += (logits.argmax(-1) == y).sum().item()
            total_n       += y.size(0)
        self.model.train()
        return total_loss / total_n, total_correct / total_n

    def _cyclic(self, loader: DataLoader) -> Iterator:
        while True:
            yield from loader

    def _log_fourier(self, history: TrainingHistory, tag: str, step: int) -> None:
        if not self.compute_fourier or not hasattr(self.model, "get_embedding_weights"):
            return
        fd = compute_fourier_features(self.model.get_embedding_weights(), self.p)
        history.fourier_data[tag] = fd
        if self._wb_enabled:
            _wandb.log({
                f"fourier/{tag}/concentration"   : fd["concentration"],
                f"fourier/{tag}/spectral_entropy": fd["entropy"],
            }, step=step)

    # ------------------------------------------------------------------

    def train(
        self,
        n_steps:               int,
        masks:                 dict | None = None,
        early_stop_after_grok: int | None  = None,
        save_checkpoints:      bool        = True,
        verbose:               bool        = True,
    ) -> TrainingHistory:
        """
        Run exactly n_steps gradient updates.

        PRECONDITION: save_init_checkpoint() MUST have been called before
        this method if IMP rewinding will be performed later.
        """
        history = TrainingHistory()
        if masks is not None:
            history.config_summary["sparsity"] = float(compute_sparsity_from_masks(masks))

        self.model.train()
        data_iter    = self._cyclic(self.train_loader)
        _cons_mem    = _cons_grok = _post_grok_steps = 0
        t0 = time.time()

        for step in range(n_steps):

            # ── Forward + backward ───────────────────────────────────────
            x, y = next(data_iter)
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                loss = self.criterion(self.model(x), y)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # ── Re-apply pruning mask ────────────────────────────────────
            if masks is not None:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in masks:
                            param.data.mul_(masks[name])

            # ── Periodic evaluation & logging ────────────────────────────
            if step % self.log_every != 0:
                continue

            train_loss, train_acc = self.evaluate(self.train_loader)
            val_loss,   val_acc   = self.evaluate(self.val_loader)
            norms     = compute_weight_norms(self.model, masks)
            sparsity  = compute_sparsity_from_masks(masks) if masks else 0.0

            history.steps.append(step)
            history.train_loss.append(train_loss)
            history.train_acc.append(train_acc)
            history.val_loss.append(val_loss)
            history.val_acc.append(val_acc)
            history.weight_l2.append(norms["global_l2"])
            history.weight_l1.append(norms["global_l1"])
            history.grad_norm.append(float(grad_norm))
            history.sparsity.append(sparsity)

            # ── W&B standard metrics ─────────────────────────────────────
            if self._wb_enabled:
                _wandb.log({
                    "train/loss"     : train_loss,
                    "train/acc"      : train_acc,
                    "val/loss"       : val_loss,
                    "val/acc"        : val_acc,
                    "weights/l2"     : norms["global_l2"],
                    "weights/l1"     : norms["global_l1"],
                    "train/grad_norm": float(grad_norm),
                    "sparsity"       : sparsity,
                }, step=step)

            # ── W&B Hessian (expensive — sparse schedule) ────────────────
            if self._wb_enabled and step > 0 and step % self._wb_hess_every == 0:
                try:
                    lam = compute_hessian_top_eigenvalue(
                        self.model, self.criterion, self.val_loader,
                        self.device, n_steps=15, n_batches=4,
                    )
                    _wandb.log({"hessian/top_eigenvalue": lam}, step=step)
                except Exception:
                    pass   # non-fatal numerical failure

            # ── W&B Fourier (medium schedule) ───────────────────────────
            if self._wb_enabled and step > 0 and step % self._wb_fourier_every == 0:
                if hasattr(self.model, "get_embedding_weights"):
                    fd = compute_fourier_features(
                        self.model.get_embedding_weights(), self.p
                    )
                    _wandb.log({
                        "fourier/concentration"   : fd["concentration"],
                        "fourier/spectral_entropy": fd["entropy"],
                    }, step=step)

            # ── Memorisation detection ───────────────────────────────────
            if history.memorization_step < 0:
                _cons_mem = _cons_mem + 1 if train_acc >= self.mem_threshold else 0
                if _cons_mem >= self.grok_window:
                    history.memorization_step = step
                    self._log_fourier(history, "memorization", step)
                    if save_checkpoints:
                        ckpt = save_checkpoint(
                            self.model, self.run_dir, step, "memorization",
                            metadata={"train_acc": train_acc, "val_acc": val_acc},
                        )
                        history.checkpoint_paths["memorization"] = str(ckpt)
                    if self._wb_enabled:
                        _wandb.log({"event/memorization_step": step}, step=step)
                    if verbose:
                        print(f"  ★ MEMORISED  step={step:,}  train={train_acc:.3f}")

            # ── Grokking detection ───────────────────────────────────────
            if history.grokking_step < 0:
                _cons_grok = _cons_grok + 1 if val_acc >= self.grok_threshold else 0
                if _cons_grok >= self.grok_window:
                    history.grokking_step = step
                    self._log_fourier(history, "grokking", step)
                    if save_checkpoints:
                        ckpt = save_checkpoint(
                            self.model, self.run_dir, step, "grokking",
                            metadata={
                                "val_acc"     : val_acc,
                                "grokking_gap": history.grokking_gap,
                                "weight_l2"   : norms["global_l2"],
                            },
                        )
                        history.checkpoint_paths["grokking"] = str(ckpt)
                    if self._wb_enabled:
                        _wandb.log({
                            "event/grokking_step" : step,
                            "event/grokking_gap"  : history.grokking_gap,
                            "event/l2_at_grok"    : norms["global_l2"],
                        }, step=step)
                    if verbose:
                        print(
                            f"  ★ GROKKED    step={step:,}  "
                            f"val={val_acc:.3f}  gap={history.grokking_gap:,}"
                        )

            # ── Verbose terminal ─────────────────────────────────────────
            if verbose and step % (self.log_every * 10) == 0:
                print(
                    f"  step {step:6,}/{n_steps:,} | "
                    f"tr={train_acc:.3f} vl={val_acc:.3f} | "
                    f"L2={norms['global_l2']:.3f} sp={sparsity:.2f} | "
                    f"{time.time()-t0:.0f}s"
                )

            # ── Early stop ───────────────────────────────────────────────
            if history.grokked and early_stop_after_grok is not None:
                _post_grok_steps += self.log_every
                if _post_grok_steps >= early_stop_after_grok:
                    if verbose:
                        print(f"  Early stop {early_stop_after_grok} steps post-grokking.")
                    break

        # ── End-of-run ───────────────────────────────────────────────────
        self._log_fourier(history, "final", n_steps)
        if save_checkpoints:
            ckpt = save_checkpoint(self.model, self.run_dir, n_steps, "final")
            history.checkpoint_paths["final"] = str(ckpt)

        history_path = self.run_dir / "history.json"
        history.save_json(history_path)

        if self._wb_enabled:
            art = _wandb.Artifact("training_history", type="result")
            art.add_file(str(history_path))
            _wandb.log_artifact(art)

        return history


# ===========================================================================
# Optimizer factory
# ===========================================================================

def make_optimizer(
    model:        nn.Module,
    lr:           float = 1e-3,
    weight_decay: float = 1e-3,
    betas:        tuple = (0.9, 0.98),
) -> optim.AdamW:
    """
    AdamW — weight decay NOT applied to 1-D params or LayerNorm.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if (param.dim() < 2 or "norm" in name) else decay).append(param)

    return optim.AdamW(
        [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr, betas=betas,
    )
