"""Training, evaluation schedules, event detection, and checkpoints.

Accuracy and loss use a fine early evaluation schedule and a coarser later one.
Costlier metrics use an independent interval. Memorization and grokking events
are detected after training from the recorded accuracy curves. Each run writes
CSV metrics, JSON history and summary files, and requested checkpoints.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.logging_utils import MetricLogger
from src.metrics import (
    compute_weight_norms,
    compute_sparsity_from_masks,
    compute_fourier_features,
    compute_hessian_top_eigenvalue,
)


# Disk checkpoint + mask helpers

def save_init_checkpoint(
    model: nn.Module,
    run_dir: str | Path,
    filename: str = "init_weights.pt",
) -> Path:
    """
    Save exact step-0 model weights to disk BEFORE any gradient update.

    The saved file is the rewind source used by the pruning code. Call this
    before the first optimizer update.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / filename
    torch.save(
        {"step": 0, "event": "init", "state_dict": copy.deepcopy(model.state_dict())},
        ckpt_path,
    )
    return ckpt_path.resolve()


def save_checkpoint(
    model: nn.Module,
    run_dir: str | Path,
    step: int,
    event: str,
    metadata: dict | None = None,
) -> Path:
    """Save a named mid-training checkpoint to disk."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / f"ckpt_step{step:07d}_{event}.pt"
    payload = {"step": step, "event": event, "state_dict": copy.deepcopy(model.state_dict())}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, ckpt_path)
    return ckpt_path.resolve()


def load_checkpoint_from_disk(path: str | Path) -> dict:
    """Load a checkpoint written by save_init_checkpoint / save_checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Ensure save_init_checkpoint() was called before IMP and that "
            "run_dir is consistent across rounds."
        )
    return torch.load(path, map_location="cpu")


def save_masks(masks: dict[str, torch.Tensor], path: str | Path) -> Path:
    """Persist a pruning mask dict to disk (boolean storage to save space)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().to(torch.bool).cpu() for k, v in masks.items()}, path)
    return path.resolve()


def load_masks(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a mask dict saved by :func:`save_masks` (returns float tensors)."""
    raw = torch.load(Path(path), map_location="cpu")
    return {k: v.float() for k, v in raw.items()}


# Evaluation schedule

@dataclass
class EvalSchedule:
    """
    Cheap-eval cadence.  Defaults: every 5 steps for the first 1000 steps, then
    every 25.  Step 0 and the final step are always evaluated by the Trainer.
    """
    fine_until: int = 1000
    fine_interval: int = 5
    coarse_interval: int = 25

    def interval_at(self, step: int) -> int:
        return self.fine_interval if step < self.fine_until else self.coarse_interval

    def should_eval(self, step: int) -> bool:
        return step % self.interval_at(step) == 0

    def to_dict(self) -> dict:
        return {
            "fine_until": self.fine_until,
            "fine_interval": self.fine_interval,
            "coarse_interval": self.coarse_interval,
        }

    @classmethod
    def from_config(cls, cfg) -> "EvalSchedule":
        """Build from an OmegaConf node / dict / int (int → fixed interval)."""
        if cfg is None:
            return cls()
        if isinstance(cfg, int):
            return cls(fine_until=0, fine_interval=cfg, coarse_interval=cfg)
        get = (lambda k, d: cfg.get(k, d)) if hasattr(cfg, "get") else (lambda k, d: getattr(cfg, k, d))
        return cls(
            fine_until=int(get("fine_until", 1000)),
            fine_interval=int(get("fine_interval", 5)),
            coarse_interval=int(get("coarse_interval", 25)),
        )


def detect_threshold_crossing(
    steps: Sequence[int],
    values: Sequence[float],
    threshold: float,
    window: int,
) -> int:
    """
    Detect an event from a logged curve after training.

    Returns the FIRST logged step that begins a run of ``window`` consecutive
    logged evaluations all ``>= threshold``, or -1 if no such run exists. The
    event remains quantized to the evaluation schedule.
    """
    window = max(1, int(window))
    run = 0
    for i, v in enumerate(values):
        if v >= threshold:
            run += 1
            if run >= window:
                return int(steps[i - window + 1])
        else:
            run = 0
    return -1


# Training history

@dataclass
class TrainingHistory:
    """Result container for one training run (checkpoint *paths*, not weights)."""

    # Cheap-eval schedule (these lists are all the same length).
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    sparsity: list[float] = field(default_factory=list)

    # Expensive-metrics schedule (own step axis; sparser than `steps`).
    metric_steps: list[int] = field(default_factory=list)
    weight_l2: list[float] = field(default_factory=list)
    weight_l1: list[float] = field(default_factory=list)

    # Populated from the completed curves.
    memorization_step: int = -1
    grokking_step: int = -1

    # Resolution context so timing uncertainty is explicit.
    eval_resolution: dict = field(default_factory=dict)

    checkpoint_paths: dict = field(default_factory=dict)
    fourier_data: dict = field(default_factory=dict)
    config_summary: dict = field(default_factory=dict)

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
            "steps": self.steps,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "sparsity": self.sparsity,
            "metric_steps": self.metric_steps,
            "weight_l2": self.weight_l2,
            "weight_l1": self.weight_l1,
            "memorization_step": self.memorization_step,
            "grokking_step": self.grokking_step,
            "grokking_gap": self.grokking_gap,
            "grokked": self.grokked,
            "memorized": self.memorized,
            "eval_resolution": self.eval_resolution,
            "checkpoint_paths": self.checkpoint_paths,
            "fourier_data": self.fourier_data,
            "config_summary": self.config_summary,
        }

    def save_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Trainer

class Trainer:
    """
    Step-based training engine with the two-schedule measurement design,
    post-hoc grokking detection, unified logging, and disk checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        run_dir: str | Path,
        p: int = 97,
        eval_schedule: EvalSchedule | None = None,
        metrics_every: int = 1000,
        grok_threshold: float = 0.95,
        mem_threshold: float = 0.95,
        grok_window: int = 2,
        compute_fourier: bool = True,
        compute_hessian: bool = False,
        logging_backend: str = "tensorboard",
        wandb_run=None,
        use_amp: bool = False,
        early_stop: dict | None = None,
        is_baseline: bool = False,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.run_dir = Path(run_dir)
        self.p = p
        self.eval_schedule = eval_schedule or EvalSchedule()
        self.metrics_every = max(1, int(metrics_every))
        self.grok_threshold = grok_threshold
        self.mem_threshold = mem_threshold
        self.grok_window = max(1, int(grok_window))
        self.compute_fourier = compute_fourier
        self.compute_hessian = compute_hessian
        self.logging_backend = logging_backend
        self.wandb_run = wandb_run
        self.is_baseline = is_baseline

        # Early stop config: {"enabled": bool, "patience": int (logged steps)}.
        es = early_stop or {}
        self._es_enabled = bool(es.get("enabled", False))
        self._es_patience = int(es.get("patience", 500))

        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.criterion = nn.CrossEntropyLoss()

        self.run_dir.mkdir(parents=True, exist_ok=True)


    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
            else:
                logits = self.model(x)
                loss = self.criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(-1) == y).sum().item()
            total_n += y.size(0)
        self.model.train()
        return total_loss / total_n, total_correct / total_n

    def _cyclic(self, loader: DataLoader) -> Iterator:
        while True:
            yield from loader

    def _grad_global_norm(self) -> float:
        sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum())
        return sq ** 0.5

    def _fourier_dict(self) -> dict | None:
        if not self.compute_fourier or not hasattr(self.model, "get_embedding_weights"):
            return None
        return compute_fourier_features(self.model.get_embedding_weights(), self.p)


    def train(
        self,
        n_steps: int,
        masks: dict | None = None,
        save_checkpoints: bool = True,
        verbose: bool = True,
        config_summary: dict | None = None,
    ) -> TrainingHistory:
        """
        Run up to ``n_steps`` full-batch gradient updates.

        Memorization and grokking steps are computed from the logged
        curves after the loop; a lightweight live detector is used only to
        trigger checkpoint/mask saves and (optionally) early stopping.
        """
        history = TrainingHistory()
        history.eval_resolution = {**self.eval_schedule.to_dict(), "metrics_every": self.metrics_every}
        if config_summary:
            history.config_summary.update(config_summary)
        if masks is not None:
            history.config_summary["sparsity"] = float(compute_sparsity_from_masks(masks))

        logger = MetricLogger(
            run_dir=self.run_dir,
            backend=self.logging_backend,
            wandb_run=self.wandb_run,
        )

        self.model.train()
        data_iter = self._cyclic(self.train_loader)
        cons_mem = cons_grok = 0
        live_grok_step = -1
        post_grok_logged = 0
        t0 = time.time()

        try:
            for step in range(n_steps + 1):
                do_eval = self.eval_schedule.should_eval(step) or step == n_steps
                do_metrics = do_eval and (step % self.metrics_every == 0 or step == n_steps)

                # Forward + backward (skip the synthetic final eval-only step)
                grad_norm = None
                if step < n_steps:
                    x, y = next(data_iter)
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.use_amp:
                        with torch.amp.autocast("cuda"):
                            loss = self.criterion(self.model(x), y)
                        self.scaler.scale(loss).backward()
                        if do_metrics:
                            self.scaler.unscale_(self.optimizer)
                            grad_norm = self._grad_global_norm()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss = self.criterion(self.model(x), y)
                        loss.backward()
                        if do_metrics:
                            grad_norm = self._grad_global_norm()
                        self.optimizer.step()

                    # Re-apply pruning mask after every update.
                    if masks is not None:
                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if name in masks:
                                    param.data.mul_(masks[name].to(param.device))

                if not do_eval:
                    continue

                # Cheap eval
                train_loss, train_acc = self.evaluate(self.train_loader)
                val_loss, val_acc = self.evaluate(self.val_loader)
                sparsity = compute_sparsity_from_masks(masks) if masks else 0.0

                history.steps.append(step)
                history.train_loss.append(train_loss)
                history.train_acc.append(train_acc)
                history.val_loss.append(val_loss)
                history.val_acc.append(val_acc)
                history.sparsity.append(sparsity)

                scalars: dict[str, float] = {
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "sparsity": sparsity,
                }

                # Expensive metrics
                if do_metrics:
                    norms = compute_weight_norms(self.model, masks)
                    history.metric_steps.append(step)
                    history.weight_l2.append(norms["global_l2"])
                    history.weight_l1.append(norms["global_l1"])
                    scalars["weights/l2"] = norms["global_l2"]
                    scalars["weights/l1"] = norms["global_l1"]
                    if grad_norm is not None:
                        scalars["train/grad_norm"] = grad_norm

                    fd = self._fourier_dict()
                    if fd is not None:
                        scalars["fourier/concentration"] = fd["concentration"]
                        scalars["fourier/spectral_entropy"] = fd["entropy"]

                    if self.compute_hessian:
                        try:
                            lam = compute_hessian_top_eigenvalue(
                                self.model, self.criterion, self.val_loader,
                                self.device, masks=masks, n_steps=15, n_batches=4,
                            )
                            scalars["hessian/lambda_max"] = lam
                        except Exception:
                            pass  # Leave training running if this optional metric fails.

                logger.log_scalars(step, scalars)

                # Live detection (drives checkpoint/mask saves only)
                cons_mem = cons_mem + 1 if train_acc >= self.mem_threshold else 0
                if cons_mem == self.grok_window and "memorization" not in history.checkpoint_paths:
                    self._record_fourier(history, "memorization")
                    if save_checkpoints:
                        ckpt = save_checkpoint(
                            self.model, self.run_dir, step, "memorization",
                            metadata={"train_acc": train_acc, "val_acc": val_acc},
                        )
                        history.checkpoint_paths["memorization"] = str(ckpt)

                cons_grok = cons_grok + 1 if val_acc >= self.grok_threshold else 0
                if cons_grok == self.grok_window and "grokking" not in history.checkpoint_paths:
                    live_grok_step = step
                    self._record_fourier(history, "grokking")
                    if save_checkpoints:
                        ckpt = save_checkpoint(
                            self.model, self.run_dir, step, "grokking",
                            metadata={"val_acc": val_acc},
                        )
                        history.checkpoint_paths["grokking"] = str(ckpt)
                    if masks is not None and save_checkpoints:
                        mp = save_masks(masks, self.run_dir / "grokked_mask.pt")
                        history.checkpoint_paths["grokked_mask"] = str(mp)
                    if verbose:
                        print(f"  GROKKED (live)  step={step:,}  val={val_acc:.3f}")

                if verbose and step % max(self.metrics_every, 1) == 0:
                    print(
                        f"  step {step:6,}/{n_steps:,} | tr={train_acc:.3f} "
                        f"vl={val_acc:.3f} | sp={sparsity:.2f} | {time.time()-t0:.0f}s"
                    )

                # Early stop (sparse / non-baseline only)
                if (self._es_enabled and not self.is_baseline and live_grok_step >= 0):
                    post_grok_logged += 1
                    if post_grok_logged >= self._es_patience:
                        if verbose:
                            print(
                                f"  Early stop: {self._es_patience} logged evaluations "
                                "including grokking confirmation."
                            )
                        break

        finally:
            # End-of-run final checkpoint + post-hoc detection + summary.
            if save_checkpoints:
                last_step = history.steps[-1] if history.steps else n_steps
                ckpt = save_checkpoint(self.model, self.run_dir, last_step, "final")
                history.checkpoint_paths["final"] = str(ckpt)

            history.memorization_step = detect_threshold_crossing(
                history.steps, history.train_acc, self.mem_threshold, self.grok_window
            )
            history.grokking_step = detect_threshold_crossing(
                history.steps, history.val_acc, self.grok_threshold, self.grok_window
            )
            history.eval_resolution["resolution_at_grok"] = (
                self.eval_schedule.interval_at(history.grokking_step)
                if history.grokking_step >= 0 else None
            )

            history.save_json(self.run_dir / "history.json")
            self._write_summary(history)
            logger.close()

        if verbose:
            print(
                f"  detected  mem={history.memorization_step}  "
                f"grok={history.grokking_step}  gap={history.grokking_gap}"
            )
        return history


    def _record_fourier(self, history: TrainingHistory, tag: str) -> None:
        fd = self._fourier_dict()
        if fd is not None:
            history.fourier_data[tag] = fd

    def _write_summary(self, history: TrainingHistory) -> None:
        """Write the per-run summary.json."""
        summary = {
            "config": history.config_summary,
            "actual_sparsity": history.config_summary.get("sparsity", 0.0),
            "memorization_step": history.memorization_step,
            "grokking_step": history.grokking_step,
            "grokking_gap": history.grokking_gap,
            "final_train_acc": history.train_acc[-1] if history.train_acc else None,
            "final_val_acc": history.val_acc[-1] if history.val_acc else None,
            "grokked": history.grokked,
            "eval_resolution": history.eval_resolution,
            "checkpoint_paths": history.checkpoint_paths,
        }
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


# Optimizer factory

def _split_decay_params(model: nn.Module):
    """Weight decay applies to >=2-D weights only (never biases / LayerNorm)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def make_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    betas: tuple = (0.9, 0.98),
    eps: float = 1e-8,
) -> optim.Optimizer:
    """
    Config-driven optimizer factory.

    Implemented
    -----------
        "adamw": default; weight decay excluded from 1-D and LayerNorm params.

    Listed but not implemented
    --------------------------
        "sgd", "adam", "muon".
    """
    name = (name or "adamw").lower()
    decay, no_decay = _split_decay_params(model)

    if name == "adamw":
        return optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr, betas=betas, eps=eps,
        )

    if name in ("sgd", "adam", "muon"):
        raise NotImplementedError(
            f"Optimizer {name!r} is not implemented. "
            "Add a branch in src.train.make_optimizer."
        )

    raise ValueError(f"Unknown optimizer name: {name!r}")
