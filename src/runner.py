"""Build data, model, optimizer, logger, and trainer objects from Hydra config."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf


def enable_utf8_stdout() -> None:
    """
    Force UTF-8 on stdout/stderr so prints with unicode (arrows, ×, box chars)
    don't crash on a Windows cp1252 console while staying correct on Kaggle.
    No-op if the stream can't be reconfigured.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

from src.data import get_dataloaders
from src.model import get_model
from src.train import EvalSchedule, Trainer, make_optimizer



def resolve_device(cfg: DictConfig) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def resolve_seeds(cfg: DictConfig) -> list[int]:
    """
    Seed list for a run. A launcher sets GROK_SEED to pin a single seed; otherwise
    ``num_seeds`` consecutive seeds starting at ``cfg.seed`` are used. ``num_seeds``
    is therefore a real, config/CLI-driven knob.
    """
    env_seed = os.environ.get("GROK_SEED")
    if env_seed is not None:
        return [int(env_seed)]
    return [int(cfg.seed) + i for i in range(int(cfg.num_seeds))]


def seed_everything(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: DictConfig, seed: int):
    """Build (train, val) loaders. Each seed gets its own reproducible split."""
    return get_dataloaders(
        p=cfg.dataset.p,
        operation=cfg.dataset.operation,
        train_frac=cfg.dataset.train_frac,
        batch_size=cfg.training.batch_size,
        full_batch=cfg.training.full_batch,
        seed=seed,
    )


def build_model(cfg: DictConfig, device: torch.device):
    return get_model(
        vocab_size=cfg.dataset.p + 2,
        n_classes=cfg.dataset.p,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        seq_len=cfg.model.seq_len,
        layernorm=cfg.model.layernorm,
    ).to(device)


def build_optimizer(cfg: DictConfig, model, weight_decay: float | None = None):
    """Build the optimizer; ``weight_decay`` overrides cfg when provided."""
    wd = cfg.training.weight_decay if weight_decay is None else weight_decay
    return make_optimizer(
        model,
        name=cfg.training.optimizer,
        lr=cfg.training.lr,
        weight_decay=wd,
        betas=(cfg.training.beta1, cfg.training.beta2),
        eps=cfg.training.eps,
    )


def eval_schedule_from_cfg(cfg: DictConfig) -> EvalSchedule:
    return EvalSchedule.from_config(cfg.training.eval_every)


def build_trainer(
    cfg: DictConfig,
    model,
    train_loader,
    val_loader,
    optimizer,
    device: torch.device,
    run_dir: str | Path,
    *,
    is_baseline: bool = False,
    compute_fourier: bool | None = None,
    logging_backend: str | None = None,
    wandb_run=None,
) -> Trainer:
    """Construct a Trainer with the configured evaluation schedules."""
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        run_dir=run_dir,
        p=cfg.dataset.p,
        eval_schedule=eval_schedule_from_cfg(cfg),
        metrics_every=cfg.training.metrics_every,
        grok_threshold=cfg.training.grok_threshold,
        mem_threshold=cfg.training.mem_threshold,
        grok_window=cfg.training.grok_window,
        compute_fourier=(cfg.pruning.compute_fourier if compute_fourier is None else compute_fourier),
        compute_hessian=cfg.get("compute_hessian", False),
        logging_backend=(logging_backend if logging_backend is not None else cfg.logging.backend),
        wandb_run=wandb_run,
        use_amp=cfg.use_amp,
        early_stop=OmegaConf.to_container(cfg.training.early_stop, resolve=True),
        is_baseline=is_baseline,
    )


# Optional Weights & Biases (opt-in only)

def wandb_enabled(cfg: DictConfig) -> bool:
    return str(cfg.logging.backend).lower() == "wandb" and bool(cfg.wandb.get("enabled", False))


def maybe_init_wandb(cfg: DictConfig, run_name: str, group: str | None = None):
    """
    Start a wandb run when logging.backend=='wandb' and wandb.enabled.
    Returns the run (or None). Never logs in or reads WANDB_API_KEY itself.
    """
    if not wandb_enabled(cfg):
        return None
    try:
        import wandb
    except ImportError:
        print("  [wandb] requested but not installed; continuing with CSV only.")
        return None
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        group=group,
        tags=list(cfg.wandb.get("tags", [])),
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )


def finish_wandb(run) -> None:
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass
