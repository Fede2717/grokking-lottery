"""Experiment B: discover a mask, rewind to initialization, and train.

The ``imp`` method trains for 400 updates per round, prunes active weights, and
rewinds weights while retaining AdamW state. The ``one_shot`` method trains a
dense model for 1,200 updates and ranks magnitudes once. Final sparse training
starts from ``W0`` with a fresh optimizer.

In the retained results, every one-shot warm-up had already grokked. The IMP
round histories overwrite one another, so only the last round is preserved.
Use ``analysis/summarize_exp_b.py`` to reconstruct the archived grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_utils import load_summary, summary_to_aggregate_row, write_run_aggregate_csv
from src.metrics import compute_grokking_metrics
from src.prune import compute_sparsity, make_empty_masks, one_shot_prune, rewind_weights, run_imp
from src.runner import (
    build_dataloaders, build_model, build_optimizer, build_trainer, enable_utf8_stdout,
    finish_wandb, maybe_init_wandb, resolve_device, resolve_seeds, seed_everything,
)
from src.train import save_init_checkpoint


def run_single(
    cfg: DictConfig,
    target_sparsity: float,
    method: str,
    seed: int,
    device,
    exp_dir: Path,
) -> dict:
    """Run one (sparsity, method, seed) cell; return one aggregate row."""
    run_dir = exp_dir / method / f"sp_{target_sparsity:.2f}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed, device)
    train_loader, val_loader = build_dataloaders(cfg, seed)
    model = build_model(cfg, device)

    # Strict init checkpoint BEFORE any gradient update (LTH source of truth).
    init_ckpt = save_init_checkpoint(model, run_dir, cfg.checkpoint.init_filename)
    is_baseline = target_sparsity == 0.0

    label = f"sp={target_sparsity:.0%}_{method}_seed={seed}"
    print(f"\n{'='*65}\n  {label}\n{'='*65}")

    # Pruning phase
    if is_baseline:
        masks = make_empty_masks(model)
        discovery_config = {"kind": "none", "updates": 0}
    elif method == "imp":
        imp_opt = build_optimizer(cfg, model)
        imp_trainer = build_trainer(
            cfg, model, train_loader, val_loader, imp_opt, device,
            run_dir / "imp_phase", compute_fourier=False, logging_backend="none",
        )
        imp_result = run_imp(
            model=model, init_ckpt_path=init_ckpt, trainer=imp_trainer,
            target_sparsity=target_sparsity,
            prune_rate_per_round=cfg.pruning.prune_rate_per_round,
            steps_per_round=cfg.pruning.imp_steps_per_round,
        )
        masks = imp_result.final_masks
        discovery_config = {
            "kind": "stateful_optimizer_imp",
            "rounds": len(imp_result.round_sparsities),
            "updates_per_round": int(cfg.pruning.imp_steps_per_round),
            "total_updates": len(imp_result.round_sparsities) * int(cfg.pruning.imp_steps_per_round),
            "prune_rate_per_round": float(cfg.pruning.prune_rate_per_round),
            "weight_rewind": "W0",
            "optimizer_state_reset_between_rounds": False,
        }
    elif method == "one_shot":
        short_opt = build_optimizer(cfg, model)
        short_trainer = build_trainer(
            cfg, model, train_loader, val_loader, short_opt, device,
            run_dir / "oneshot_pretrain", compute_fourier=False, logging_backend="none",
        )
        warmup_steps = int(cfg.pruning.imp_steps_per_round) * 3
        short_trainer.train(
            n_steps=warmup_steps,
            save_checkpoints=False, verbose=False,
            config_summary={
                "experiment": "exp_b", "stage": "one_shot_warmup",
                "method": method, "target_sparsity": target_sparsity,
                "seed": seed, "warmup_steps": warmup_steps,
                "weight_decay": float(cfg.training.weight_decay),
            },
        )
        masks = one_shot_prune(model, init_ckpt, target_sparsity)
        discovery_config = {
            "kind": "dense_warmup_one_shot_magnitude",
            "updates": warmup_steps,
            "weight_decay": float(cfg.training.weight_decay),
            "weight_rewind": "W0",
        }
    else:
        raise ValueError(f"Unknown method: {method!r}")

    rewind_weights(model, init_ckpt, masks)
    actual_sp = compute_sparsity(masks)
    print(f"  actual sparsity = {actual_sp:.2%}")

    # Grokking phase
    grok_dir = run_dir / "grok_phase"
    wandb_run = maybe_init_wandb(cfg, run_name=label, group=f"exp_b_{method}")
    grok_opt = build_optimizer(cfg, model)
    grok_trainer = build_trainer(
        cfg, model, train_loader, val_loader, grok_opt, device, grok_dir,
        is_baseline=is_baseline, wandb_run=wandb_run,
    )
    history = grok_trainer.train(
        n_steps=cfg.training.n_grok_steps,
        masks=masks,
        save_checkpoints=True,
        config_summary={
            "experiment": "exp_b",
            "method": method,
            "target_sparsity": target_sparsity,
            "actual_sparsity": actual_sp,
            "seed": seed,
            "weight_decay": float(cfg.training.weight_decay),
            "n_grok_steps": int(cfg.training.n_grok_steps),
            "optimizer": {
                "name": str(cfg.training.optimizer), "lr": float(cfg.training.lr),
                "betas": [float(cfg.training.beta1), float(cfg.training.beta2)],
                "eps": float(cfg.training.eps), "state_fresh_for_final_phase": True,
            },
            "measurement": {
                "grok_threshold": float(cfg.training.grok_threshold),
                "mem_threshold": float(cfg.training.mem_threshold),
                "window": int(cfg.training.grok_window),
                "eval_schedule": {
                    "fine_until": int(cfg.training.eval_every.fine_until),
                    "fine_interval": int(cfg.training.eval_every.fine_interval),
                    "coarse_interval": int(cfg.training.eval_every.coarse_interval),
                },
            },
            "early_stop": {
                "enabled": bool(cfg.training.early_stop.enabled),
                "patience_evaluations": int(cfg.training.early_stop.patience),
                "dense_baseline_exempt": True,
            },
            "discovery": discovery_config,
        },
    )
    finish_wandb(wandb_run)

    gm = compute_grokking_metrics(history.to_dict())
    print(
        f"  RESULT: grokked={gm['grokked']} grok_step={gm['grokking_step']} "
        f"gap={gm['grokking_gap']} final_val={gm['final_val_acc']:.3f}"
    )
    return summary_to_aggregate_row(load_summary(grok_dir))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    enable_utf8_stdout()
    device = resolve_device(cfg)
    seeds = resolve_seeds(cfg)
    sparsities = list(cfg.pruning.target_sparsities)
    exp_dir = Path(cfg.results_dir) / cfg.experiment.name

    print(f"\n{'#'*65}")
    print(f"  Exp B - mask discovery, rewind, final training | device={device} seeds={seeds}")
    print(f"  sparsities={sparsities} wd={cfg.training.weight_decay} steps={cfg.training.n_grok_steps:,}")
    print(f"{'#'*65}")

    rows: list[dict] = []
    for seed in seeds:
        for sp in sparsities:
            rows.append(run_single(cfg, sp, "imp", seed, device, exp_dir))
            if cfg.experiment.run_one_shot_ablation:
                rows.append(run_single(cfg, sp, "one_shot", seed, device, exp_dir))

    agg = write_run_aggregate_csv(exp_dir / "aggregate.csv", rows)
    print(f"\n  Aggregate → {agg}")
    print(f"  Per-run summaries/CSV under {exp_dir}/  (plot offline with analysis/plot_exp_b.py)")


if __name__ == "__main__":
    main()
