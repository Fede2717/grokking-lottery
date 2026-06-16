"""
experiments/exp_c_wd_ablation.py
================================
Experiment C — ABLATION: weight-decay × sparsity grid.

Pruning method
--------------
This experiment uses ONE-SHOT MAGNITUDE PRUNING (never IMP). For each grid cell
it does a short pretraining pass to give weights a magnitude ranking, prunes
once to the target sparsity, rewinds to W_0, applies the mask, then runs the
grokking phase.

Weight-decay confound control
------------------------------
Weight decay is the variable under study, so it must not leak into the pruning
step. The short pretraining (which only produces the magnitude ranking) uses
weight_decay = 0 for ALL cells. The grid's weight-decay value is applied ONLY in
the grokking phase. This isolates the effect of weight decay during
generalization from any effect it would have on which weights get pruned.

Mapping
-------
Tests whether weight decay and sparsity are substitutes or complements for
closing the memorization→generalization gap. Plots are produced offline by
analysis/plot_exp_c_heatmap.py from aggregate.csv (no TensorBoard/wandb needed).
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_utils import load_summary, summary_to_aggregate_row, write_aggregate_csv
from src.metrics import compute_grokking_metrics
from src.prune import compute_sparsity, make_empty_masks, one_shot_prune, rewind_weights
from src.runner import (
    build_dataloaders, build_model, build_optimizer, build_trainer, enable_utf8_stdout,
    finish_wandb, maybe_init_wandb, resolve_device, resolve_seeds, seed_everything,
)
from src.train import save_init_checkpoint


def run_condition(
    cfg: DictConfig,
    weight_decay: float,
    target_sparsity: float,
    seed: int,
    device,
    exp_dir: Path,
) -> dict:
    """Run one (weight_decay, sparsity) cell; return one aggregate row."""
    run_dir = exp_dir / f"wd{weight_decay:.0e}_sp{target_sparsity:.2f}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed, device)
    train_loader, val_loader = build_dataloaders(cfg, seed)
    model = build_model(cfg, device)
    init_ckpt = save_init_checkpoint(model, run_dir, cfg.checkpoint.init_filename)
    is_baseline = target_sparsity == 0.0

    # ── One-shot magnitude pruning with the WD confound removed ─────────────
    # Short pretrain uses weight_decay=0 for EVERY cell (ranking only).
    if not is_baseline:
        short_opt = build_optimizer(cfg, model, weight_decay=0.0)
        short_trainer = build_trainer(
            cfg, model, train_loader, val_loader, short_opt, device,
            run_dir / "short_pretrain", compute_fourier=False, logging_backend="none",
        )
        short_trainer.train(
            n_steps=cfg.pruning.imp_steps_per_round * 2,
            save_checkpoints=False, verbose=False,
        )
        masks = one_shot_prune(model, init_ckpt, target_sparsity)
    else:
        masks = make_empty_masks(model)
        rewind_weights(model, init_ckpt, masks)
    actual_sp = compute_sparsity(masks)

    # ── Grokking phase: the grid weight_decay is applied ONLY here ──────────
    grok_dir = run_dir / "grok_phase"
    label = f"wd={weight_decay:.0e}_sp={target_sparsity:.0%}_seed={seed}"
    wandb_run = maybe_init_wandb(cfg, run_name=label, group="exp_c_wd_sparsity")
    grok_opt = build_optimizer(cfg, model, weight_decay=weight_decay)
    trainer = build_trainer(
        cfg, model, train_loader, val_loader, grok_opt, device, grok_dir,
        is_baseline=is_baseline, compute_fourier=False, wandb_run=wandb_run,
    )
    history = trainer.train(
        n_steps=cfg.training.n_grok_steps,
        masks=masks,
        save_checkpoints=False,
        verbose=False,
        config_summary={
            "experiment": "exp_c",
            "method": "one_shot_magnitude",
            "weight_decay": float(weight_decay),
            "target_sparsity": target_sparsity,
            "actual_sparsity": actual_sp,
            "seed": seed,
        },
    )
    finish_wandb(wandb_run)

    gm = compute_grokking_metrics(history.to_dict())
    print(
        f"  wd={weight_decay:.0e} sp={target_sparsity:.0%} → grokked={gm['grokked']} "
        f"grok_step={gm['grokking_step']} val={gm['final_val_acc']:.3f}"
    )
    return summary_to_aggregate_row(load_summary(grok_dir))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    enable_utf8_stdout()
    device = resolve_device(cfg)
    seeds = resolve_seeds(cfg)
    wd_values = list(cfg.experiment.wd_values)
    sp_values = list(cfg.experiment.sparsity_values)
    exp_dir = Path(cfg.results_dir) / cfg.experiment.name

    print(f"\n{'#'*65}")
    print(f"  Exp C — WD × Sparsity (one-shot magnitude pruning)")
    print(f"  {len(wd_values)}×{len(sp_values)} cells | seeds={seeds} | device={device}")
    print(f"{'#'*65}")

    rows: list[dict] = []
    for seed in seeds:
        for sp, wd in product(sp_values, wd_values):
            rows.append(run_condition(cfg, wd, sp, seed, device, exp_dir))

    agg = write_aggregate_csv(exp_dir / "aggregate.csv", rows)
    print(f"\n  Aggregate → {agg}")
    print(f"  Plot offline with analysis/plot_exp_c_heatmap.py")


if __name__ == "__main__":
    main()
