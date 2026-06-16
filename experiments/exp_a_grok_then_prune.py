"""
experiments/exp_a_grok_then_prune.py
====================================
Experiment A — CONTROL: grok first, then prune + rewind (the faithful
"grokked ticket").

What it does
------------
    1. Train a DENSE model to full grokking; checkpoints are saved at
       initialization (W_0), memorization (W_mem) and grokking (W_grok).
    2. Circuit survival: prune W_grok at each sparsity and evaluate WITHOUT
       retraining (does the generalizing circuit survive pruning?).
    3. Grokked ticket: rank magnitudes on W_grok to build the mask (the
       post-generalization ticket), rewind to W_0 vs W_mem, and retrain. The
       grokked mask is saved per sparsity for the mask-overlap analysis.
    4. Masks are also extracted at the memorization and grokking stages and
       saved, so analysis/mask_overlap.py can measure how the magnitude mask
       changes from memorization → generalization.

Mapping to Minegishi et al. (TMLR 2025)
---------------------------------------
This is the canonical direction: a subnetwork extracted AFTER generalization.
Exp B (prune before grokking) is the extension. Plots are produced offline by
analysis/plot_exp_a.py from the CSV / summary.json / aggregate.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_utils import load_summary, summary_to_aggregate_row, write_aggregate_csv
from src.metrics import compute_grokking_metrics
from src.prune import (
    apply_global_magnitude_pruning, apply_masks, compute_sparsity,
    make_empty_masks, rewind_weights,
)
from src.runner import (
    build_dataloaders, build_model, build_optimizer, build_trainer, enable_utf8_stdout,
    finish_wandb, maybe_init_wandb, resolve_device, resolve_seeds, seed_everything,
)
from src.train import load_checkpoint_from_disk, save_init_checkpoint, save_masks


def _extract_mask(model, ckpt_path, target_sparsity, device) -> dict:
    """Load weights from a checkpoint and return a fresh magnitude mask at sp."""
    ckpt = load_checkpoint_from_disk(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    masks = make_empty_masks(model)
    return apply_global_magnitude_pruning(model, masks, target_sparsity)


@torch.no_grad()
def _val_accuracy(model, val_loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total


def run_seed(cfg: DictConfig, seed: int, device, exp_dir: Path) -> list[dict]:
    """Full Exp A pipeline for one seed; returns aggregate rows."""
    base = exp_dir / f"seed_{seed}"
    base.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # ── Phase 1: dense → grokking ───────────────────────────────────────────
    seed_everything(seed, device)
    train_loader, val_loader = build_dataloaders(cfg, seed)
    model = build_model(cfg, device)
    init_ckpt = save_init_checkpoint(model, base / "dense", cfg.checkpoint.init_filename)

    print(f"\n{'#'*65}\n  Exp A seed={seed}: dense → grokking ({cfg.training.n_grok_steps:,} steps)\n{'#'*65}")
    wandb_run = maybe_init_wandb(cfg, run_name=f"exp_a_dense_seed{seed}", group="exp_a_dense")
    dense_opt = build_optimizer(cfg, model)
    dense_trainer = build_trainer(
        cfg, model, train_loader, val_loader, dense_opt, device, base / "dense",
        is_baseline=True, wandb_run=wandb_run,
    )
    dense_hist = dense_trainer.train(
        n_steps=cfg.training.n_grok_steps, save_checkpoints=True,
        config_summary={"experiment": "exp_a", "stage": "dense", "seed": seed,
                        "target_sparsity": 0.0, "actual_sparsity": 0.0,
                        "weight_decay": float(cfg.training.weight_decay)},
    )
    finish_wandb(wandb_run)
    dense_gm = compute_grokking_metrics(dense_hist.to_dict())
    print(f"  dense grokked={dense_gm['grokked']} grok_step={dense_gm['grokking_step']} gap={dense_gm['grokking_gap']}")

    rows.append(summary_to_aggregate_row(
        load_summary(base / "dense"), extra={"rewind": "none", "phase": "dense"}
    ))

    init_path = str(init_ckpt)
    grok_path = dense_hist.checkpoint_paths.get("grokking", dense_hist.checkpoint_paths.get("final", init_path))
    mem_path = dense_hist.checkpoint_paths.get("memorization", init_path)
    rewind_sources = {"W_init": init_path, "W_mem": mem_path}

    probe_sparsities = [s for s in cfg.pruning.target_sparsities if s > 0.0]

    for sp in probe_sparsities:
        masks_dir = base / "masks" / f"sp_{sp:.2f}"
        masks_dir.mkdir(parents=True, exist_ok=True)

        # ── Save masks extracted at each stage (for mask_overlap analysis) ──
        for stage, ckpt in [("init", init_path), ("mem", mem_path), ("grok", grok_path)]:
            stage_mask = _extract_mask(model, ckpt, sp, device)
            save_masks(stage_mask, masks_dir / f"mask_{stage}.pt")
        # The grokked ticket (canonical) = mask ranked on W_grok.
        grok_mask = _extract_mask(model, grok_path, sp, device)
        save_masks(grok_mask, masks_dir / "grokked_mask.pt")

        # ── Circuit survival: prune W_grok, evaluate WITHOUT retraining ─────
        ckpt = load_checkpoint_from_disk(grok_path)
        model.load_state_dict(ckpt["state_dict"]); model.to(device)
        apply_masks(model, grok_mask)
        surv_acc = _val_accuracy(model, val_loader, device)
        print(f"  [sp={sp:.0%}] circuit-survival val_acc (no retrain) = {surv_acc:.3f}")

        # ── Grokked ticket: rewind to W_init / W_mem, retrain ───────────────
        for rewind_label, rewind_src in rewind_sources.items():
            seed_everything(seed + int(sp * 100), device)
            ckpt = load_checkpoint_from_disk(grok_path)
            model.load_state_dict(ckpt["state_dict"]); model.to(device)
            masks = make_empty_masks(model)
            masks = apply_global_magnitude_pruning(model, masks, sp)
            rewind_weights(model, rewind_src, masks)
            actual_sp = compute_sparsity(masks)

            run_dir = base / "rewind" / f"{rewind_label}_sp{sp:.2f}"
            opt = build_optimizer(cfg, model)
            trainer = build_trainer(
                cfg, model, train_loader, val_loader, opt, device, run_dir,
                is_baseline=False,
            )
            hist = trainer.train(
                n_steps=cfg.training.n_grok_steps, masks=masks, save_checkpoints=True,
                verbose=False,
                config_summary={
                    "experiment": "exp_a", "phase": "rewind", "rewind": rewind_label,
                    "target_sparsity": sp, "actual_sparsity": actual_sp, "seed": seed,
                    "circuit_survival_acc": surv_acc,
                    "weight_decay": float(cfg.training.weight_decay),
                },
            )
            gm = compute_grokking_metrics(hist.to_dict())
            print(f"  [sp={sp:.0%} {rewind_label}] grok_step={gm['grokking_step']} gap={gm['grokking_gap']}")
            rows.append(summary_to_aggregate_row(load_summary(run_dir)))

    return rows


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    enable_utf8_stdout()
    device = resolve_device(cfg)
    seeds = resolve_seeds(cfg)
    exp_dir = Path(cfg.results_dir) / cfg.experiment.name

    rows: list[dict] = []
    for seed in seeds:
        rows.extend(run_seed(cfg, seed, device, exp_dir))

    agg = write_aggregate_csv(exp_dir / "aggregate.csv", rows)
    print(f"\n  Aggregate → {agg}")
    print(f"  Plot offline with analysis/plot_exp_a.py and analysis/mask_overlap.py")


if __name__ == "__main__":
    main()
