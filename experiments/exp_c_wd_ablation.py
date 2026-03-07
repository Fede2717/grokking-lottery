"""
experiments/exp_c_wd_ablation.py  (v2)
========================================
Experiment C — ABLATION: Weight Decay × Sparsity Grid
======================================================

v2: Hydra @hydra.main, W&B logging, disk checkpointing.

Usage
-----
    python -m experiments.exp_c_wd_ablation experiment=exp_c
"""

from __future__ import annotations

import json
import os
import sys
from itertools import product
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data    import get_dataloaders
from src.model   import get_model
from src.prune   import (
    make_empty_masks, one_shot_prune, compute_sparsity, rewind_weights,
)
from src.train   import (
    Trainer, make_optimizer, TrainingHistory,
    save_init_checkpoint, init_wandb,
)
from src.metrics import compute_grokking_metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ===========================================================================
# Single grid condition
# ===========================================================================

def run_condition(
    cfg:             DictConfig,
    weight_decay:    float,
    target_sparsity: float,
    device:          torch.device,
    seed:            int,
    run_dir:         Path,
) -> dict:
    """Run one (λ, sparsity) cell and return a JSON-serialisable dict."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        p=cfg.dataset.p, operation=cfg.dataset.operation,
        train_frac=cfg.dataset.train_frac, batch_size=cfg.dataset.batch_size,
        seed=cfg.seed,
    )
    model = get_model(
        vocab_size=cfg.dataset.p + 2, n_classes=cfg.dataset.p,
        d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers, d_ff=cfg.model.d_ff,
    ).to(device)

    # ── Strict init checkpoint ─────────────────────────────────────────────
    init_ckpt_path = save_init_checkpoint(model, run_dir, "init_weights.pt")

    wandb_cfg = OmegaConf.to_container(cfg.wandb, resolve=True)
    run_name  = f"exp_c_wd{weight_decay:.0e}_sp{target_sparsity:.0%}_seed{seed}"
    init_wandb(
        cfg_dict={**OmegaConf.to_container(cfg, resolve=True),
                  "override_weight_decay": weight_decay,
                  "override_sparsity": target_sparsity},
        run_name=run_name, group="exp_c_wd_sparsity",
    )

    # ── One-shot pruning (fast for grid search) ────────────────────────────
    if target_sparsity > 0.0:
        opt_short = make_optimizer(model, lr=cfg.training.lr,
                                   weight_decay=weight_decay,
                                   betas=(cfg.training.beta1, cfg.training.beta2))
        trainer_short = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=opt_short, device=device,
            run_dir=run_dir / "short_pretrain", p=cfg.dataset.p,
            log_every=999999, compute_fourier=False,
            use_amp=cfg.use_amp, wandb_cfg={"enabled": False},
        )
        trainer_short.train(
            n_steps=cfg.pruning.imp_steps_per_round * 2,
            save_checkpoints=False, verbose=False,
        )
        masks = one_shot_prune(model, init_ckpt_path, target_sparsity)
    else:
        masks = make_empty_masks(model)
        rewind_weights(model, init_ckpt_path, masks)

    actual_sp = compute_sparsity(masks)

    # ── Grokking phase (with overridden weight_decay) ──────────────────────
    opt = make_optimizer(model, lr=cfg.training.lr,
                         weight_decay=weight_decay,
                         betas=(cfg.training.beta1, cfg.training.beta2))
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=opt, device=device,
        run_dir=run_dir / "grok_phase", p=cfg.dataset.p,
        log_every=cfg.training.log_every,
        grok_threshold=cfg.training.grok_threshold,
        mem_threshold=cfg.training.mem_threshold,
        grok_window=cfg.training.grok_window,
        compute_fourier=False,   # skip Fourier in grid — saves time
        use_amp=cfg.use_amp,
        wandb_cfg=wandb_cfg,
    )
    history = trainer.train(
        n_steps=cfg.training.n_grok_steps, masks=masks,
        save_checkpoints=False, verbose=False,
    )

    gm = compute_grokking_metrics(history.to_dict())
    result = {
        "weight_decay"    : weight_decay,
        "target_sparsity" : target_sparsity,
        "actual_sparsity" : actual_sp,
        "seed"            : seed,
        **{k: gm[k] for k in ["grokked","grokking_step","grokking_gap",
                                "final_val_acc","final_train_acc","memorization_step"]},
    }

    print(f"  λ={weight_decay:.0e}  sp={target_sparsity:.0%}  → "
          f"grokked={result['grokked']}  S_G={result['grokking_step']:,}  "
          f"val={result['final_val_acc']:.3f}")

    if _WANDB_AVAILABLE and wandb_cfg.get("enabled", False):
        _wandb.summary.update({k: result[k] for k in ["grokked","grokking_step","final_val_acc"]})
        _wandb.finish()

    return result


# ===========================================================================
# Plots
# ===========================================================================

def plot_heatmap(
    grid:        np.ndarray,
    wd_values:   list[float],
    sp_values:   list[float],
    title:       str,
    output_path: str,
    cmap:        str = "RdYlGn_r",
    fmt:         str = ".0f",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title, fontsize=12, fontweight="bold")
    im = ax.imshow(grid, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(wd_values)))
    ax.set_xticklabels([f"λ={wd:.0e}" for wd in wd_values], fontsize=9)
    ax.set_yticks(range(len(sp_values)))
    ax.set_yticklabels([f"{s:.0%}" for s in sp_values], fontsize=9)
    ax.set_xlabel("Weight Decay (λ)", fontsize=10)
    ax.set_ylabel("Sparsity", fontsize=10)
    for i in range(len(sp_values)):
        for j in range(len(wd_values)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, format(v, fmt) if v >= 0 else "DNF",
                        ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > np.nanmax(grid) * 0.6 else "black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_wd_sparsity_lines(
    results:     list[dict],
    wd_values:   list[float],
    sp_values:   list[float],
    n_steps:     int,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Weight Decay × Sparsity Interaction\n"
                 "Are they substitutes or complements for grokking?",
                 fontsize=12, fontweight="bold")
    colors = ["#E53935","#FB8C00","#43A047","#1E88E5"]

    for ax, (mk, ml, log_y) in zip(axes, [
        ("grokking_step", "Steps to Grok (S_G)", True),
        ("final_val_acc", "Final Validation Accuracy", False),
    ]):
        for i, wd in enumerate(wd_values):
            vals = []
            for sp in sp_values:
                matches = [r for r in results
                           if abs(r["weight_decay"] - wd) < 1e-9
                           and abs(r["target_sparsity"] - sp) < 0.01]
                if matches:
                    r = matches[0]
                    v = r[mk] if (mk != "grokking_step" or r["grokked"]) \
                        else n_steps * 1.05
                else:
                    v = float("nan")
                vals.append(v)

            ax.plot([s * 100 for s in sp_values], vals, marker="o",
                    color=colors[i % len(colors)], label=f"λ={wd:.0e}",
                    linewidth=2, markersize=8)

        ax.set_xlabel("Sparsity (%)", fontsize=11)
        ax.set_ylabel(ml, fontsize=11)
        if log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ===========================================================================
# Hydra main
# ===========================================================================

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    seed = int(os.environ.get("GROK_SEED", cfg.seed))

    exp_cfg = cfg.experiment
    wd_values = list(exp_cfg.wd_values)
    sp_values = list(exp_cfg.sparsity_values)
    total     = len(wd_values) * len(sp_values)

    print(f"\n{'#'*65}")
    print(f"  Exp C — WD × Sparsity  |  {len(wd_values)}×{len(sp_values)}={total} conditions")
    print(f"  device={device}  seed={seed}")
    print(f"{'#'*65}\n")

    base_dir = Path(cfg.results_dir) / "exp_c" / f"seed_{seed}"
    all_results: list[dict] = []
    seed_offset = 0

    for sp, wd in product(sp_values, wd_values):
        run_dir = base_dir / f"wd{wd:.0e}_sp{sp:.2f}"
        result  = run_condition(cfg, wd, sp, device, seed + seed_offset, run_dir)
        all_results.append(result)
        seed_offset += 1

    # ── Heatmaps ──────────────────────────────────────────────────────────
    sg_grid  = np.full((len(sp_values), len(wd_values)), np.nan)
    acc_grid = np.full((len(sp_values), len(wd_values)), np.nan)

    for r in all_results:
        try:
            i = sp_values.index(r["target_sparsity"])
            j = wd_values.index(r["weight_decay"])
        except ValueError:
            continue
        sg_grid[i, j]  = r["grokking_step"] if r["grokked"] else -1
        acc_grid[i, j] = r["final_val_acc"]

    results_path = Path(cfg.results_dir) / "exp_c_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    fig_dir = Path(cfg.results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmap(sg_grid, wd_values, sp_values,
                 "Steps to Grok (S_G)\n(darker = faster grokking)",
                 str(fig_dir / "exp_c_heatmap_sg.png"), cmap="RdYlGn_r")
    plot_heatmap(acc_grid, wd_values, sp_values,
                 "Final Validation Accuracy",
                 str(fig_dir / "exp_c_heatmap_val_acc.png"),
                 cmap="RdYlGn", fmt=".2f")
    plot_wd_sparsity_lines(all_results, wd_values, sp_values,
                           cfg.training.n_grok_steps,
                           str(fig_dir / "exp_c_wd_sparsity_lines.png"))

    print(f"\n  Experiment C complete.")


if __name__ == "__main__":
    main()
