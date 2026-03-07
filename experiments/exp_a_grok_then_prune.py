"""
experiments/exp_a_grok_then_prune.py  (v2)
==========================================
Experiment A — CONTROL: Grok First → Prune → Rewind
====================================================

v2 changes
----------
    Hydra @hydra.main, W&B logging, disk-based checkpointing,
    multi-seed support (via GROK_SEED env var from launcher).

Usage
-----
    python -m experiments.exp_a_grok_then_prune
    python -m experiments.exp_a_grok_then_prune experiment=exp_a
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data    import get_dataloaders
from src.model   import get_model
from src.prune   import (
    make_empty_masks, apply_global_magnitude_pruning,
    compute_sparsity, rewind_weights, apply_masks,
)
from src.train   import (
    Trainer, make_optimizer, TrainingHistory,
    save_init_checkpoint, save_checkpoint,
    load_checkpoint_from_disk, init_wandb,
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
# Phase 1: Dense training to grokking
# ===========================================================================

def train_dense_to_grokking(
    cfg:     DictConfig,
    device:  torch.device,
    seed:    int,
    run_dir: Path,
) -> tuple[object, object, object, TrainingHistory, Path]:
    """
    Train dense model to grokking.  Save W_0 to disk BEFORE training.

    Returns
    -------
    (model, train_loader, val_loader, history, init_ckpt_path)
    """
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
        dropout=cfg.model.dropout,
    ).to(device)

    # ── Strict init checkpoint ─────────────────────────────────────────────
    init_ckpt_path = save_init_checkpoint(model, run_dir, "init_weights.pt")
    print(f"  [Init] W_0 saved → {init_ckpt_path}")

    wandb_cfg = OmegaConf.to_container(cfg.wandb, resolve=True)
    init_wandb(
        cfg_dict=OmegaConf.to_container(cfg, resolve=True),
        run_name=f"exp_a_dense_seed{seed}",
        group="exp_a_control",
    )

    optimizer = make_optimizer(
        model, lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(cfg.training.beta1, cfg.training.beta2),
    )
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, device=device,
        run_dir=run_dir / "dense_phase",
        p=cfg.dataset.p,
        log_every=cfg.training.log_every,
        grok_threshold=cfg.training.grok_threshold,
        mem_threshold=cfg.training.mem_threshold,
        grok_window=cfg.training.grok_window,
        compute_fourier=cfg.pruning.compute_fourier,
        use_amp=cfg.use_amp,
        wandb_cfg=wandb_cfg,
    )

    print(f"\n  Phase 1: Dense training ({cfg.training.n_grok_steps:,} steps)")
    history = trainer.train(n_steps=cfg.training.n_grok_steps, save_checkpoints=True)

    if not history.grokked:
        print("  WARNING: Dense model did not grok within step budget.")

    if _WANDB_AVAILABLE and wandb_cfg.get("enabled", False):
        _wandb.finish()

    return model, train_loader, val_loader, history, init_ckpt_path


# ===========================================================================
# Phase 2: Prune + evaluate (no retraining)
# ===========================================================================

@torch.no_grad()
def evaluate_pruned_accuracy(
    model:        object,
    val_loader:   object,
    grok_ckpt_path: str,
    target_sparsity: float,
    device:       torch.device,
) -> tuple[float, float]:
    """
    Load post-grokking weights, prune, evaluate WITHOUT retraining.
    Returns (val_acc, actual_sparsity).
    """
    ckpt = load_checkpoint_from_disk(grok_ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    masks = make_empty_masks(model)
    masks = apply_global_magnitude_pruning(model, masks, target_sparsity)
    apply_masks(model, masks)
    actual_sp = compute_sparsity(masks)

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss    += criterion(logits, y).item() * y.size(0)
        total_correct += (logits.argmax(-1) == y).sum().item()
        total_n       += y.size(0)
    model.train()

    return total_correct / total_n, actual_sp


# ===========================================================================
# Phase 3: Prune + Rewind + Retrain
# ===========================================================================

def prune_rewind_retrain(
    model:           object,
    train_loader:    object,
    val_loader:      object,
    cfg:             DictConfig,
    device:          torch.device,
    target_sparsity: float,
    grok_ckpt_path:  str,        # source for pruning magnitude ranking
    rewind_ckpt_path:str,        # source for weight rewinding
    rewind_label:    str,
    seed:            int,
    run_dir:         Path,
) -> TrainingHistory:
    """Load grokked weights to rank magnitudes, prune, rewind, retrain."""
    ckpt = load_checkpoint_from_disk(grok_ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    masks = make_empty_masks(model)
    masks = apply_global_magnitude_pruning(model, masks, target_sparsity)
    rewind_weights(model, rewind_ckpt_path, masks)

    actual_sp = compute_sparsity(masks)
    print(f"\n  [{rewind_label} sp={actual_sp:.1%}]  Retraining ({cfg.training.n_grok_steps:,} steps)...")

    wandb_cfg = OmegaConf.to_container(cfg.wandb, resolve=True)
    run_name  = f"exp_a_{rewind_label}_sp{target_sparsity:.0%}_seed{seed}"
    init_wandb(
        cfg_dict=OmegaConf.to_container(cfg, resolve=True),
        run_name=run_name, group="exp_a_rewind",
    )

    optimizer = make_optimizer(
        model, lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(cfg.training.beta1, cfg.training.beta2),
    )
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, device=device,
        run_dir=run_dir / f"{rewind_label}_sp{target_sparsity:.2f}",
        p=cfg.dataset.p,
        log_every=cfg.training.log_every,
        grok_threshold=cfg.training.grok_threshold,
        mem_threshold=cfg.training.mem_threshold,
        grok_window=cfg.training.grok_window,
        compute_fourier=cfg.pruning.compute_fourier,
        use_amp=cfg.use_amp,
        wandb_cfg=wandb_cfg,
    )

    history = trainer.train(n_steps=cfg.training.n_grok_steps, masks=masks)
    history.config_summary.update({
        "target_sparsity": target_sparsity,
        "actual_sparsity": actual_sp,
        "rewind_label"   : rewind_label,
        "seed"           : seed,
    })

    gm = compute_grokking_metrics(history.to_dict())
    print(f"    grokked={gm['grokked']}  S_G={gm['grokking_step']:,}  val={gm['final_val_acc']:.3f}")

    if _WANDB_AVAILABLE and wandb_cfg.get("enabled", False):
        _wandb.finish()

    return history


# ===========================================================================
# Plotting
# ===========================================================================

def plot_rewind_comparison(
    results:      dict,
    probe_sp:     list[float],
    dense_sg:     int,
    n_grok_steps: int,
    output_path:  str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Effect of Rewind Strategy on Grokking Speed\n(lower = faster grokking)",
                 fontsize=12, fontweight="bold")

    for key, label, color, marker in [
        ("W_init", "Rewind to W_0 (step 0)",     "#1f77b4", "o"),
        ("W_mem",  "Rewind to W_mem (memorised)", "#ff7f0e", "s"),
    ]:
        sp_arr, sg_arr = [], []
        for sp in probe_sp:
            sp_key = f"{sp:.2f}"
            recs = results.get(key, {}).get(sp_key, [])
            sg_vals = [compute_grokking_metrics(h.to_dict())["grokking_step"]
                       for h in recs
                       if compute_grokking_metrics(h.to_dict())["grokked"]]
            sp_arr.append(sp * 100)
            sg_arr.append(np.mean(sg_vals) if sg_vals else n_grok_steps * 1.05)
        ax.plot(sp_arr, sg_arr, marker=marker, color=color, label=label,
                linewidth=2, markersize=8)

    ax.axhline(dense_sg, color="gray", linestyle="--",
               label=f"Dense baseline (S_G={dense_sg:,})", lw=1.5)
    ax.set_xlabel("Sparsity (%)", fontsize=11)
    ax.set_ylabel("Steps to Grok (S_G)", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_post_grok_accuracy(
    sparsities:  list[float],
    accuracies:  list[float],
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(
        "Generalisation Circuit Survival After Pruning\n"
        "(Validation accuracy of pruned post-grokking net — no retraining)",
        fontsize=11, fontweight="bold",
    )
    colors = ["#4CAF50" if a >= 0.9 else "#F44336" for a in accuracies]
    bars   = ax.bar([f"{s:.0%}" for s in sparsities], accuracies,
                    color=colors, alpha=0.8, edgecolor="k")
    ax.axhline(0.95, color="k", linestyle="--", lw=1, label="Grok threshold")
    ax.set_xlabel("Sparsity", fontsize=11)
    ax.set_ylabel("Val Accuracy (no retraining)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=9)
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
    base_dir = Path(cfg.results_dir) / "exp_a" / f"seed_{seed}"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*65}")
    print(f"  Exp A — Grok → Prune  |  device={device}  seed={seed}")
    print(f"{'#'*65}\n")

    # ── Phase 1: dense grokking ────────────────────────────────────────────
    model, train_loader, val_loader, dense_hist, init_ckpt_path = \
        train_dense_to_grokking(cfg, device, seed, base_dir / "dense")

    dense_gm  = compute_grokking_metrics(dense_hist.to_dict())
    dense_sg  = dense_gm["grokking_step"]
    grok_ckpt = dense_hist.checkpoint_paths.get("grokking",
                    dense_hist.checkpoint_paths.get("final", str(init_ckpt_path)))
    mem_ckpt  = dense_hist.checkpoint_paths.get("memorization", str(init_ckpt_path))

    probe_sparsities = [s for s in cfg.pruning.target_sparsities if s > 0.0]

    # ── Post-grokking accuracy (no retraining) ────────────────────────────
    post_grok_accs = []
    for sp in probe_sparsities:
        acc, actual_sp = evaluate_pruned_accuracy(
            model, val_loader, grok_ckpt, sp, device,
        )
        post_grok_accs.append(acc)
        print(f"  Post-grok pruned ({sp:.0%}) val_acc={acc:.3f}  actual_sp={actual_sp:.2%}")

    # ── Phase 2: prune + rewind + retrain ─────────────────────────────────
    results: dict[str, dict[str, list[TrainingHistory]]] = {
        "W_init": {}, "W_mem": {}
    }

    for sp in probe_sparsities:
        sp_key = f"{sp:.2f}"
        for rewind_label, rewind_src in [
            ("W_init", str(init_ckpt_path)),
            ("W_mem",  mem_ckpt),
        ]:
            torch.manual_seed(seed + int(sp * 100))
            model_run = get_model(
                vocab_size=cfg.dataset.p + 2, n_classes=cfg.dataset.p,
                d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
                n_layers=cfg.model.n_layers, d_ff=cfg.model.d_ff,
            ).to(device)

            h = prune_rewind_retrain(
                model=model_run, train_loader=train_loader,
                val_loader=val_loader, cfg=cfg, device=device,
                target_sparsity=sp,
                grok_ckpt_path=grok_ckpt,
                rewind_ckpt_path=rewind_src,
                rewind_label=rewind_label, seed=seed,
                run_dir=base_dir / "rewind_runs",
            )
            results[rewind_label].setdefault(sp_key, []).append(h)

    # ── Save & plot ────────────────────────────────────────────────────────
    save_data = {
        "dense_summary"  : dense_gm,
        "post_grok_accs" : dict(zip([f"{s:.2f}" for s in probe_sparsities], post_grok_accs)),
        "results_W_init" : {k: [h.to_dict() for h in hs]
                            for k, hs in results["W_init"].items()},
        "results_W_mem"  : {k: [h.to_dict() for h in hs]
                            for k, hs in results["W_mem"].items()},
    }
    results_path = Path(cfg.results_dir) / "exp_a_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)

    fig_dir = Path(cfg.results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_rewind_comparison(
        results, probe_sparsities, dense_sg, cfg.training.n_grok_steps,
        str(fig_dir / "exp_a_rewind_comparison.png"),
    )
    plot_post_grok_accuracy(
        probe_sparsities, post_grok_accs,
        str(fig_dir / "exp_a_post_grok_accuracy.png"),
    )

    print(f"\n  Experiment A complete.  Dense S_G={dense_sg:,}")


if __name__ == "__main__":
    main()
