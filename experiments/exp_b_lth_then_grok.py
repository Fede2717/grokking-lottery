"""
experiments/exp_b_lth_then_grok.py  (v2)
=========================================
Experiment B — PRIMARY: Lottery Ticket Hypothesis First, Then Grok
===================================================================

v2 changes
----------
    • Hydra @hydra.main — all hyperparameters via configs/config.yaml.
    • W&B — every metric logged per step; Fourier + Hessian on schedule.
    • Multi-seed — runs cfg.num_seeds independent seeds per sparsity level.
      Mean ± 95 % CI plotted on all charts.
    • Disk rewinding — IMP calls rewind_weights() from the init .pt file.
    • Strict init save — save_init_checkpoint() called before any gradient.
    • Parallel seeds are handled by the LAUNCHER (run_parallel_seeds.py);
      this script runs one seed per invocation (GPU-0 or GPU-1 as set by
      CUDA_VISIBLE_DEVICES in the environment).

Usage
-----
    # Single run with default config
    python -m experiments.exp_b_lth_then_grok

    # Override sparsities and weight decay
    python -m experiments.exp_b_lth_then_grok \\
        "pruning.target_sparsities=[0.0,0.5,0.9]" \\
        training.weight_decay=5e-3

    # Parallel multi-seed (launched by run_parallel_seeds.py)
    CUDA_VISIBLE_DEVICES=0 python -m experiments.exp_b_lth_then_grok seed=1
    CUDA_VISIBLE_DEVICES=1 python -m experiments.exp_b_lth_then_grok seed=2
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

# Ensure src/ is importable when run as __main__
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data    import get_dataloaders
from src.model   import get_model
from src.prune   import (
    make_empty_masks, run_imp, one_shot_prune,
    compute_sparsity, rewind_weights,
)
from src.train   import (
    Trainer, make_optimizer, TrainingHistory,
    save_init_checkpoint, init_wandb,
)
from src.metrics import compute_grokking_metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2"]


# ===========================================================================
# Single-sparsity, single-seed runner
# ===========================================================================

def run_single_sparsity(
    cfg:             DictConfig,
    target_sparsity: float,
    device:          torch.device,
    pruning_method:  str,   # "imp" | "one_shot"
    seed:            int,
    run_dir:         Path,
) -> TrainingHistory:
    """
    Full pipeline for one (sparsity, method, seed) triple.

    1. Fresh model — save W_0 to disk (init_weights.pt).
    2. IMP or one-shot pruning phase.
    3. Rewind to W_0 (from disk).
    4. Long grokking phase with W&B logging.

    Returns the grokking-phase TrainingHistory.
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    init_ckpt_path = run_dir / "init_weights.pt"

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        p          = cfg.dataset.p,
        operation  = cfg.dataset.operation,
        train_frac = cfg.dataset.train_frac,
        batch_size = cfg.dataset.batch_size,
        seed       = cfg.seed,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = get_model(
        vocab_size = cfg.dataset.p + 2,
        n_classes  = cfg.dataset.p,
        d_model    = cfg.model.d_model,
        n_heads    = cfg.model.n_heads,
        n_layers   = cfg.model.n_layers,
        d_ff       = cfg.model.d_ff,
        dropout    = cfg.model.dropout,
    ).to(device)

    # ── STRICT INIT CHECKPOINT (Requirement 5) ────────────────────────────
    # MUST happen before any optimizer.step() is ever called.
    saved_init = save_init_checkpoint(model, run_dir, filename="init_weights.pt")
    print(f"  [Init] W_0 saved → {saved_init}")

    def _fresh_opt():
        return make_optimizer(
            model,
            lr           = cfg.training.lr,
            weight_decay = cfg.training.weight_decay,
            betas        = (cfg.training.beta1, cfg.training.beta2),
        )

    wandb_cfg = OmegaConf.to_container(cfg.wandb, resolve=True)

    label = f"sp={target_sparsity:.0%}_method={pruning_method}_seed={seed}"
    print(f"\n{'='*65}\n  {label}\n{'='*65}")

    # ── W&B init (run-level) ──────────────────────────────────────────────
    wb_enabled = init_wandb(
        cfg_dict = OmegaConf.to_container(cfg, resolve=True),
        run_name = label,
        group    = f"exp_b_{pruning_method}",
    )

    # ── Pruning phase ──────────────────────────────────────────────────────
    if target_sparsity == 0.0:
        masks = make_empty_masks(model)

    elif pruning_method == "imp":
        opt_imp = _fresh_opt()
        trainer_imp = Trainer(
            model          = model,
            train_loader   = train_loader,
            val_loader     = val_loader,
            optimizer      = opt_imp,
            device         = device,
            run_dir        = run_dir / "imp_phase",
            p              = cfg.dataset.p,
            log_every      = cfg.training.log_every * 5,
            grok_threshold = cfg.training.grok_threshold,
            mem_threshold  = cfg.training.mem_threshold,
            compute_fourier= False,
            use_amp        = cfg.use_amp,
            wandb_cfg      = {"enabled": False},   # no W&B during IMP phase
        )
        imp_result = run_imp(
            model                = model,
            init_ckpt_path       = init_ckpt_path,
            trainer              = trainer_imp,
            target_sparsity      = target_sparsity,
            prune_rate_per_round = cfg.pruning.prune_rate_per_round,
            steps_per_round      = cfg.pruning.imp_steps_per_round,
        )
        masks = imp_result.final_masks

    elif pruning_method == "one_shot":
        opt_short = _fresh_opt()
        trainer_short = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=opt_short, device=device,
            run_dir=run_dir / "oneshot_pretrain", p=cfg.dataset.p,
            log_every=999999, compute_fourier=False,
            use_amp=cfg.use_amp, wandb_cfg={"enabled": False},
        )
        trainer_short.train(
            n_steps=cfg.pruning.imp_steps_per_round * 3,
            save_checkpoints=False, verbose=False,
        )
        masks = one_shot_prune(model, init_ckpt_path, target_sparsity)

    else:
        raise ValueError(f"Unknown pruning_method: {pruning_method!r}")

    # Rewind to W_0 with final mask (disk-based)
    rewind_weights(model, init_ckpt_path, masks)
    actual_sp = compute_sparsity(masks)
    print(f"  Actual sparsity: {actual_sp:.2%}")

    if wb_enabled:
        _wandb.log({"pruning/actual_sparsity": actual_sp, "pruning/target_sparsity": target_sparsity})

    # ── Grokking phase ─────────────────────────────────────────────────────
    opt_grok = _fresh_opt()
    grok_trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        optimizer      = opt_grok,
        device         = device,
        run_dir        = run_dir / "grok_phase",
        p              = cfg.dataset.p,
        log_every      = cfg.training.log_every,
        grok_threshold = cfg.training.grok_threshold,
        mem_threshold  = cfg.training.mem_threshold,
        grok_window    = cfg.training.grok_window,
        compute_fourier= cfg.pruning.compute_fourier,
        use_amp        = cfg.use_amp,
        wandb_cfg      = wandb_cfg,
        max_grad_norm  = cfg.training.max_grad_norm,
    )

    print(f"\n  Grokking phase: {cfg.training.n_grok_steps:,} steps")
    history = grok_trainer.train(
        n_steps=cfg.training.n_grok_steps, masks=masks, save_checkpoints=True,
    )

    history.config_summary.update({
        "target_sparsity": target_sparsity,
        "actual_sparsity": actual_sp,
        "pruning_method" : pruning_method,
        "seed"           : seed,
        "weight_decay"   : cfg.training.weight_decay,
        "n_grok_steps"   : cfg.training.n_grok_steps,
    })

    gm = compute_grokking_metrics(history.to_dict())
    print(
        f"\n  RESULT: grokked={gm['grokked']}  "
        f"S_G={gm['grokking_step']:,}  gap={gm['grokking_gap']:,}  "
        f"final_val={gm['final_val_acc']:.3f}"
    )

    if wb_enabled:
        _wandb.summary.update({
            "grokked"       : gm["grokked"],
            "S_G"           : gm["grokking_step"],
            "grokking_gap"  : gm["grokking_gap"],
            "final_val_acc" : gm["final_val_acc"],
            "actual_sparsity": actual_sp,
        })
        _wandb.finish()

    return history


# ===========================================================================
# Multi-seed runner (per sparsity)
# ===========================================================================

def run_sparsity_multiseed(
    cfg:             DictConfig,
    target_sparsity: float,
    device:          torch.device,
    pruning_method:  str,
    seeds:           list[int],
    base_dir:        Path,
) -> list[TrainingHistory]:
    """
    Run one sparsity level across multiple seeds.
    Returns a list of TrainingHistory objects (one per seed).
    """
    histories = []
    for seed in seeds:
        run_dir = base_dir / pruning_method / f"sp_{target_sparsity:.2f}" / f"seed_{seed}"
        h = run_single_sparsity(
            cfg=cfg, target_sparsity=target_sparsity, device=device,
            pruning_method=pruning_method, seed=seed, run_dir=run_dir,
        )
        histories.append(h)
    return histories


# ===========================================================================
# Plotting  (mean ± CI across seeds)
# ===========================================================================

def _align_histories(
    histories: list[TrainingHistory],
    field:     str,
    n_grok_steps: int,
    log_every:    int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align per-seed histories to a common step grid and return
    (steps, mean, std) arrays.  Shorter histories are right-padded
    with their last value.
    """
    max_len = n_grok_steps // log_every + 1
    grid    = np.arange(max_len) * log_every
    matrix  = []

    for h in histories:
        values = np.array(getattr(h, field), dtype=float)
        steps  = np.array(h.steps, dtype=int)
        # Interpolate onto grid
        aligned = np.interp(grid, steps, values,
                            left=values[0] if len(values) else 0.0,
                            right=values[-1] if len(values) else 0.0)
        matrix.append(aligned)

    mat  = np.stack(matrix, axis=0)    # (n_seeds, T)
    mean = mat.mean(axis=0)
    std  = mat.std(axis=0)
    return grid, mean, std


def _ci95(std: np.ndarray, n: int) -> np.ndarray:
    """95% CI half-width: 1.96 × std / sqrt(n)."""
    return 1.96 * std / (n ** 0.5)


def plot_grokking_curves_ci(
    all_histories: dict[str, list[TrainingHistory]],
    sparsity_levels: list[float],
    cfg: DictConfig,
    output_path: str,
    method: str = "IMP",
) -> None:
    """
    4-panel grokking curves with mean ± 95% CI shading across seeds.
    Primary LinkedIn / CV visualisation.
    """
    n_grok = cfg.training.n_grok_steps
    log_e  = cfg.training.log_every
    n_seeds = cfg.num_seeds

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Grokking × Lottery Ticket Hypothesis  ({method})\n"
        f"Modular Addition (a+b) mod {cfg.dataset.p}  "
        f"— mean ± 95% CI over {n_seeds} seeds",
        fontsize=13, fontweight="bold",
    )

    panels = [
        (axes[0,0], "val_acc",   "Validation Accuracy",   True),
        (axes[0,1], "train_acc", "Training Accuracy",     False),
        (axes[1,0], "weight_l2", "Global L2 Weight Norm", False),
        (axes[1,1], "val_loss",  "Validation Loss",       False),
    ]

    for ax, field_name, title, draw_grok_thresh in panels:
        for i, (key, hist_list) in enumerate(all_histories.items()):
            sp    = sparsity_levels[i]
            color = COLORS[i % len(COLORS)]
            label = f"{sp:.0%} sparse" if sp > 0 else "dense (0%)"

            steps, mean, std = _align_histories(hist_list, field_name, n_grok, log_e)
            ci = _ci95(std, len(hist_list))

            ax.plot(steps, mean, color=color, label=label, linewidth=1.8)
            ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.15)

            # Mark mean grokking step
            if field_name == "val_acc":
                grok_steps = [h.grokking_step for h in hist_list if h.grokked]
                if grok_steps:
                    mean_sg = np.mean(grok_steps)
                    ax.axvline(mean_sg, color=color, linestyle="--", alpha=0.4, lw=1)

        if draw_grok_thresh:
            ax.axhline(cfg.training.grok_threshold, color="gray", linestyle=":",
                       lw=1, label=f"grok threshold ({cfg.training.grok_threshold})")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training Steps")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    # Add shared legend to first panel
    axes[0,0].legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_efficiency_frontier_ci(
    summary_imp:      dict,
    summary_oneshot:  dict,
    sparsity_levels:  list[float],
    n_grok_steps:     int,
    output_path:      str,
) -> None:
    """
    Grokking-Efficiency Frontier: sparsity vs S_G (mean ± CI), IMP vs one-shot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "The Grokking-Efficiency Frontier\n"
        "IMP Lottery Tickets vs One-Shot Pruning",
        fontsize=13, fontweight="bold",
    )

    for ax, summary, method_label in [
        (axes[0], summary_imp,     "IMP (primary)"),
        (axes[1], summary_oneshot, "One-shot (ablation)"),
    ]:
        sp_pct, mean_sg, ci_sg, grokked_frac = [], [], [], []

        for sp in sparsity_levels:
            key = f"{sp:.2f}"
            if key not in summary:
                continue
            rec = summary[key]
            sp_pct.append(sp * 100)
            mean_sg.append(rec["mean_sg"] if rec["grok_rate"] > 0 else n_grok_steps * 1.05)
            ci_sg.append(rec["ci_sg"])
            grokked_frac.append(rec["grok_rate"])

        if not sp_pct:
            ax.set_visible(False)
            continue

        sp_arr  = np.array(sp_pct)
        sg_arr  = np.array(mean_sg)
        ci_arr  = np.array(ci_sg)
        gf_arr  = np.array(grokked_frac)
        colors  = [plt.cm.RdYlGn(g) for g in gf_arr]

        ax.errorbar(sp_arr, sg_arr, yerr=ci_arr, fmt="o-", color="#1f77b4",
                    ecolor="gray", capsize=5, linewidth=2, markersize=8, zorder=5)

        # Color each point by grok_rate
        ax.scatter(sp_arr, sg_arr, c=gf_arr, cmap="RdYlGn",
                   s=120, vmin=0, vmax=1, zorder=6, edgecolors="k", lw=0.5)

        for x, y, gf in zip(sp_arr, sg_arr, gf_arr):
            label = f"{y:,.0f}" if gf > 0 else "DNF"
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(5, 6), fontsize=8)

        ax.set_xlabel("Sparsity (%)", fontsize=11)
        ax.set_ylabel("Steps to Grok S_G (mean ± 95% CI)", fontsize=11)
        ax.set_title(method_label, fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.8, label="Grokking rate (fraction of seeds)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_weight_norm_collapse_ci(
    all_histories:   dict[str, list[TrainingHistory]],
    sparsity_levels: list[float],
    cfg:             DictConfig,
    output_path:     str,
) -> None:
    """Weight L2 norm over training, with CI shading and grokking markers."""
    n_grok = cfg.training.n_grok_steps
    log_e  = cfg.training.log_every

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title(
        "Weight Norm Collapse at the Grokking Transition\n"
        f"(mean ± 95% CI, {cfg.num_seeds} seeds — dashed lines = mean S_G)",
        fontsize=12,
    )

    for i, (key, hist_list) in enumerate(all_histories.items()):
        sp    = sparsity_levels[i]
        color = COLORS[i % len(COLORS)]
        label = f"{sp:.0%} sparse" if sp > 0 else "dense"

        steps, mean, std = _align_histories(hist_list, "weight_l2", n_grok, log_e)
        ci = _ci95(std, len(hist_list))

        ax.plot(steps, mean, color=color, label=label, linewidth=1.8)
        ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.12)

        grok_steps = [h.grokking_step for h in hist_list if h.grokked]
        if grok_steps:
            ax.axvline(np.mean(grok_steps), color=color, linestyle="--",
                       alpha=0.5, lw=1.2)

    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Global L2 Weight Norm  ‖W‖₂", fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_fourier_checkpoints(
    all_histories:   dict[str, list[TrainingHistory]],
    sparsity_levels: list[float],
    p:               int,
    output_path:     str,
) -> None:
    """Fourier power spectra at memorisation vs grokking (first seed)."""
    keys      = list(all_histories.keys())
    dense_key = keys[0]
    sparse_key = next(
        (k for k in reversed(keys)
         if any(h.grokked and "grokking" in h.fourier_data
                for h in all_histories[k])),
        keys[1] if len(keys) > 1 else keys[0],
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    fig.suptitle(
        "Fourier Feature Emergence at the Grokking Transition\n"
        "(Embedding DFT power spectrum — first seed shown)",
        fontsize=12, fontweight="bold",
    )

    freqs = list(range(p // 2 + 1))
    for row, (key, net_label) in enumerate([
        (dense_key,  "Dense (0% sparse)"),
        (sparse_key, f"Sparse ({sparsity_levels[keys.index(sparse_key)]:.0%})"),
    ]):
        # Use first seed that has both checkpoints
        hist = next(
            (h for h in all_histories[key]
             if "memorization" in h.fourier_data and "grokking" in h.fourier_data),
            all_histories[key][0],
        )
        for col, (tag, phase) in enumerate([
            ("memorization", "Pre-grokking (just memorised)"),
            ("grokking",     "Post-grokking"),
        ]):
            ax = axes[row, col]
            if tag in hist.fourier_data:
                fd   = hist.fourier_data[tag]
                pw   = fd["power_spectrum"]
                top  = fd["top_frequencies"][:3]
                conc = fd.get("concentration", 0.0)
                ax.bar(freqs, pw, color="#4C9BE8", alpha=0.7, width=0.8)
                for tf in top:
                    ax.axvline(tf, color="#E84C4C", linestyle="--", alpha=0.8, lw=1.5)
                ax.set_title(
                    f"{net_label} — {phase}\nconcentration={conc:.2%}", fontsize=9
                )
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_title(f"{net_label} — {phase}", fontsize=9)

            ax.set_xlabel("Frequency k", fontsize=9)
            ax.set_ylabel("Mean Power",  fontsize=9)
            ax.set_xlim(0, p // 2)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ===========================================================================
# Summary statistics (multi-seed → mean/CI for frontier plot)
# ===========================================================================

def build_summary(
    all_histories:   dict[str, list[TrainingHistory]],
    sparsity_levels: list[float],
    n_grok_steps:    int,
) -> dict:
    """Build JSON-serialisable summary with multi-seed statistics."""
    out = {}
    for i, (key, hist_list) in enumerate(all_histories.items()):
        sp = sparsity_levels[i]

        grok_steps = [h.grokking_step for h in hist_list if h.grokked]
        grok_gaps  = [h.grokking_gap  for h in hist_list if h.grokked]
        final_vals = [h.val_acc[-1]   for h in hist_list if h.val_acc]

        out[key] = {
            "target_sparsity": sp,
            "grok_rate"      : len(grok_steps) / max(1, len(hist_list)),
            "mean_sg"        : float(np.mean(grok_steps)) if grok_steps else -1,
            "ci_sg"          : float(_ci95(np.std(grok_steps), len(grok_steps)))
                                if len(grok_steps) > 1 else 0.0,
            "mean_gap"       : float(np.mean(grok_gaps))  if grok_gaps  else -1,
            "mean_final_val" : float(np.mean(final_vals)) if final_vals else 0.0,
        }
    return out


# ===========================================================================
# Hydra main
# ===========================================================================

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # ── Resolve device ─────────────────────────────────────────────────────
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    # ── Determine seed list ─────────────────────────────────────────────────
    # If launched by run_parallel_seeds.py, cfg.seed is a single integer.
    # In that case run only that seed; the launcher handles multi-seed.
    # If running standalone, run cfg.num_seeds seeds from cfg.seed.
    seeds = [cfg.seed + i for i in range(cfg.num_seeds)]

    # Override with single seed from launcher if environment variable set
    launcher_seed_env = os.environ.get("GROK_SEED")
    if launcher_seed_env is not None:
        seeds = [int(launcher_seed_env)]

    sparsity_levels = list(cfg.pruning.target_sparsities)
    base_dir = Path(cfg.results_dir) / cfg.experiment.name

    print(f"\n{'#'*65}")
    print(f"  Exp B — LTH → Grok  |  device={device}  seeds={seeds}")
    print(f"  sparsities={sparsity_levels}")
    print(f"  n_grok_steps={cfg.training.n_grok_steps:,}  wd={cfg.training.weight_decay}")
    print(f"{'#'*65}\n")

    # ── Run experiments ────────────────────────────────────────────────────
    imp_histories:     dict[str, list[TrainingHistory]] = {}
    oneshot_histories: dict[str, list[TrainingHistory]] = {}

    for sp in sparsity_levels:
        key = f"{sp:.2f}"

        imp_histories[key] = run_sparsity_multiseed(
            cfg=cfg, target_sparsity=sp, device=device,
            pruning_method="imp", seeds=seeds,
            base_dir=base_dir,
        )
        if cfg.experiment.run_one_shot_ablation:
            oneshot_histories[key] = run_sparsity_multiseed(
                cfg=cfg, target_sparsity=sp, device=device,
                pruning_method="one_shot", seeds=seeds,
                base_dir=base_dir,
            )

    # ── Summaries ──────────────────────────────────────────────────────────
    summary_imp     = build_summary(imp_histories,     sparsity_levels, cfg.training.n_grok_steps)
    summary_oneshot = build_summary(oneshot_histories, sparsity_levels, cfg.training.n_grok_steps)

    results = {
        "config"          : OmegaConf.to_container(cfg, resolve=True),
        "seeds"           : seeds,
        "summary_imp"     : summary_imp,
        "summary_oneshot" : summary_oneshot,
        "imp_histories"   : {k: [h.to_dict() for h in hs]
                             for k, hs in imp_histories.items()},
        "oneshot_histories": {k: [h.to_dict() for h in hs]
                              for k, hs in oneshot_histories.items()},
    }

    results_path = base_dir / "exp_b_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # ── Figures ────────────────────────────────────────────────────────────
    fig_dir = Path(cfg.results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if cfg.experiment.plots.grokking_curves:
        plot_grokking_curves_ci(
            imp_histories, sparsity_levels, cfg,
            str(fig_dir / "exp_b_grokking_curves_imp.png"), method="IMP",
        )
        if oneshot_histories:
            plot_grokking_curves_ci(
                oneshot_histories, sparsity_levels, cfg,
                str(fig_dir / "exp_b_grokking_curves_oneshot.png"), method="One-Shot",
            )

    if cfg.experiment.plots.efficiency_frontier:
        plot_efficiency_frontier_ci(
            summary_imp, summary_oneshot, sparsity_levels,
            cfg.training.n_grok_steps,
            str(fig_dir / "exp_b_efficiency_frontier.png"),
        )

    if cfg.experiment.plots.weight_norm_collapse:
        plot_weight_norm_collapse_ci(
            imp_histories, sparsity_levels, cfg,
            str(fig_dir / "exp_b_weight_norm_collapse.png"),
        )

    if cfg.experiment.plots.fourier_checkpoints:
        plot_fourier_checkpoints(
            imp_histories, sparsity_levels, cfg.dataset.p,
            str(fig_dir / "exp_b_fourier_checkpoints.png"),
        )

    print(f"\n{'='*65}")
    print(f"  Experiment B complete.  Figures → {fig_dir}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
