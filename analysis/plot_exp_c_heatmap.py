"""
analysis/plot_exp_c_heatmap.py — Experiment C figures (offline, CSV-only)
========================================================================

Reads results/<exp_c>/aggregate.csv and renders the weight-decay × sparsity
heatmaps (mean grokking step and mean final validation accuracy). Depends only
on pandas/matplotlib — no TensorBoard or wandb — so plots reproduce from CSV.

Usage
-----
    python analysis/plot_exp_c_heatmap.py --exp-dir results/exp_c
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _grid(df: pd.DataFrame, value: str, only_grokked: bool) -> tuple:
    sp_vals = sorted(df["target_sparsity"].unique())
    wd_vals = sorted(df["weight_decay"].unique())
    grid = np.full((len(sp_vals), len(wd_vals)), np.nan)
    for i, sp in enumerate(sp_vals):
        for j, wd in enumerate(wd_vals):
            cell = df[(df["target_sparsity"] == sp) & (df["weight_decay"] == wd)]
            if only_grokked:
                cell = cell[cell["grokked"] == True]  # noqa: E712
            if not cell.empty:
                grid[i, j] = cell[value].mean()
    return grid, sp_vals, wd_vals


def _heatmap(ax, grid, sp_vals, wd_vals, title, cmap, fmt, dnf_for_nan):
    im = ax.imshow(grid, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(wd_vals)))
    ax.set_xticklabels([f"λ={wd:.0e}" for wd in wd_vals], fontsize=8)
    ax.set_yticks(range(len(sp_vals)))
    ax.set_yticklabels([f"{s:.0%}" for s in sp_vals], fontsize=8)
    ax.set_xlabel("Weight decay (λ)")
    ax.set_ylabel("Sparsity")
    ax.set_title(title, fontsize=10, fontweight="bold")
    vmax = np.nanmax(grid) if np.isfinite(grid).any() else 1.0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            txt = (dnf_for_nan if np.isnan(v) else format(v, fmt))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if (not np.isnan(v) and abs(v) > vmax * 0.6) else "black")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default="results/exp_c")
    ap.add_argument("--figures-dir", default=None)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else exp_dir.parent / "figures"
    agg = exp_dir / "aggregate.csv"
    if not agg.exists():
        print(f"  no aggregate.csv at {agg}; run experiments/exp_c first.")
        return

    df = pd.read_csv(agg)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment C — weight decay × sparsity (one-shot magnitude pruning)",
                 fontweight="bold")

    sg_grid, sp_vals, wd_vals = _grid(df, "grokking_step", only_grokked=True)
    _heatmap(axes[0], sg_grid, sp_vals, wd_vals,
             "Mean steps to generalization\n(DNF = did not grok)", "viridis_r", ".0f", "DNF")

    acc_grid, _, _ = _grid(df, "final_val_acc", only_grokked=False)
    _heatmap(axes[1], acc_grid, sp_vals, wd_vals,
             "Mean final validation accuracy", "viridis", ".2f", "n/a")

    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "exp_c_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
