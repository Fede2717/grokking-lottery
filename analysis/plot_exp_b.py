"""
analysis/plot_exp_b.py — Experiment B figures (offline, CSV-only)
================================================================

Reproduces the Experiment B headline figures from CSV alone (no TensorBoard /
wandb):
    * Efficiency frontier: sparsity vs steps-to-generalization (mean ± 95% CI),
      per pruning method, point colour = fraction of seeds that grokked.
    * Gap vs sparsity: the memorization→generalization delay shrinking toward 0.
    * Grokking curves: val accuracy vs step per sparsity (read from per-run
      metrics.csv, averaged across seeds on a common grid).

Correct terminology: a sparse subnetwork ELIMINATES the delay (gap → 0); it does
not merely "grok faster".

Usage
-----
    python analysis/plot_exp_b.py --exp-dir results/exp_b
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ci95(std: float, n: int) -> float:
    return 1.96 * std / max(n, 1) ** 0.5 if n > 1 else 0.0


def plot_frontier(df: pd.DataFrame, ax_sg, ax_gap) -> None:
    for method, sub in df.groupby("method"):
        sp_vals = sorted(sub["target_sparsity"].unique())
        sg_mean, sg_ci, grok_rate, gap_mean, sp_pct = [], [], [], [], []
        for sp in sp_vals:
            cell = sub[sub["target_sparsity"] == sp]
            grokked = cell[cell["grokked"] == True]  # noqa: E712
            rate = len(grokked) / max(len(cell), 1)
            sp_pct.append(sp * 100)
            grok_rate.append(rate)
            if not grokked.empty:
                sg_mean.append(grokked["grokking_step"].mean())
                sg_ci.append(_ci95(grokked["grokking_step"].std(ddof=0), len(grokked)))
                gap_mean.append(grokked["grokking_gap"].mean())
            else:
                sg_mean.append(np.nan); sg_ci.append(0.0); gap_mean.append(np.nan)

        ax_sg.errorbar(sp_pct, sg_mean, yerr=sg_ci, fmt="o-", capsize=4, label=method)
        sc = ax_sg.scatter(sp_pct, sg_mean, c=grok_rate, cmap="RdYlGn", vmin=0, vmax=1,
                           s=90, edgecolors="k", lw=0.5, zorder=5)
        ax_gap.plot(sp_pct, gap_mean, "o-", label=method)

    ax_sg.set_xlabel("Sparsity (%)"); ax_sg.set_ylabel("Steps to generalization")
    ax_sg.set_title("Efficiency frontier (colour = grok rate)", fontsize=10, fontweight="bold")
    ax_sg.grid(True, alpha=0.3); ax_sg.legend(fontsize=8)

    ax_gap.axhline(0, color="gray", lw=1, ls=":")
    ax_gap.set_xlabel("Sparsity (%)"); ax_gap.set_ylabel("Mean grokking gap (steps)")
    ax_gap.set_title("Delay elimination (gap → 0)", fontsize=10, fontweight="bold")
    ax_gap.grid(True, alpha=0.3); ax_gap.legend(fontsize=8)


def plot_curves(exp_dir: Path, ax, method: str = "imp") -> None:
    grid = np.linspace(0, 1, 200)  # normalised training progress
    plotted = False
    for sp_dir in sorted((exp_dir / method).glob("sp_*")):
        sp = float(sp_dir.name.split("_")[-1])
        curves = []
        for csv in sp_dir.glob("seed_*/grok_phase/metrics.csv"):
            d = pd.read_csv(csv)
            v = d[d["tag"] == "val/acc"]
            if v.empty:
                continue
            steps = v["step"].to_numpy(); acc = v["value"].to_numpy()
            if steps.max() > steps.min():
                xs = (steps - steps.min()) / (steps.max() - steps.min())
                curves.append(np.interp(grid, xs, acc))
        if curves:
            mean = np.mean(curves, axis=0)
            ax.plot(grid * 100, mean, label=f"{sp:.0%} sparse" if sp > 0 else "dense")
            plotted = True
    ax.set_xlabel("Training progress (%)"); ax.set_ylabel("Val accuracy")
    ax.set_title(f"Grokking curves ({method})", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default="results/exp_b")
    ap.add_argument("--figures-dir", default=None)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else exp_dir.parent / "figures"
    agg = exp_dir / "aggregate.csv"
    if not agg.exists():
        print(f"  no aggregate.csv at {agg}; run experiments/exp_b first.")
        return

    df = pd.read_csv(agg)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    fig.suptitle("Experiment B — LTH ticket then grok", fontweight="bold")
    plot_frontier(df, axes[0], axes[1])
    plot_curves(exp_dir, axes[2], method="imp")

    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "exp_b_frontier.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
