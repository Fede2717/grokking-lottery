"""Plot Experiment A aggregate results.

The two panels show validation accuracy immediately after pruning and grokking
steps after rewinding the selected mask to ``W0`` or ``W_mem``. The script reads
a direct aggregate CSV or per-seed shards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from .aggregate_utils import read_aggregate_tables
except ImportError:  # direct script execution
    from aggregate_utils import read_aggregate_tables


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default="results/exp_a")
    ap.add_argument("--figures-dir", default=None)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else exp_dir.parent / "figures"
    df = read_aggregate_tables(exp_dir)
    if df.empty:
        print(f"  no aggregate tables under {exp_dir}; run experiments/exp_a first.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment A: grok, prune, and rewind", fontweight="bold")

    # Circuit survival (no retraining)
    ax = axes[0]
    if "circuit_survival_acc" in df:
        surv = (df.dropna(subset=["circuit_survival_acc"])
                  .groupby("target_sparsity")["circuit_survival_acc"].mean().reset_index())
        if not surv.empty:
            colors = ["#4CAF50" if a >= 0.9 else "#F44336" for a in surv["circuit_survival_acc"]]
            ax.bar([f"{s:.0%}" for s in surv["target_sparsity"]], surv["circuit_survival_acc"],
                   color=colors, edgecolor="k", alpha=0.85)
            ax.axhline(0.95, color="k", ls="--", lw=1, label="grok threshold")
            ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Sparsity"); ax.set_ylabel("Val accuracy (no retraining)")
    ax.set_title("Generalizing-circuit survival under pruning", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Rewind comparison
    ax = axes[1]
    rewind_rows = df[df.get("rewind").isin(["W_init", "W_mem"])] if "rewind" in df else pd.DataFrame()
    for label, sub in (rewind_rows.groupby("rewind") if not rewind_rows.empty else []):
        grokked = sub[sub["grokked"] == True]  # noqa: E712
        if grokked.empty:
            continue
        m = grokked.groupby("target_sparsity")["grokking_step"].mean().reset_index()
        ax.plot(m["target_sparsity"] * 100, m["grokking_step"], "o-", label=f"rewind {label}")
    dense = df[df.get("phase") == "dense"] if "phase" in df else pd.DataFrame()
    if not dense.empty and (dense["grokked"] == True).any():  # noqa: E712
        ax.axhline(dense[dense["grokked"] == True]["grokking_step"].mean(),  # noqa: E712
                   color="gray", ls="--", lw=1.5, label="dense baseline")
    ax.set_xlabel("Sparsity (%)"); ax.set_ylabel("Steps to generalization")
    ax.set_title("Rewind strategy: W_0 vs W_mem", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)

    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "exp_a_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
