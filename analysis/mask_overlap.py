"""Compare Experiment A masks across training stages.

The script computes Jaccard overlap between masks saved at initialization,
memorization, and grokking. It reads either a direct aggregate CSV or per-seed
aggregate shards and writes ``mask_overlap.csv`` plus a figure. No Experiment A
results from the current implementation are included in this repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train import load_masks  # noqa: E402
try:
    from .aggregate_utils import read_aggregate_tables  # noqa: E402
except ImportError:  # direct script execution
    from aggregate_utils import read_aggregate_tables  # type: ignore  # noqa: E402



def jaccard(mask_a: dict, mask_b: dict) -> float:
    """
    Jaccard / IoU between the KEPT (mask==1) weight sets of two masks.

    Intersection-over-union of the surviving weight positions, aggregated over
    all shared parameters. 1.0 = identical subnetworks, 0.0 = disjoint.
    """
    inter = union = 0
    for name in mask_a.keys() & mask_b.keys():
        a = mask_a[name].bool()
        b = mask_b[name].bool()
        inter += int((a & b).sum())
        union += int((a | b).sum())
    return inter / union if union > 0 else float("nan")


def _sparsity_of(mask: dict) -> float:
    total = sum(m.numel() for m in mask.values())
    kept = sum(int(m.bool().sum()) for m in mask.values())
    return 1.0 - kept / total if total > 0 else 0.0


def collect_overlaps(exp_dir: Path) -> pd.DataFrame:
    """Walk Experiment A's saved stage masks and compute per-(seed, sparsity) IoU."""
    rows = []
    for masks_dir in sorted(exp_dir.glob("seed_*/masks/sp_*")):
        seed = int(masks_dir.parts[-3].split("_")[-1])
        sp = float(masks_dir.name.split("_")[-1])
        files = {f.stem: f for f in masks_dir.glob("*.pt")}
        if "mask_grok" not in files:
            continue
        grok = load_masks(files["mask_grok"])
        row = {"seed": seed, "target_sparsity": sp, "actual_sparsity": _sparsity_of(grok)}
        if "mask_mem" in files:
            row["jaccard_mem_grok"] = jaccard(load_masks(files["mask_mem"]), grok)
        if "mask_init" in files:
            row["jaccard_init_grok"] = jaccard(load_masks(files["mask_init"]), grok)
        rows.append(row)
    return pd.DataFrame(rows)


def gap_vs_sparsity(exp_dir: Path) -> pd.DataFrame:
    """Mean observed memorization-to-generalization gap per sparsity."""
    df = read_aggregate_tables(exp_dir)
    if "target_sparsity" not in df or "grokking_gap" not in df:
        return pd.DataFrame()
    valid = df[df["grokking_gap"] >= 0]
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby("target_sparsity")["grokking_gap"]
        .mean().reset_index().rename(columns={"grokking_gap": "mean_gap"})
    )



def plot(overlaps: pd.DataFrame, gaps: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Mask overlap and the memorization-to-generalization gap", fontweight="bold")

    ax = axes[0]
    if not overlaps.empty:
        agg = overlaps.groupby("target_sparsity").mean(numeric_only=True).reset_index()
        if "jaccard_mem_grok" in agg:
            ax.plot(agg["target_sparsity"] * 100, agg["jaccard_mem_grok"],
                    "o-", label="IoU(mask@mem, mask@grok)")
        if "jaccard_init_grok" in agg:
            ax.plot(agg["target_sparsity"] * 100, agg["jaccard_init_grok"],
                    "s--", label="IoU(mask@init, mask@grok)")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "no stage masks found", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("Mask Jaccard / IoU")
    ax.set_title("Subnetwork stability across training stages")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if not gaps.empty:
        ax.plot(gaps["target_sparsity"] * 100, gaps["mean_gap"], "o-", color="#d62728")
        ax.set_ylabel("Mean grokking gap (steps)")
    else:
        ax.text(0.5, 0.5, "no measured gaps yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Mean grokking gap (steps)")
    ax.set_xlabel("Sparsity (%)")
    ax.set_title("Observed gap vs sparsity")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default="results/exp_a", help="experiment results dir (Exp A)")
    ap.add_argument("--figures-dir", default=None, help="where to write figures (default: <results>/figures)")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else exp_dir.parent / "figures"

    overlaps = collect_overlaps(exp_dir)
    gaps = gap_vs_sparsity(exp_dir)

    if not overlaps.empty:
        out_csv = exp_dir / "mask_overlap.csv"
        overlaps.to_csv(out_csv, index=False)
        print(f"  saved {out_csv}")
        print(overlaps.to_string(index=False))
    else:
        print(f"  no stage masks found under {exp_dir} (run experiments/exp_a first)")

    plot(overlaps, gaps, figures_dir / "mask_overlap.png")


if __name__ == "__main__":
    main()
