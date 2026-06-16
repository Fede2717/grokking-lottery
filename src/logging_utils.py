"""
src/logging_utils.py — Unified Metric Logging
==============================================

A single abstraction, ``MetricLogger``, fans every scalar out to all active
sinks through one call::

    logger.log_scalars(step, {"train/acc": 0.9, "val/acc": 0.5})

Because every sink receives the *same* ``{tag: value}`` dict, the tags are
identical across sinks — which guarantees that the offline CSV plots match
TensorBoard (and Weights & Biases, if enabled) exactly.

Sinks
-----
    CSV          ALWAYS ON, regardless of the chosen viewer backend.  Appends to
                 ``<run_dir>/metrics.csv`` in LONG format with columns
                 ``step,tag,value`` (tag names identical to the TB tags).
    tensorboard  DEFAULT viewer.  Writes TB event files to ``<run_dir>/tb/``.
    none         No live viewer.  CSV (and the run's summary.json, written by the
                 Trainer) are still produced.
    wandb        OPTIONAL, opt-in only.  Requires an explicitly-passed wandb run;
                 the logger never logs in or reads ``WANDB_API_KEY`` itself.

The viewer backend is selected from ``configs/config.yaml`` via
``logging.backend`` ∈ {tensorboard, csv, none, wandb}.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

try:  # TensorBoard is the default viewer but kept optional at import time.
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when TB missing
    SummaryWriter = None  # type: ignore
    _TB_AVAILABLE = False


# Canonical scalar tags used across the project.  Kept here so experiments,
# the Trainer and the offline plotters share one vocabulary.
CANONICAL_TAGS = (
    "train/loss",
    "train/acc",
    "val/loss",
    "val/acc",
    "weights/l2",
    "weights/l1",
    "fourier/concentration",
    "fourier/spectral_entropy",
    "hessian/lambda_max",
    "sparsity",
)


class MetricLogger:
    """Fan-out scalar logger (CSV always on; one optional live viewer)."""

    VALID_BACKENDS = ("tensorboard", "csv", "none", "wandb")

    def __init__(
        self,
        run_dir: str | Path,
        backend: str = "tensorboard",
        wandb_run=None,
        csv_filename: str = "metrics.csv",
    ) -> None:
        backend = (backend or "none").lower()
        if backend not in self.VALID_BACKENDS:
            raise ValueError(
                f"Unknown logging backend {backend!r}; "
                f"valid options: {self.VALID_BACKENDS}"
            )

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend

        # ── CSV sink (ALWAYS ON) ──────────────────────────────────────────
        self.csv_path = self.run_dir / csv_filename
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["step", "tag", "value"])
        self._csv_file.flush()

        # ── Optional live viewer sink ─────────────────────────────────────
        self._tb = None
        self._wandb_run = None

        if backend == "tensorboard":
            if not _TB_AVAILABLE:
                raise ImportError(
                    "logging.backend='tensorboard' but TensorBoard is not "
                    "installed. Run `pip install tensorboard` or set "
                    "logging.backend='none'."
                )
            self._tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))

        elif backend == "wandb":
            # Opt-in only. The caller is responsible for wandb.init(); we never
            # log in or touch WANDB_API_KEY here.
            if wandb_run is None:
                raise ValueError(
                    "logging.backend='wandb' requires an explicit wandb run to "
                    "be passed (wandb_run=...). The logger never logs in itself."
                )
            self._wandb_run = wandb_run

    # ------------------------------------------------------------------

    def log_scalars(self, step: int, scalars: Mapping[str, float]) -> None:
        """Fan a {tag: value} dict out to every active sink at ``step``."""
        # CSV (always on) — long format, one row per (step, tag).
        for tag, value in scalars.items():
            if value is None:
                continue
            self._csv_writer.writerow([int(step), tag, float(value)])
        self._csv_file.flush()

        # TensorBoard.
        if self._tb is not None:
            for tag, value in scalars.items():
                if value is None:
                    continue
                self._tb.add_scalar(tag, float(value), int(step))

        # Weights & Biases (opt-in).
        if self._wandb_run is not None:
            payload = {t: float(v) for t, v in scalars.items() if v is not None}
            if payload:
                self._wandb_run.log(payload, step=int(step))

    # ------------------------------------------------------------------

    def flush(self) -> None:
        self._csv_file.flush()
        if self._tb is not None:
            self._tb.flush()

    def close(self) -> None:
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass
        if self._tb is not None:
            self._tb.close()
        # wandb run lifecycle (init/finish) is owned by the caller.

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ===========================================================================
# Per-experiment aggregate export (headline plots read this offline)
# ===========================================================================

def write_aggregate_csv(path: str | Path, rows: Sequence[Mapping]) -> Path:
    """
    Write one row per (condition, seed) to ``results/<exp>/aggregate.csv``.

    Columns are the union of all keys across rows (stable, first-seen order),
    so experiment params (sparsity, weight_decay, method, seed) and summary
    fields (grokking_step, grokking_gap, final_val_acc, ...) all land here for
    the offline headline plotters — no TensorBoard / wandb dependency.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in columns:
                columns.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
    return path.resolve()


def summary_to_aggregate_row(summary: Mapping, extra: Mapping | None = None) -> dict:
    """
    Flatten a per-run ``summary.json`` dict into a single aggregate row.

    The nested ``config`` block is promoted to top-level columns (e.g. sparsity,
    weight_decay, method, seed) and merged with the headline summary fields.
    """
    row: dict = {}
    cfg = summary.get("config", {}) or {}
    for k, v in cfg.items():
        if not isinstance(v, (dict, list)):
            row[k] = v
    for k in (
        "actual_sparsity", "memorization_step", "grokking_step", "grokking_gap",
        "final_train_acc", "final_val_acc", "grokked",
    ):
        if k in summary:
            row[k] = summary[k]
    res = summary.get("eval_resolution", {}) or {}
    if "resolution_at_grok" in res:
        row["resolution_at_grok"] = res["resolution_at_grok"]
    if extra:
        row.update(extra)
    return row


def load_summary(run_dir: str | Path) -> dict:
    """Load a run's ``summary.json`` (returns {} if absent)."""
    p = Path(run_dir) / "summary.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)
