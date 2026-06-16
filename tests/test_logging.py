"""Logging: a logged scalar lands in metrics.csv under the SAME tag as the TB tag."""
import csv
import tempfile
from pathlib import Path

from src.logging_utils import (
    MetricLogger, summary_to_aggregate_row, write_aggregate_csv,
)


def _read_csv(path):
    with open(path) as f:
        return list(csv.reader(f))


def test_scalar_appears_in_csv_with_same_tag():
    rd = Path(tempfile.mkdtemp())
    with MetricLogger(rd, backend="tensorboard") as logger:
        logger.log_scalars(0, {"train/acc": 0.5, "val/acc": 0.25})
        logger.log_scalars(5, {"train/acc": 0.9})

    rows = _read_csv(rd / "metrics.csv")
    assert rows[0] == ["step", "tag", "value"]          # long format
    body = rows[1:]
    assert ["0", "train/acc", "0.5"] in body            # same tag passed to TB
    assert ["0", "val/acc", "0.25"] in body
    assert ["5", "train/acc", "0.9"] in body
    # TensorBoard event files written under tb/ (same tags, same call).
    assert any((rd / "tb").glob("events.out.tfevents.*"))


def test_csv_always_on_for_none_backend():
    rd = Path(tempfile.mkdtemp())
    with MetricLogger(rd, backend="none") as logger:
        logger.log_scalars(1, {"val/loss": 2.0})
    rows = _read_csv(rd / "metrics.csv")
    assert ["1", "val/loss", "2.0"] in rows[1:]
    assert not (rd / "tb").exists()                      # no viewer, but CSV present


def test_wandb_backend_requires_run():
    rd = Path(tempfile.mkdtemp())
    try:
        MetricLogger(rd, backend="wandb")          # no run passed
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_aggregate_csv_from_summary():
    rd = Path(tempfile.mkdtemp())
    summary = {
        "config": {"method": "imp", "target_sparsity": 0.5, "seed": 0},
        "actual_sparsity": 0.5, "grokking_step": 1200, "grokking_gap": 0,
        "final_val_acc": 0.99, "grokked": True,
        "eval_resolution": {"resolution_at_grok": 25},
    }
    row = summary_to_aggregate_row(summary)
    assert row["method"] == "imp" and row["grokking_step"] == 1200
    assert row["resolution_at_grok"] == 25
    out = write_aggregate_csv(rd / "aggregate.csv", [row])
    rows = _read_csv(out)
    assert "grokking_step" in rows[0] and "method" in rows[0]
