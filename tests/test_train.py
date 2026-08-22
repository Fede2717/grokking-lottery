"""Training schedules and threshold detection from logged evaluations."""
import tempfile
from pathlib import Path

import torch

from src.data import get_dataloaders
from src.model import get_model
from src.train import EvalSchedule, Trainer, detect_threshold_crossing, make_optimizer


def test_detector_reports_true_crossing_not_floor():
    # Curve crosses threshold at step 8 and stays. Old design would floor to
    # log_every*window (=300); post-hoc detection must report ~8.
    steps = [0, 2, 4, 6, 8, 10, 12, 14]
    vals = [0.1, 0.1, 0.2, 0.3, 0.99, 0.99, 0.99, 0.99]
    assert detect_threshold_crossing(steps, vals, threshold=0.95, window=2) == 8


def test_detector_ignores_single_blip():
    steps = [0, 5, 10, 15, 20]
    vals = [0.1, 0.99, 0.1, 0.99, 0.99]   # lone spike at 5 must not count
    assert detect_threshold_crossing(steps, vals, threshold=0.95, window=2) == 15


def test_detector_returns_minus_one_when_never_crosses():
    assert detect_threshold_crossing([0, 5, 10], [0.1, 0.2, 0.3], 0.95, 2) == -1


def test_eval_schedule_fine_then_coarse():
    s = EvalSchedule(fine_until=1000, fine_interval=5, coarse_interval=25)
    assert s.should_eval(0) and s.should_eval(5) and not s.should_eval(7)
    assert s.interval_at(10) == 5 and s.interval_at(2000) == 25
    assert s.should_eval(1000) and not s.should_eval(1005)


def test_trainer_records_unfloored_step_and_writes_outputs():
    # Tiny run: memorization should be detected at a small step (the fine schedule),
    # never floored to 300, and summary.json must be written.
    torch.manual_seed(0)
    dev = torch.device("cpu")
    tl, vl = get_dataloaders(p=7, train_frac=0.6, full_batch=True, seed=0)
    m = get_model(vocab_size=9, n_classes=7, d_model=32, n_heads=4, n_layers=1, d_ff=64)
    opt = make_optimizer(m, weight_decay=0.0, lr=3e-3)
    rd = Path(tempfile.mkdtemp()) / "run"
    tr = Trainer(
        m, tl, vl, opt, dev, rd, p=7,
        eval_schedule=EvalSchedule(fine_until=400, fine_interval=5, coarse_interval=25),
        metrics_every=50, grok_threshold=0.95, mem_threshold=0.95, grok_window=2,
        compute_fourier=False, logging_backend="none",
    )
    h = tr.train(n_steps=400, save_checkpoints=False, verbose=False)
    assert (rd / "summary.json").exists()
    assert (rd / "metrics.csv").exists()
    # If it memorized at all, the step must be far below the old 300 floor.
    if h.memorized:
        assert h.memorization_step < 200
    # Cheap-eval lists stay aligned; expensive metrics have their own axis.
    assert len(h.steps) == len(h.val_acc) == len(h.train_acc) == len(h.sparsity)
    assert len(h.metric_steps) == len(h.weight_l2)
