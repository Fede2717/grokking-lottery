import json
import zipfile

import pytest

from analysis.summarize_exp_b import ResultValidationError, load_exp_b, summarize_cells


def test_summary_excludes_missing_event_sentinels():
    cells = [
        {
            "method": "imp", "target_sparsity": 0.9, "memorized": True,
            "grokked": True, "memorization_step": 50, "grokking_step": 100,
            "grokking_gap": 50, "final_val_acc": 0.99, "stopped_before_budget": True,
        },
        {
            "method": "imp", "target_sparsity": 0.9, "memorized": False,
            "grokked": False, "memorization_step": None, "grokking_step": None,
            "grokking_gap": None, "final_val_acc": 0.20, "stopped_before_budget": False,
        },
    ]

    row = summarize_cells(cells)[0]

    assert row["n_runs"] == 2
    assert row["n_grokked"] == 1
    assert row["success_rate"] == 0.5
    assert row["grokking_step_n"] == 1
    assert row["grokking_step_mean"] == 100
    assert row["grokking_gap_mean"] == 50


def test_summary_keeps_a_computed_negative_gap():
    cells = [{
        "method": "one_shot", "target_sparsity": 0.5, "memorized": True,
        "grokked": True, "memorization_step": 100, "grokking_step": 95,
        "grokking_gap": -5, "final_val_acc": 0.99, "stopped_before_budget": False,
    }]

    row = summarize_cells(cells)[0]

    assert row["grokking_gap_n"] == 1
    assert row["grokking_gap_mean"] == -5


def test_duplicate_cells_are_rejected_even_when_identical(tmp_path):
    entry = "exp_b/imp/sp_0.00/seed_0/grok_phase/summary.json"
    payload = json.dumps({"config": {"method": "imp", "target_sparsity": 0, "seed": 0}})
    archives = [tmp_path / "first.zip", tmp_path / "second.zip"]
    for archive in archives:
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(entry, payload)

    with pytest.raises(ResultValidationError, match="duplicate identical artifact"):
        load_exp_b(archives, strict=False)
