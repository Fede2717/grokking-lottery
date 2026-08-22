import pandas as pd
import pytest

from analysis.aggregate_utils import read_aggregate_tables


def test_reads_parallel_seed_shards(tmp_path):
    pd.DataFrame([{"seed": 0, "grokked": True}]).to_csv(
        tmp_path / "aggregate_seed_0.csv", index=False
    )
    pd.DataFrame([{"seed": 1, "grokked": False}]).to_csv(
        tmp_path / "aggregate_seed_1.csv", index=False
    )

    table = read_aggregate_tables(tmp_path)

    assert sorted(table["seed"].tolist()) == [0, 1]


def test_rejects_seed_shard_path_mismatch(tmp_path):
    pd.DataFrame([{"seed": 1}]).to_csv(tmp_path / "aggregate_seed_0.csv", index=False)

    with pytest.raises(ValueError, match="seed shard/path mismatch"):
        read_aggregate_tables(tmp_path)
