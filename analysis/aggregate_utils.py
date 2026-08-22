"""Read direct-run aggregate tables or parallel per-seed shards."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


_SHARD_RE = re.compile(r"aggregate_seed_(-?[0-9]+)\.csv$")


def read_aggregate_tables(exp_dir: Path) -> pd.DataFrame:
    """Prefer process-safe seed shards; fall back to a direct-run aggregate."""
    shards = sorted(exp_dir.glob("aggregate_seed_*.csv"))
    paths = shards or ([exp_dir / "aggregate.csv"] if (exp_dir / "aggregate.csv").exists() else [])
    if not paths:
        return pd.DataFrame()

    tables = []
    for path in paths:
        table = pd.read_csv(path)
        match = _SHARD_RE.match(path.name)
        if match and "seed" in table:
            expected_seed = int(match.group(1))
            actual_seeds = {int(value) for value in table["seed"].dropna().unique()}
            if actual_seeds and actual_seeds != {expected_seed}:
                raise ValueError(
                    f"seed shard/path mismatch in {path}: expected {expected_seed}, got {actual_seeds}"
                )
        tables.append(table)
    return pd.concat(tables, ignore_index=True)
