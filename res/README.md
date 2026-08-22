# Experiment B result data

`raw/` contains one copy of each unique Kaggle ZIP bundle supplied for the June
17-19, 2026 Experiment B runs. The ZIPs are retained unchanged, including their
TensorBoard event files. Final tables do not use TensorBoard or the archived
`aggregate.csv` files.

`derived/` contains reproducible CSV tables, a short Markdown table, and two
figures generated from individual run JSON artifacts:

```powershell
python analysis/summarize_exp_b.py --figures
```

The validator expects exactly 70 final cells: two methods, seven target
sparsities, five seeds. It rejects duplicate cells, verifies path/config
agreement, represents absent events as empty CSV values, and never averages the
raw `-1` failure sentinel.

The supplied `results_exp_b_part1.zip` and `results_exp_b_part2.zip` had different
ZIP-container hashes but identical internal entry names and bytes. Only the
former content is retained as `raw/exp_b_sp_70_80.zip`. Both original hashes and
the normalized content-manifest hash are recorded in
`exp_b_provenance.json`.

The exact source commit is not recorded. The commit in the manifest is a
high-confidence inference because the Kaggle notebooks cloned the default branch
without recording `git rev-parse HEAD`. The manifest gives the supporting
evidence and remaining provenance gaps.
