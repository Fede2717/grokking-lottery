# Experiment B results

Generated from individual `grok_phase/summary.json` and `history.json` artifacts.
Archived `aggregate.csv` files were ignored because parallel workers overwrote them.

## Final grokking success

| Method | Sparsity | Success | Rate | Median grok step (successes only) |
|---|---:|---:|---:|---:|
| Stateful IMP | 0% | 5/5 | 100% | 935 |
| Stateful IMP | 20% | 5/5 | 100% | 835 |
| Stateful IMP | 50% | 5/5 | 100% | 29275 |
| Stateful IMP | 70% | 3/5 | 60% | 34625 |
| Stateful IMP | 80% | 4/5 | 80% | 572.5 |
| Stateful IMP | 90% | 1/5 | 20% | 34925 |
| Stateful IMP | 95% | 0/5 | 0% | NA |
| Post-grok one-shot | 0% | 5/5 | 100% | 935 |
| Post-grok one-shot | 20% | 5/5 | 100% | 380 |
| Post-grok one-shot | 50% | 4/5 | 80% | 1107.5 |
| Post-grok one-shot | 70% | 4/5 | 80% | 687.5 |
| Post-grok one-shot | 80% | 4/5 | 80% | 672.5 |
| Post-grok one-shot | 90% | 3/5 | 60% | 560 |
| Post-grok one-shot | 95% | 1/5 | 20% | 1975 |

`NA` means that no seed met the two-evaluation 95% validation threshold within 40,000 steps.
Failure sentinels (`-1`) are represented as missing and are never averaged.

## Dense warm-up used for one-shot masks

| Seed | Repeated sparsities | Identical | Memorization step | Grokking step | Final val accuracy |
|---:|---:|:---:|---:|---:|---:|
| 0 | 6 | True | 150 | 980 | 0.996174 |
| 1 | 6 | True | 150 | 935 | 0.998937 |
| 2 | 6 | True | 145 | 965 | 0.999575 |
| 3 | 6 | True | 175 | 690 | 0.997024 |
| 4 | 6 | True | 140 | 470 | 0.998512 |

All warm-ups grokked before magnitude measurement at update 1,200. The one-shot masks
were selected after grokking and rewound to initialization.
