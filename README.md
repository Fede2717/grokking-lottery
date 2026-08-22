# Grokking and sparse subnetworks

I started this project to study whether the sparse structure selected by a
trained network can reproduce grokking after its surviving weights are rewound
to initialization. The experiments use modular addition with a small
Transformer and compare magnitude-pruned subnetworks across several sparsity
levels.

The current results come from Experiment B. Experiments A and C are implemented
as exploratory drivers, but there are no result sets for them under the current
implementation.

## Background

Grokking is a delayed transition from fitting the training set to generalizing
on held-out examples. On modular arithmetic, a network may memorize the training
pairs well before it learns a rule that works across the full operation table.

The Lottery Ticket Hypothesis asks whether a dense network contains a sparse
subnetwork that can train successfully when its surviving weights are restored
to an early checkpoint. This project uses the initial parameter state, `W0`, as
the rewind point.

The closest prior work is Minegishi, Iwasawa, and Matsuo (2025), *Bridging
Lottery Ticket and Grokking: Understanding Grokking from Inner Structure of
Networks*. It studies the relationship between sparse network structure and
grokking directly. The pruning procedures used here have their own details,
especially the optimizer state retained during iterative mask discovery.

## Experimental setup

| Item | Value |
|---|---|
| Task | Modular addition, `p = 97` |
| Examples | All `97^2 = 9,409` ordered pairs |
| Input | `[a, +, b, =]`, vocabulary size `99` |
| Split | `4,704` train and `4,705` validation examples |
| Model | One-block non-causal Transformer, no LayerNorm |
| Width | `d_model = 128`, four attention heads |
| MLP | `d_ff = 512`, ReLU |
| Dropout | `0.0` |
| Position encoding | Learned, four positions |
| Output | Final-position state through an untied linear head |
| Parameters | `223,360` total, `209,024` prunable |
| Optimizer | AdamW, learning rate `1e-3`, weight decay `1.0` |
| AdamW settings | `betas = (0.9, 0.98)`, `eps = 1e-8` |
| Batch size | Full training split |
| Final-run budget | `40,000` updates |

Each experiment seed controls Python, NumPy, PyTorch, CUDA, model
initialization, and the dataset split. The split therefore changes across
seeds.

Every `nn.Linear.weight` tensor is prunable, including the attention
projections, MLP matrices, and output head. Embeddings, biases, and normalization
parameters are excluded. Masks are reapplied after every optimizer update, so
pruned weights cannot regrow.

Training and validation are evaluated at labeled step `0`, every five steps
through step `1000`, and every 25 steps afterward. Step `0` is the first
post-update evaluation. Memorization and grokking are the first evaluations in
runs of two consecutive train or validation accuracies at or above `0.95`. The
earliest reported event is step `0`, confirmed at step `5`.

The default config disables early stopping. The Experiment B run configs enable
it for sparse conditions with `patience = 500`; the confirming evaluation counts
toward that patience. Dense baselines run for the full budget.

## Experiment B protocols

The grid contains sparsities `0%, 20%, 50%, 70%, 80%, 90%, 95%` and seeds
`0, 1, 2, 3, 4` for each method.

### Post-grok one-shot

For each nonzero sparsity, the code:

1. Initializes a dense model and saves `W0`.
2. Trains for `1,200` full-batch AdamW updates.
3. Ranks all prunable weights globally by absolute magnitude and constructs one
   mask at the target sparsity.
4. Restores surviving weights from `W0`.
5. Trains the sparse model with a fresh AdamW optimizer.

All five dense warm-ups grokked before update `1,200`. These masks characterize
weights after grokking; the experiment does not cover mask discovery before
grokking or at initialization. The same seed-specific warm-up is repeated
across the six nonzero sparsity cells.

### Stateful IMP

The iterative procedure trains for 400 updates per round, prunes `20%` of the
currently active weights globally, and restores surviving weights from `W0`.
Model weights are rewound after each round, while the AdamW optimizer and its
moments are retained. Final sparse training starts from `W0` with a fresh
optimizer.

| Target sparsity | Rounds | Discovery updates |
|---:|---:|---:|
| 20% | 1 | 400 |
| 50% | 4 | 1,600 |
| 70% | 6 | 2,400 |
| 80% | 8 | 3,200 |
| 90% | 11 | 4,400 |
| 95% | 14 | 5,600 |

This differs from the usual lottery-ticket rewind because optimizer state is
not reset during mask discovery. The saved results contain only the final IMP
round for each cell, so earlier-round memorization and generalization cannot be
reconstructed.

## Results

A run is successful when validation accuracy reaches `0.95` within the final
training budget. Medians use successful seeds only. `DNF` means that no seed in
the condition reached the threshold.

| Method | Sparsity | Successful seeds | Median grokking step |
|---|---:|---:|---:|
| Stateful IMP | 0% | 5/5 | 935.0 |
| Stateful IMP | 20% | 5/5 | 835.0 |
| Stateful IMP | 50% | 5/5 | 29,275.0 |
| Stateful IMP | 70% | 3/5 | 34,625.0 |
| Stateful IMP | 80% | 4/5 | 572.5 |
| Stateful IMP | 90% | 1/5 | 34,925.0 |
| Stateful IMP | 95% | 0/5 | DNF |
| Post-grok one-shot | 0% | 5/5 | 935.0 |
| Post-grok one-shot | 20% | 5/5 | 380.0 |
| Post-grok one-shot | 50% | 4/5 | 1,107.5 |
| Post-grok one-shot | 70% | 4/5 | 687.5 |
| Post-grok one-shot | 80% | 4/5 | 672.5 |
| Post-grok one-shot | 90% | 3/5 | 560.0 |
| Post-grok one-shot | 95% | 1/5 | 1,975.0 |

Post-grok one-shot has more successful seeds at 70%, 90%, and 95% sparsity.
Stateful IMP has one more success at 50%, and the methods tie at 0%, 20%, and
80%. Among successful runs, the one-shot medians from 20% through 90% are often
earlier than the dense median. These medians are conditioned on success, which
matters when several seeds fail.

The results support a limited conclusion: masks selected from trained dense
models can help some rewound subnetworks reach high validation accuracy. They do
not show that sparse models generally grok faster, that the two pruning methods
are equivalent, or that 95% sparsity is reliable.

Seed-level tables and figures are in [`res/derived/`](res/derived/). The raw
result bundles are in [`res/raw/`](res/raw/), with checksums and provenance notes
in [`res/exp_b_provenance.json`](res/exp_b_provenance.json).

## Limitations

- Experiment B has no random-pruning control or pruning-at-initialization
  baseline such as SNIP, GraSP, SynFlow, or edge-popup.
- The one-shot masks are measured after grokking, so they do not answer the
  original pre-grokking pruning question.
- IMP retains optimizer state across weight rewinds, and intermediate rounds
  are absent from the saved results.
- Model initialization and dataset split variation share the same seed.
- The auxiliary mechanistic metrics in the code are not present in the
  Experiment B results, so these runs do not support conclusions from them.
- Experiment C fixes discovery weight decay at zero and applies the grid value
  only during final training, but no Experiment C results are included.

## Reproduction

Install the package and development dependencies:

```bash
pip install -e ".[dev]"
pytest -q
```

Reconstruct the Experiment B tables and figures from the raw bundles:

```bash
python analysis/summarize_exp_b.py --figures
```

Run one seed of each experiment:

```bash
python experiments/exp_a_grok_then_prune.py experiment=exp_a seed=0 num_seeds=1
python experiments/exp_b_lth_then_grok.py experiment=exp_b seed=0 num_seeds=1
python experiments/exp_c_wd_ablation.py experiment=exp_c seed=0 num_seeds=1
```

Run five Experiment B seeds across two GPUs:

```bash
python scripts/run_parallel_seeds.py --experiment exp_b --num-seeds 5 --base-seed 0 --num-gpus 2
```

## References

- Power et al. (2022), *Grokking: Generalization Beyond Overfitting on Small
  Algorithmic Datasets*.
- Frankle and Carbin (2019), *The Lottery Ticket Hypothesis: Finding Sparse,
  Trainable Neural Networks*.
- Gouki Minegishi, Yusuke Iwasawa, and Yutaka Matsuo (2025), *Bridging Lottery
  Ticket and Grokking: Understanding Grokking from Inner Structure of Networks*,
  Transactions on Machine Learning Research.
