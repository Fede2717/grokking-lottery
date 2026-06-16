# Grokking × Lottery Ticket Hypothesis

A **partial reproduction** of Minegishi, Iwasawa & Matsuo, *"Bridging Lottery Ticket and Grokking"*
(TMLR 2025, [arXiv:2310.19470](https://arxiv.org/abs/2310.19470)), plus original ablations,
built as clean, extensible research infrastructure.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-orange.svg)](https://pytorch.org)
[![Hydra](https://img.shields.io/badge/config-Hydra-89b4fa.svg)](https://hydra.cc)

---

## The question

**Grokking** (Power et al., 2022): a network trained far past memorization on a small
algorithmic task suddenly generalizes — a delayed phase transition.
**The Lottery Ticket Hypothesis** (Frankle & Carlin, 2019): a dense network contains a
sparse subnetwork that trains well from (near-)initialization.

Minegishi et al. (2025) connect the two. The measured claim we reproduce, stated carefully:

> A sparse subnetwork found by magnitude pruning **eliminates the
> memorization→generalization delay** — the gap shrinks toward **zero** — rather than
> the dense network "grokking faster". The surviving structure (the mask) and its
> overlap across training stages track the generalization improvement.

We use the **correct terminology** throughout: pruned/sparse subnetworks **eliminate the
delay (gap → 0)**; we never claim "grok faster", "4–15×", or "100% grokking at 95%".
Earlier numbers like `S_G = 300` and `gap = 0` for every sparse condition were
**measurement-floor artifacts** (see below), not findings.

---

## Canonical setup (the default)

The default configuration is the canonical grokking-transformer setup
(Nanda 2023; Varma 2023; Power 2022):

| Component | Default |
|---|---|
| Architecture | **1-layer** decoder-only transformer, **no LayerNorm** |
| Width | `d_model=128`, `4 heads` (`d_head=32`), `d_mlp=512`, ReLU |
| Embeddings | learned positional; **untied** embed/unembed; logits read at the `=` token; head has no bias |
| Optimizer | AdamW, `lr=1e-3`, `weight_decay=1.0`, `betas=(0.9, 0.98)`, `eps=1e-8`, **no gradient clipping** |
| Batching | **full-batch** gradient descent |
| Task | `(a + b) mod 97`, 50% train split (Power; Nanda's mainline uses 0.3) |
| Thresholds | train (memorization) and val (generalization) both `P = 0.95` |

A **declared non-canonical variant** is kept for comparison: a 2-layer **Pre-LayerNorm**
model (`model=transformer_2l_preln`) and a mini-batch regime (`training=minibatch`).

---

## The measurement fix (read this first)

The original detector only evaluated on multiples of `log_every` and required
`grok_window` consecutive passing evals, so the **earliest detectable step was
`log_every × grok_window = 300`**. Every fast (sparse) condition was pinned to 300 with
`gap = 0` — an artifact, not a result.

The fix (in [src/train.py](src/train.py)):
- **Two independent schedules.** Cheap eval (train/val acc + loss) runs on a *fine-early*
  schedule — every 5 steps for the first 1000 steps, then every 25
  ([`EvalSchedule`](src/train.py)). Expensive metrics (Fourier, Hessian, weight norms)
  run on `metrics_every` (default 1000).
- **Post-hoc detection.** `memorization_step` / `grokking_step` / `grokking_gap` are
  computed from the logged accuracy curves as the *first genuine threshold crossing*
  satisfying the window — decoupled from the log period
  ([`detect_threshold_crossing`](src/train.py)).
- The eval resolution is recorded in each run's `summary.json`, so the timing
  uncertainty is explicit.

---

## Repository layout

```
grokking-lottery/
├── configs/            Hydra config tree (canonical defaults + labeled variants)
│   ├── config.yaml             root: composes canonical model+training; logging.backend
│   ├── dataset/                modular_add (default), modular_mul (seam)
│   ├── model/                  transformer_1l (default), transformer_2l_preln (variant)
│   ├── training/               default (full-batch), minibatch (variant)
│   ├── pruning/                imp
│   └── experiment/             exp_a, exp_b, exp_c
├── src/
│   ├── data.py         Modular-arithmetic dataset; full-batch loader; task seam
│   ├── model.py        Custom no-LayerNorm decoder block; isinstance-based pruning surface
│   ├── train.py        Trainer (two-schedule measurement), post-hoc detection, optimizer factory
│   ├── prune.py        Global magnitude pruning, IMP, one-shot, disk-based rewind
│   ├── metrics.py      Fourier, weight norms, effective rank, GSNR, exact-HVP Hessian
│   ├── logging_utils.py  MetricLogger (CSV always-on; TensorBoard default; wandb opt-in)
│   └── runner.py       Config → object helpers shared by experiments
├── experiments/        exp_a / exp_b / exp_c — orchestration only (no plotting)
├── analysis/           Offline plots + mask-overlap, read CSV/JSON only
├── scripts/            run_parallel_seeds.py (multi-seed launcher)
├── tests/              pytest suite
└── results/            per-run metrics.csv / summary.json + per-exp aggregate.csv + figures/
```

---

## Outputs (reproducible from CSV alone)

Every run writes, regardless of viewer backend:

- `results/<exp>/<run>/metrics.csv` — **long format** `step,tag,value` (tags identical to
  the TensorBoard tags, so offline plots match TensorBoard exactly).
- `results/<exp>/<run>/summary.json` — config, sparsity, memorization/grokking/gap,
  final accuracies, `grokked`, eval resolution, checkpoint paths.
- `results/<exp>/aggregate.csv` — one row per (condition, seed) for headline plots.

The **live viewer** is selected by `logging.backend` ∈ `{tensorboard (default), csv, none,
wandb}`. CSV + JSON are always produced — **no external account is required**. View
TensorBoard with `tensorboard --logdir results/`.

---

## Install & run (local)

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows; use bin/activate on Linux/macOS
pip install -e ".[dev]"                                 # CPU torch by default
```

**Fast debug run** (CPU, ~seconds; confirms the pipeline end-to-end):

```bash
python -m experiments.exp_b_lth_then_grok \
  dataset.p=11 training.n_grok_steps=600 training.metrics_every=150 \
  "pruning.target_sparsities=[0.0,0.5]" pruning.imp_steps_per_round=40 \
  num_seeds=1 logging.backend=none results_dir=results_debug
```

**Canonical run** (GPU recommended):

```bash
python -m experiments.exp_b_lth_then_grok                 # canonical defaults
python -m experiments.exp_a_grok_then_prune experiment=exp_a
python -m experiments.exp_c_wd_ablation     experiment=exp_c
```

**Common overrides:**

```bash
python -m experiments.exp_b_lth_then_grok training.weight_decay=1e-2
python -m experiments.exp_b_lth_then_grok model=transformer_2l_preln   # non-canonical variant
python -m experiments.exp_b_lth_then_grok dataset=modular_mul          # task seam
python -m experiments.exp_b_lth_then_grok compute_hessian=true         # expensive/fragile, off by default
```

**Plots (offline, from CSV/JSON only):**

```bash
python analysis/plot_exp_b.py        --exp-dir results/exp_b
python analysis/plot_exp_a.py        --exp-dir results/exp_a
python analysis/plot_exp_c_heatmap.py --exp-dir results/exp_c
python analysis/mask_overlap.py      --exp-dir results/exp_a
```

## Run on Kaggle (no keys needed)

```python
!git clone https://github.com/YOUR_USERNAME/grokking-lottery.git
%cd grokking-lottery
!pip install -e ".[dev]" -q
# Multi-seed across the 2× T4s; defaults to TensorBoard + CSV, no login:
!python scripts/run_parallel_seeds.py --experiment exp_b --num-seeds 5 --num-gpus 2
```

Optional Weights & Biases (opt-in only): `pip install -e ".[wandb]"`, `wandb login`, then
add `logging.backend=wandb wandb.enabled=true wandb.entity=<you>`. The launcher never
injects `WANDB_API_KEY` or assumes a login.

---

## Experiments

- **Exp A — control (canonical direction):** train dense to full grokking, then prune the
  grokked weights to extract the **"grokked ticket"**, rewind (to `W_0` vs `W_mem`), and
  retrain. Also measures **circuit survival** (val accuracy of the pruned post-grokking
  net with no retraining). Maps to a subnetwork extracted *after* generalization.
- **Exp B — extension:** find a sparse ticket *before* grokking via **IMP** (with one-shot
  magnitude pruning as an ablation), rewind to `W_0`, then run the full grokking phase.
  Tests whether the sparse subnetwork eliminates the delay.
- **Exp C — ablation:** weight-decay × sparsity grid using **one-shot magnitude pruning**
  (not IMP). The short pre-pruning pass uses `weight_decay=0` for **all** cells (removing
  the weight-decay confound); the grid value is applied **only** in the grokking phase.

**Reproduction-fidelity analysis:** [analysis/mask_overlap.py](analysis/mask_overlap.py)
computes Jaccard/IoU between masks extracted at the init / memorization / grokking stages
and reports the gap-vs-sparsity relationship — reproducing Minegishi's measured claims
(structure matters; the sparse subnetwork eliminates the delay; mask change tracks the
generalization improvement).

---

## What we reproduce / what differs

**Reproduce (qualitatively, on a single task family):**
- The LTH↔grokking link: a sparse magnitude-pruned subnetwork eliminates the
  memorization→generalization delay rather than the dense net grokking faster.
- The mask-overlap framing (subnetwork structure stabilizes from memorization to
  generalization).

**Differs from Minegishi et al.:**
- Single task family (modular arithmetic) and a small model; we do not sweep the paper's
  full architecture/task matrix.
- Default uses 50% train split (Power) rather than 0.3; this is configurable.
- Quantitative thresholds, schedules and seed counts are our own; we report
  measured numbers with explicit eval resolution, not the paper's exact values.

## Limitations

- Single task family (modular `add`/`mul` mod 97) and one small architecture — findings
  should not be over-generalized.
- Grokking step counts depend on seed, split and schedule; treat them as estimates with
  the recorded eval resolution as uncertainty.
- The Hessian sharpness metric (exact HVP) is **off by default** — expensive and
  numerically fragile.

---

## References

```
Minegishi, Iwasawa, Matsuo (2025). Bridging Lottery Ticket and Grokking. TMLR. arXiv:2310.19470.
Frankle & Carlin (2019). The Lottery Ticket Hypothesis. ICLR.
Power et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177.
Nanda et al. (2023). Progress Measures for Grokking via Mechanistic Interpretability. ICLR.
Varma et al. (2023). Explaining Grokking Through Circuit Efficiency. arXiv:2309.02390.
Liu et al. (2022). Towards Understanding Grokking. NeurIPS.
```
