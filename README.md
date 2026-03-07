# Grokking × Lottery Ticket Hypothesis
### *Can a sparse sub-network still have an "aha" moment?*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-orange.svg)](https://pytorch.org)
[![Hydra 1.3](https://img.shields.io/badge/config-Hydra_1.3-89b4fa.svg)](https://hydra.cc)
[![W&B](https://img.shields.io/badge/tracking-W%26B-yellow.svg)](https://wandb.ai)
[![uv](https://img.shields.io/badge/package_manager-uv-7c3aed.svg)](https://github.com/astral-sh/uv)

---

## The Paradox

Two of the most fascinating phenomena in deep learning point in opposite directions:

**The Lottery Ticket Hypothesis** (Frankle & Carlin, 2019) tells us that dense networks are wasteful — hidden inside is a small, sparse sub-network that can match or beat the full model when trained from its original initialisation.

**Grokking** (Power et al., 2022) tells us that if you train a dense network *far* beyond overfitting — for tens of thousands of steps after it has already memorised training data — it suddenly and dramatically generalises. A phase transition, not gradient descent.

> **The research question no one has cleanly answered: Does grokking require overparameterisation? Can the lottery ticket — the small, efficient sub-network — still grok? Or does pruning destroy the architectural substrate that makes the phase transition possible?**

---

## Key Finding *(update after experiments)*

> *"Moderate pruning (≤ 70% sparsity) does not impair grokking — it accelerates it. Above 90% sparsity, the phase transition disappears entirely, defining a critical threshold. The surviving weights are precisely the Fourier features responsible for generalisation."*

---

## Repository Structure

```
grokking-lottery/
├── pyproject.toml                  ← uv-managed, fully reproducible deps
│
├── configs/                        ← Hydra configuration hierarchy
│   ├── config.yaml                 ← Root config (seed, W&B, checkpoint policy)
│   ├── dataset/modular_add.yaml    ← p=97, operation=add, batch=512
│   ├── model/small_transformer.yaml← d=128, 2L, 4H, ~500K params
│   ├── training/default.yaml       ← lr, weight_decay, n_grok_steps
│   ├── pruning/imp.yaml            ← target_sparsities, prune_rate, IMP schedule
│   └── experiment/{exp_a,b,c}.yaml ← per-experiment overrides and plot flags
│
├── src/
│   ├── data.py        Modular arithmetic dataset: all p² pairs, reproducible split
│   ├── model.py       GrokTransformer: Pre-LN, ~500K params, Fourier-probeable
│   ├── train.py       Trainer + W&B logging + strict disk checkpointing
│   ├── prune.py       IMP (disk rewind) + one-shot pruning (ablation)
│   └── metrics.py     Fourier analysis, GSNR, Hessian λ_max, effective rank
│
├── experiments/
│   ├── config.py                      Legacy dataclass config (kept for reference)
│   ├── exp_b_lth_then_grok.py         PRIMARY: LTH → Grok, multi-seed, CI plots
│   ├── exp_a_grok_then_prune.py       CONTROL: Grok → Prune → Rewind comparison
│   └── exp_c_wd_ablation.py           ABLATION: 4×4 WD × sparsity grid
│
├── scripts/
│   ├── run_parallel_seeds.py          GPU-parallel seed launcher (Python)
│   └── run_all.sh                     Sequential full-suite runner
│
└── results/
    ├── figures/                        Publication-ready plots
    └── exp_*/                          Per-run JSON histories and .pt checkpoints
```

---

## How to Reproduce on Kaggle (with W&B)

### Step 1 — Clone the repository

In a Kaggle notebook cell:
```bash
!git clone https://github.com/YOUR_USERNAME/grokking-lottery.git
%cd grokking-lottery
```

### Step 2 — Install `uv` and sync dependencies

`uv` creates a fully reproducible virtual environment from `pyproject.toml` in a single command — no `requirements.txt` drift, no pip version conflicts.

```bash
# Install uv (fast Rust-based package manager)
!curl -LsSf https://astral.sh/uv/install.sh | sh
!source $HOME/.cargo/env   # or restart shell

# Sync exact environment from pyproject.toml
!uv sync

# For CUDA 12.x on Kaggle T4, use the PyTorch CUDA wheel:
!uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
```

Alternatively, with standard pip:
```bash
!pip install -e ".[dev]" --quiet
```

### Step 3 — Authenticate with Weights & Biases

**Option A — API key in cell (recommended for Kaggle):**
```python
import wandb
wandb.login(key="YOUR_WANDB_API_KEY_HERE")
# Get your key at: https://wandb.ai/authorize
```

**Option B — Kaggle Secrets (more secure):**
```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
wandb_key = secrets.get_secret("WANDB_API_KEY")

import wandb
wandb.login(key=wandb_key)
```

**Option C — Environment variable:**
```bash
export WANDB_API_KEY="your_key_here"
```

Then confirm W&B is enabled in `configs/config.yaml`:
```yaml
wandb:
  enabled: true
  project: "grokking-lottery"
  entity:  "your-wandb-username"   # ← set this
```

### Step 4 — Run with parallel seed launcher (2× T4)

The launcher distributes seeds across both GPUs concurrently using Python threads. No DDP, no NCCL, no shared memory — each process is fully independent.

```bash
# Primary experiment: IMP → Grok (5 seeds, 2 GPUs)
!python scripts/run_parallel_seeds.py \
    --experiment exp_b \
    --num-seeds 5 \
    --num-gpus 2

# GPU assignment (automatic round-robin):
#   GPU-0: seeds 0, 2, 4  (sequential)
#   GPU-1: seeds 1, 3     (sequential)
# Wall-clock ≈ max(time_GPU0, time_GPU1)
```

**Debug run first** (confirm everything works, ~15 minutes):
```bash
!python scripts/run_parallel_seeds.py \
    --experiment exp_b \
    --debug \
    --num-gpus 2
# Runs 2 seeds, 5k steps, 3 sparsities
```

**Full suite** (all 3 experiments, ~6-8 hours on 2× T4):
```bash
!python scripts/run_parallel_seeds.py \
    --experiment all \
    --num-seeds 5 \
    --num-gpus 2
```

### Step 5 — Hydra config overrides

```bash
# Change weight decay (Exp C sweep)
!python -m experiments.exp_b_lth_then_grok training.weight_decay=5e-3

# Custom sparsity levels
!python -m experiments.exp_b_lth_then_grok \
    "pruning.target_sparsities=[0.0,0.5,0.8,0.9,0.95]"

# Run modular multiplication instead of addition (harder task)
!python -m experiments.exp_b_lth_then_grok dataset=modular_mul
```

### Step 6 — View results on W&B

After running, your results appear at `https://wandb.ai/YOUR_USERNAME/grokking-lottery`.

Key dashboards to create:
- **val/acc vs step** (group by sparsity) — grokking curves
- **event/grokking_step** (group by sparsity) — the efficiency frontier
- **weights/l2 vs step** — weight norm collapse
- **fourier/concentration vs step** — Fourier feature emergence
- **hessian/top_eigenvalue vs step** — landscape sharpness

### Step 7 — View local figures

All publication-ready figures are saved to `results/figures/`:

| File | Description |
|---|---|
| `exp_b_grokking_curves_imp.png` | Grokking curves (mean ± 95% CI, all sparsities) |
| `exp_b_efficiency_frontier.png` | Sparsity vs S_G scatter — the hero image |
| `exp_b_weight_norm_collapse.png` | L2 norm collapse at grokking moment |
| `exp_b_fourier_checkpoints.png` | Fourier spectra: memorised → grokked |
| `exp_a_rewind_comparison.png` | Rewind strategy comparison |
| `exp_a_post_grok_accuracy.png` | Circuit survival under pruning |
| `exp_c_heatmap_sg.png` | 2D grid: WD × sparsity |

---

## Experiments

### Experiment B — PRIMARY: LTH First, Then Grok

For each sparsity `s ∈ {0%, 20%, 50%, 70%, 80%, 90%, 95%}`:
1. Save `W_0` to disk (strict init checkpoint)
2. IMP: prune 20%/round → rewind to `W_0` → repeat until sparsity = `s`
3. Long grokking phase: 50k steps, full metrics, W&B logging
4. Repeat for 5 seeds; plot mean ± 95% CI

### Experiment A — CONTROL: Grok First, Then Prune

1. Train dense model to grokking; save `W_init`, `W_mem`, `W_grok`
2. Prune `W_grok` at each sparsity — *immediate* accuracy test (circuit survival)
3. Rewind to `W_init` vs `W_mem` — compare S_G across rewind strategies

### Experiment C — ABLATION: Weight Decay × Sparsity Grid

4×4 grid: `λ ∈ {1e-4, 1e-3, 5e-3, 1e-2}` × `sparsity ∈ {0%, 50%, 80%, 90%}`.
Tests whether weight decay and pruning are *substitutes* or *complements*.

---

## Metrics Tracked

| Metric | Type | W&B key |
|---|---|---|
| Steps to grok S_G | Primary | `event/grokking_step` |
| Grokking gap G_gap | Primary | `event/grokking_gap` |
| ‖W‖₂ at grokking | Primary | `event/l2_at_grok` |
| Fourier concentration | Mechanistic | `fourier/concentration` |
| Spectral entropy | Mechanistic | `fourier/spectral_entropy` |
| Hessian λ_max | Sharpness | `hessian/top_eigenvalue` |
| Global L2/L1 norm | Weight dynamics | `weights/l2`, `weights/l1` |
| Sparsity ratio | Efficiency | `sparsity` |
| Gradient signal-to-noise | Training quality | computed in `metrics.py` |

---

## References

```
Frankle & Carlin (2019). The Lottery Ticket Hypothesis. ICLR 2019.
Power et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177.
Nanda et al. (2023). Progress Measures for Grokking via Mechanistic Interpretability. ICLR 2023.
Varma et al. (2023). Explaining Grokking Through Circuit Efficiency. arXiv:2309.02390.
Liu et al. (2022). Towards Understanding Grokking. NeurIPS 2022.
Frankle et al. (2020). Linear Mode Connectivity and the Lottery Ticket Hypothesis. ICML 2020.
```

---

## Citation

```bibtex
@misc{groklth2024,
  title  = {Grokking × Lottery Ticket Hypothesis: Can a Sparse Sub-network Grok?},
  author = {[Your Name]},
  year   = {2024},
  url    = {https://github.com/[your-handle]/grokking-lottery}
}
```
