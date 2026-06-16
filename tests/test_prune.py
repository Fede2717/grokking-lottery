"""Pruning: achieved sparsity, mask persistence after a step, exact rewind."""
import tempfile
from pathlib import Path

import pytest
import torch

from src.model import get_model
from src.prune import (
    apply_global_magnitude_pruning, apply_masks, compute_sparsity,
    make_empty_masks, one_shot_prune, rewind_weights,
)
from src.train import make_optimizer, save_init_checkpoint


def _train_a_bit(model):
    opt = make_optimizer(model, weight_decay=0.0, lr=1e-2)
    crit = torch.nn.CrossEntropyLoss()
    x = torch.randint(0, 99, (64, 4)); y = torch.randint(0, 97, (64,))
    for _ in range(15):
        opt.zero_grad(); crit(model(x), y).backward(); opt.step()


@pytest.mark.parametrize("target", [0.2, 0.5, 0.8, 0.95])
def test_achieved_sparsity_matches_target(target):
    torch.manual_seed(0)
    m = get_model(vocab_size=99, n_classes=97)
    _train_a_bit(m)
    masks = apply_global_magnitude_pruning(m, make_empty_masks(m), target)
    assert abs(compute_sparsity(masks) - target) < 0.01


def test_masked_weights_stay_zero_after_step():
    torch.manual_seed(0)
    m = get_model(vocab_size=99, n_classes=97)
    _train_a_bit(m)
    masks = apply_global_magnitude_pruning(m, make_empty_masks(m), 0.7)

    opt = make_optimizer(m, weight_decay=1.0, lr=1e-2)
    crit = torch.nn.CrossEntropyLoss()
    x = torch.randint(0, 99, (64, 4)); y = torch.randint(0, 97, (64,))
    opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    apply_masks(m, masks)

    for name, w in m.get_prunable_named_parameters().items():
        assert torch.all(w.detach()[~masks[name].bool()] == 0)


def test_rewind_restores_w0_on_unmasked():
    torch.manual_seed(0)
    m = get_model(vocab_size=99, n_classes=97)
    rd = Path(tempfile.mkdtemp())
    init = save_init_checkpoint(m, rd)
    w0 = {k: v.detach().clone() for k, v in m.get_prunable_named_parameters().items()}

    _train_a_bit(m)
    masks = one_shot_prune(m, init, 0.8)   # ranks on trained weights, rewinds to W0

    for name, w in m.get_prunable_named_parameters().items():
        keep = masks[name].bool()
        assert torch.equal(w.detach()[keep], w0[name][keep])     # exact on unmasked
        assert torch.all(w.detach()[~keep] == 0)                 # pruned == 0
