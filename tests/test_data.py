"""Dataset: disjoint reproducible split with correct sizes."""
import torch

from src.data import ModularArithmeticDataset, get_dataloaders


def _pairs(ds):
    return {(a, b) for a, b, _ in ds.data}


def test_train_val_disjoint_and_complete():
    p, frac = 17, 0.5
    train = ModularArithmeticDataset(p=p, operation="add", split="train", train_frac=frac, seed=1)
    val = ModularArithmeticDataset(p=p, operation="add", split="val", train_frac=frac, seed=1)
    tp, vp = _pairs(train), _pairs(val)
    assert tp.isdisjoint(vp)                       # no leakage
    assert len(tp) + len(vp) == p * p              # full enumeration
    assert tp | vp == {(a, b) for a in range(p) for b in range(p)}


def test_split_sizes():
    p, frac = 17, 0.5
    train = ModularArithmeticDataset(p=p, split="train", train_frac=frac, seed=1)
    assert len(train) == int(p * p * frac)


def test_reproducible():
    a = ModularArithmeticDataset(p=11, split="train", seed=7).data
    b = ModularArithmeticDataset(p=11, split="train", seed=7).data
    assert a == b
    c = ModularArithmeticDataset(p=11, split="train", seed=8).data
    assert a != c                                  # different seed -> different split


def test_labels_correct():
    ds = ModularArithmeticDataset(p=11, operation="add", split="train", seed=0)
    x, y = ds[0]
    a, op, b, eq = x.tolist()
    assert op == ds.op_token and eq == ds.eq_token
    assert y.item() == (a + b) % 11


def test_full_batch_loader_is_single_batch():
    tl, vl = get_dataloaders(p=11, full_batch=True, seed=0)
    batches = list(tl)
    assert len(batches) == 1                       # full-batch GD => one batch/step
    xb, yb = batches[0]
    assert xb.shape[1] == 4 and xb.dtype == torch.long
