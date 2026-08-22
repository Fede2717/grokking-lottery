"""Modular arithmetic datasets for the grokking experiments.

The dataset enumerates all ordered pairs ``(a, b)`` and predicts
``(a OP b) mod p`` from ``[a, OP, b, =]``. Number tokens occupy ``0..p-1``;
the operation and equals tokens are ``p`` and ``p+1``. A seeded permutation is
split into training and validation subsets.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Dataset

class ModularArithmeticDataset(Dataset):
    """
    Full enumeration of (a OP b) mod p.

    Parameters
    ----------
    p          : Modulus.  p=97 gives 9,409 total examples.
    operation  : One of {"add", "sub", "mul"}.
    split      : "train" or "val".
    train_frac : Fraction of data used for training (default 0.5).
    seed       : Random seed for reproducible shuffling.
    """

    # Add new modular operations here and in configs/dataset/.
    OPERATIONS: dict[str, callable] = {
        "add": lambda a, b, p: (a + b) % p,
        "sub": lambda a, b, p: (a - b) % p,
        "mul": lambda a, b, p: (a * b) % p,
    }

    def __init__(
        self,
        p: int = 97,
        operation: str = "add",
        split: str = "train",
        train_frac: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        assert operation in self.OPERATIONS, f"Unknown operation: {operation!r}"
        assert split in ("train", "val"),    f"split must be 'train' or 'val'"

        self.p          = p
        self.operation  = operation
        self.op_fn      = self.OPERATIONS[operation]
        self.vocab_size = p + 2        # numbers + op + eq
        self.op_token   = p
        self.eq_token   = p + 1
        self.n_classes  = p            # output classes = {0 .. p-1}

        # Generate full dataset (all p² pairs)
        pairs: list[tuple[int, int, int]] = [
            (a, b, self.op_fn(a, b, p))
            for a in range(p)
            for b in range(p)
        ]

        # Reproducible shuffle
        rng   = np.random.RandomState(seed)
        order = rng.permutation(len(pairs))
        pairs = [pairs[i] for i in order]

        # Train / val split
        n_train   = int(len(pairs) * train_frac)
        self.data = pairs[:n_train] if split == "train" else pairs[n_train:]


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        a, b, result = self.data[idx]
        x = torch.tensor([a, self.op_token, b, self.eq_token], dtype=torch.long)
        y = torch.tensor(result, dtype=torch.long)
        return x, y

    def __repr__(self) -> str:
        return (
            f"ModularArithmeticDataset("
            f"p={self.p}, op={self.operation!r}, "
            f"n={len(self)}, vocab={self.vocab_size})"
        )


# DataLoader factory

def get_dataloaders(
    p: int = 97,
    operation: str = "add",
    train_frac: float = 0.5,
    batch_size: int = 512,
    seed: int = 42,
    full_batch: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Parameters
    ----------
    full_batch : When True, the train loader yields the full training set as one
                 batch per step. ``batch_size`` is then ignored for
                 the train loader. The val loader always evaluates in one batch.

    Returns
    -------
    (train_loader, val_loader)
    """
    cuda = torch.cuda.is_available()

    train_ds = ModularArithmeticDataset(
        p=p, operation=operation, split="train",
        train_frac=train_frac, seed=seed,
    )
    val_ds = ModularArithmeticDataset(
        p=p, operation=operation, split="val",
        train_frac=train_frac, seed=seed,
    )

    train_bs = len(train_ds) if full_batch else batch_size

    train_loader = DataLoader(
        train_ds, batch_size=train_bs, shuffle=not full_batch,
        num_workers=num_workers, pin_memory=(pin_memory and cuda),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=len(val_ds), shuffle=False,
        num_workers=num_workers, pin_memory=(pin_memory and cuda),
        drop_last=False,
    )

    return train_loader, val_loader


# Quick sanity check

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(p=97, batch_size=512)
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    x, y = next(iter(train_loader))
    print(f"x shape : {x.shape}  (batch, seq_len=4)")
    print(f"y shape : {y.shape}  (batch,)")
    print(f"Sample  : {x[0].tolist()}  →  {y[0].item()}")
