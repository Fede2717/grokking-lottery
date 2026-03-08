"""
src/data.py — Modular Arithmetic Dataset for Grokking × LTH Research
======================================================================

Implements the canonical grokking dataset from Power et al. (2022):
    Task : predict (a OP b) mod p, for a, b in {0, ..., p-1}
    Split: 50 % train / 50 % val  (shuffled, reproducible)

Vocabulary layout
-----------------
    0  ..  p-1  : number tokens
    p            : operation token  (e.g. '+')
    p+1          : equals token     ('=')
    Total vocab  : p + 2  (99 at default p = 97)

Each sample
-----------
    x : LongTensor (4,)   [a, op_tok, b, eq_tok]
    y : LongTensor scalar   (a OP b) mod p

References
----------
    Power et al. (2022) "Grokking: Generalization Beyond Overfitting on
    Small Algorithmic Datasets." arXiv:2201.02177
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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

        # --- Generate full dataset (all p² pairs) -------------------------
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

    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    p: int = 97,
    operation: str = "add",
    train_frac: float = 0.5,
    batch_size: int = 512,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

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

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(pin_memory and cuda),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(pin_memory and cuda),
        drop_last=False,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(p=97, batch_size=512)
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    x, y = next(iter(train_loader))
    print(f"x shape : {x.shape}  (batch, seq_len=4)")
    print(f"y shape : {y.shape}  (batch,)")
    print(f"Sample  : {x[0].tolist()}  →  {y[0].item()}")
