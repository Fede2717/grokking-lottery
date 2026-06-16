"""
src/model.py — Decoder-style Transformer for Grokking × LTH
============================================================

Canonical setup (DEFAULT)
-------------------------
    1-layer decoder-only transformer, d_model=128, 4 heads (d_head=32),
    d_mlp=512, ReLU, NO LayerNorm, learned positional embeddings, untied
    embed/unembed, logits read from the last ("=") token, head has no bias.
    (Nanda 2023; Power 2022; Varma 2023.)

Why a custom block?
-------------------
    ``nn.TransformerEncoderLayer`` ALWAYS contains norm1/norm2, so a faithful
    no-LayerNorm model is impossible with it. This module implements a minimal
    block (multi-head self-attention + ReLU MLP) with a ``layernorm`` mode in
    {"none", "pre", "post"} so the canonical no-LN path and the 2-layer Pre-LN
    variant share one code path. Attention projections are explicit ``nn.Linear``
    layers (q/k/v/out), which gives a clean, module-based pruning surface
    (``nn.MultiheadAttention`` stores ``in_proj_weight`` as a bare parameter).

Pruning surface
---------------
    ``get_prunable_named_parameters`` selects exactly the ``nn.Linear`` weight
    matrices via ``isinstance`` checks — excluding ``nn.Embedding`` and
    ``nn.LayerNorm`` parameters and all 1-D params (biases, norm scales).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_mode(layernorm) -> str:
    """Normalise the config value into {"none", "pre", "post"}."""
    if layernorm is True:
        return "pre"
    if layernorm is False or layernorm is None:
        return "none"
    mode = str(layernorm).lower()
    if mode not in ("none", "pre", "post"):
        raise ValueError(f"layernorm must be one of none/pre/post (or bool); got {layernorm!r}")
    return mode


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Full (bidirectional) multi-head self-attention with explicit q/k/v/out."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        H, dh = self.n_heads, self.d_head

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, S, H, dh).transpose(1, 2)  # (B, H, S, dh)

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)   # (B, H, S, S)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = attn @ v                                       # (B, H, S, dh)
        out = out.transpose(1, 2).reshape(B, S, H * dh)      # (B, S, d_model)
        return self.out_proj(out)


class MLP(nn.Module):
    """Two-layer ReLU feed-forward block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class DecoderBlock(nn.Module):
    """
    Minimal transformer block with selectable LayerNorm placement.

        none : x = x + attn(x);            x = x + mlp(x)            [canonical]
        pre  : x = x + attn(norm1(x));     x = x + mlp(norm2(x))
        post : x = norm1(x + attn(x));     x = norm2(x + mlp(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, layernorm: str):
        super().__init__()
        self.mode = _norm_mode(layernorm)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.mlp = MLP(d_model, d_ff, dropout)
        if self.mode != "none":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.norm1 = self.norm2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "pre":
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        elif self.mode == "post":
            x = self.norm1(x + self.attn(x))
            x = self.norm2(x + self.mlp(x))
        else:  # none
            x = x + self.attn(x)
            x = x + self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GrokTransformer(nn.Module):
    """Decoder-style transformer for algorithmic sequence classification."""

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        d_ff: int = 512,
        dropout: float = 0.0,
        seq_len: int = 4,
        layernorm: str = "none",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.layernorm = _norm_mode(layernorm)

        # Embeddings (token + learned positional).
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        # Transformer body.
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout, self.layernorm)
            for _ in range(n_layers)
        ])

        # Untied unembedding head, no bias (cleaner weight analysis).
        self.head = nn.Linear(d_model, n_classes, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.head:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: LongTensor (B, seq_len) → logits (B, n_classes), read at last token."""
        _, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)   # (1, S)
        h = self.token_emb(x) + self.pos_emb(pos)             # (B, S, d)
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, -1, :])                         # (B, n_classes)

    # ------------------------------------------------------------------

    def get_embedding_weights(self) -> torch.Tensor:
        """Token embedding matrix (vocab_size, d_model). Rows 0..p-1 = numbers."""
        return self.token_emb.weight

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)

    def get_prunable_named_parameters(self) -> dict[str, nn.Parameter]:
        """
        Weight tensors eligible for magnitude pruning: exactly the ``nn.Linear``
        weight matrices, selected by ``isinstance``.

        Excluded by construction: ``nn.Embedding`` and ``nn.LayerNorm`` params
        (not nn.Linear) and all 1-D params (biases, norm scales — a Linear's
        ``.bias`` is never added; only its 2-D ``.weight`` is).
        """
        prunable: dict[str, nn.Parameter] = {}
        for mod_name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                name = f"{mod_name}.weight" if mod_name else "weight"
                prunable[name] = module.weight
        return prunable

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_classes={self.n_classes}, "
            f"layernorm={self.layernorm!r}, params={self.count_parameters():,}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(
    vocab_size: int,
    n_classes: int,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 1,
    d_ff: int = 512,
    dropout: float = 0.0,
    seq_len: int = 4,
    layernorm: str = "none",
) -> GrokTransformer:
    """Construct the GrokTransformer (canonical defaults: 1 layer, no LayerNorm)."""
    return GrokTransformer(
        vocab_size=vocab_size,
        n_classes=n_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        seq_len=seq_len,
        layernorm=layernorm,
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = 97
    model = get_model(vocab_size=p + 2, n_classes=p)
    print(model)
    x = torch.randint(0, p + 2, (8, 4))
    logits = model(x)
    print(f"Input  : {x.shape}")
    print(f"Logits : {logits.shape}")
    prunable = model.get_prunable_named_parameters()
    print(f"Prunable params : {len(prunable)}")
    print(f"  names : {list(prunable.keys())}")
    n_ln = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    print(f"LayerNorm modules (canonical=0) : {n_ln}")
