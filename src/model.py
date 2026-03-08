"""
src/model.py — Small Transformer for Grokking × LTH Research
=============================================================

Architecture (defaults: d_model=128, n_heads=4, n_layers=2, d_ff=512)
----------------------------------------------------------------------
    Embedding   : vocab_size → d_model  (token + learned positional)
    Body        : N × TransformerEncoderLayer  (Pre-LN, full attention)
    Head        : d_model → n_classes  (applied to last token, no bias)

Parameter count
---------------
    ~500 K at default settings — fits on a single T4 with large batch.
    Enough layers/heads to exhibit meaningful lottery tickets.

Design choices
--------------
    Pre-LayerNorm  : more stable training over tens of thousands of steps.
    No dropout     : standard for grokking studies (dropout suppresses it).
    batch_first    : (B, S, d) convention throughout.
    Full attention : no causal mask — model sees full [a, op, b, =].
    Classification : logits from the representation at the '=' token (pos=3).

References
----------
    Power et al. (2022) arXiv:2201.02177
    Nanda  et al. (2023) arXiv:2301.05217
    Frankle & Carlin (2019) arXiv:1803.03635
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GrokTransformer(nn.Module):
    """
    Two-layer Transformer encoder for algorithmic sequence classification.

    Parameters
    ----------
    vocab_size : Total vocabulary size  (= p + 2 for modular arithmetic).
    n_classes  : Number of output classes  (= p).
    d_model    : Hidden / embedding dimension.
    n_heads    : Number of attention heads  (must divide d_model).
    n_layers   : Number of Transformer encoder layers.
    d_ff       : Inner dimension of feed-forward sublayers.
    dropout    : Dropout probability (0.0 recommended for grokking).
    seq_len    : Input sequence length (4 for [a, op, b, =]).
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        d_model: int = 128,
        n_heads: int  = 4,
        n_layers: int = 2,
        d_ff: int     = 512,
        dropout: float = 0.0,
        seq_len: int   = 4,
    ) -> None:
        super().__init__()

        self.d_model   = d_model
        self.n_classes = n_classes
        self.seq_len   = seq_len

        # ── Embeddings ────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(seq_len,    d_model)

        # ── Transformer body ──────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward= d_ff,
            dropout        = dropout,
            activation     = "relu",
            batch_first    = True,   # convention: (B, S, d)
            norm_first     = True,   # Pre-LN: more stable for long training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # ── Classification head (no bias — cleaner weight analysis) ───────
        self.head = nn.Linear(d_model, n_classes, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Weight initialisation following Power et al.:
            embeddings  : N(0, 0.02)
            head        : Xavier uniform
            attn / ff   : Xavier uniform on 2-D tensors, zeros on biases
        """
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)
        nn.init.xavier_uniform_(self.head.weight)

        for layer in self.transformer.layers:
            for name, param in layer.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : LongTensor (B, 4)  →  [a, op_tok, b, eq_tok]

        Returns
        -------
        logits : FloatTensor (B, n_classes)
        """
        B, S   = x.shape
        pos    = torch.arange(S, device=x.device).unsqueeze(0)   # (1, S)
        emb    = self.token_emb(x) + self.pos_emb(pos)           # (B, S, d)
        out    = self.transformer(emb)                             # (B, S, d)
        logits = self.head(out[:, -1, :])                          # (B, n_classes)
        return logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_embedding_weights(self) -> torch.Tensor:
        """
        Return the token embedding matrix (vocab_size, d_model).
        Rows 0..p-1 are the number tokens used in Fourier analysis.
        """
        return self.token_emb.weight

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return total (trainable) parameter count."""
        return sum(
            p.numel() for p in self.parameters()
            if not trainable_only or p.requires_grad
        )

    def get_prunable_named_parameters(self) -> dict[str, nn.Parameter]:
        """
        Return all weight tensors eligible for magnitude-based pruning.

        Excluded:
            - 1-D parameters (biases)
            - Embedding matrices  (token_emb, pos_emb)
            - LayerNorm parameters (norm1, norm2, final norm)

        These exclusions follow standard LTH practice: embeddings are
        input-facing and LayerNorm scales are near-scalar; neither shows
        meaningful magnitude structure for pruning.
        """
        prunable: dict[str, nn.Parameter] = {}
        exclude_keywords = ("norm", "emb",)   # substring match

        for name, param in self.named_parameters():
            if param.dim() < 2:
                continue
            if any(kw in name for kw in exclude_keywords):
                continue
            prunable[name] = param

        return prunable

    def extra_repr(self) -> str:
        n = self.count_parameters()
        return (
            f"d_model={self.d_model}, n_classes={self.n_classes}, "
            f"params={n:,}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(
    vocab_size: int,
    n_classes: int,
    d_model: int  = 128,
    n_heads: int  = 4,
    n_layers: int = 2,
    d_ff: int     = 512,
    dropout: float = 0.0,
) -> GrokTransformer:
    """Construct the standard GrokTransformer with given hyperparameters."""
    return GrokTransformer(
        vocab_size=vocab_size,
        n_classes=n_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
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
    print(f"Prunable params : {len(model.get_prunable_named_parameters())}")
    print(f"  names : {list(model.get_prunable_named_parameters().keys())}")
