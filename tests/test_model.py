"""Model forward shapes, no-LayerNorm default, and prunable selection."""
import torch
import torch.nn as nn

from src.model import get_model


def test_forward_shape():
    m = get_model(vocab_size=99, n_classes=97, n_layers=1, layernorm="none")
    out = m(torch.randint(0, 99, (8, 4)))
    assert out.shape == (8, 97)


def test_canonical_has_no_layernorm():
    m = get_model(vocab_size=99, n_classes=97, n_layers=1, layernorm="none")
    assert sum(isinstance(mod, nn.LayerNorm) for mod in m.modules()) == 0


def test_preln_variant_has_layernorm():
    m = get_model(vocab_size=99, n_classes=97, n_layers=2, layernorm="pre")
    assert sum(isinstance(mod, nn.LayerNorm) for mod in m.modules()) == 4  # 2 per block


def test_prunable_selection_isinstance():
    m = get_model(vocab_size=99, n_classes=97, n_layers=2, layernorm="pre")
    prunable = m.get_prunable_named_parameters()
    named = dict(m.named_parameters())

    # Every prunable entry is an existing 2-D Linear weight.
    for name, param in prunable.items():
        assert name in named and param.dim() == 2
        assert "bias" not in name           # no 1-D params
        assert "emb" not in name            # no embeddings
        assert "norm" not in name           # no LayerNorm

    # Count matches the number of nn.Linear modules.
    n_linear = sum(isinstance(mod, nn.Linear) for mod in m.modules())
    assert len(prunable) == n_linear

    # Embeddings and LayerNorm params exist but are excluded.
    assert any("emb" in n for n in named)
    assert any("norm" in n for n in named)
    assert not any("emb" in n or "norm" in n for n in prunable)


def test_untied_head_no_bias():
    m = get_model(vocab_size=99, n_classes=97)
    assert m.head.bias is None
    assert m.head.weight is not m.token_emb.weight   # untied unembedding


def test_logits_read_from_last_token():
    m = get_model(vocab_size=99, n_classes=97)
    x = torch.randint(0, 99, (3, 4))
    # Changing only the last token should change the logits (it's the read position).
    out1 = m(x)
    x2 = x.clone(); x2[:, -1] = (x2[:, -1] + 1) % 99
    out2 = m(x2)
    assert not torch.allclose(out1, out2)
