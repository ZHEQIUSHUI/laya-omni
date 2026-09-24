"""Without image or audio, laya-omni must be Laya exactly. Runs on CPU with a tiny random model."""

import copy

import pytest
import torch
from transformers import ModernBertConfig, ModernBertModel

from laya_omni.model import DecisionModel, OmniFusion

IN_DIMS = {"image": 48, "audio": 40}


def tiny_models(seed=0):
    torch.manual_seed(seed)
    cfg = ModernBertConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
        num_attention_heads=2, pad_token_id=0, max_position_embeddings=256,
    )
    cfg._attn_implementation = "sdpa"
    base = DecisionModel(ModernBertModel(cfg), {"head_layers": 2}).eval()
    omni = copy.deepcopy(base)
    omni.fusion = OmniFusion(128, 2, IN_DIMS, num_latents=8, depth=2).eval()
    return base, omni


def batch(n=3, length=24):
    g = torch.Generator().manual_seed(1)
    attn = torch.ones(n, length, dtype=torch.long)
    attn[1, 18:] = 0
    return {
        "input_ids": torch.randint(1, 128, (n, length), generator=g),
        "attention_mask": attn,
        "marker_pos": torch.tensor([[3, 6, 9, 0], [3, 6, 0, 0], [3, 6, 9, 12]]),
        "marker_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool),
        "qtype": torch.tensor([0, 2, 1]),
    }


def feats(n=3):
    g = torch.Generator().manual_seed(2)
    return {
        "image": (torch.randn(n, 16, IN_DIMS["image"], generator=g), None, None),
        "audio": (torch.randn(n, 20, IN_DIMS["audio"], generator=g), torch.ones(n, 20, dtype=torch.bool), None),
    }


def open_gates(model):
    with torch.no_grad():
        for block in model.fusion.blocks:
            block.attn_gate.fill_(0.5)
            block.ff_gate.fill_(0.5)


@torch.no_grad()
def test_no_modality_is_bit_identical():
    base, omni = tiny_models()
    open_gates(omni)
    b = batch()
    for mods in (None, {}):
        for got, want in zip(omni(**b, modalities=mods), base(**b)):
            assert torch.equal(got, want)


@torch.no_grad()
def test_zero_gates_start_from_laya():
    base, omni = tiny_models()
    b = batch()
    for got, want in zip(omni(**b, modalities=feats()), base(**b)):
        assert torch.equal(got, want)


@torch.no_grad()
def test_rows_without_modality_match_laya_in_a_mixed_batch():
    base, omni = tiny_models()
    open_gates(omni)
    b, m = batch(), feats()
    present = torch.tensor([True, False, True])  # row 1 has neither image nor audio
    m = {k: (f, mask, present) for k, (f, mask, _) in m.items()}
    got, want = omni(**b, modalities=m), base(**b)
    assert torch.equal(got[0][1], want[0][1]) and torch.equal(got[1][1], want[1][1])
    assert not torch.allclose(got[0][0], want[0][0], atol=1e-5)


@pytest.mark.parametrize("names", [("image",), ("audio",), ("image", "audio")])
def test_each_modality_is_optional_and_changes_output(names):
    base, omni = tiny_models()
    open_gates(omni)
    b, m = batch(), feats()
    logits, _ = omni(**b, modalities={k: m[k] for k in names})
    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits, base(**b)[0])


def test_only_fusion_trains():
    _, omni = tiny_models()
    for name, p in omni.named_parameters():
        p.requires_grad_(name.startswith("fusion."))
    logits, act = omni(**batch(), modalities=feats())
    (logits.logsumexp(-1).sum() + act.sum()).backward()
    grads = {n for n, p in omni.named_parameters() if p.grad is not None}
    assert grads and all(n.startswith("fusion.") for n in grads)
    # Zero gates must still pass gradient to themselves, or training never starts.
    assert omni.fusion.blocks[0].attn_gate.grad.abs().sum() > 0
