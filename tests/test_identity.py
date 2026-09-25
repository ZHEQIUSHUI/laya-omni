"""Without image or audio, laya-omni must be Laya exactly. Runs on CPU with a tiny random model."""

import copy

import pytest
import torch
from transformers import ModernBertConfig, ModernBertModel

from laya_omni.model import DecisionModel, OmniFusion

IN_DIMS = {"image": 48, "audio": 40}


def tiny_models(seed=0, lora=0):
    torch.manual_seed(seed)
    cfg = ModernBertConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
        num_attention_heads=2, pad_token_id=0, max_position_embeddings=256,
    )
    cfg._attn_implementation = "sdpa"
    base = DecisionModel(ModernBertModel(cfg), {"head_layers": 2}).eval()
    omni = copy.deepcopy(base)
    omni.fusion = OmniFusion(128, IN_DIMS, lora_rank=lora).eval()
    if lora:  # give the zero-initialised deltas something to do
        with torch.no_grad():
            for m in omni.fusion.lora.values():
                m.B.normal_(std=0.5)
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
    audio_mask = torch.ones(n, 20, dtype=torch.bool)
    audio_mask[0, 12:] = False  # a shorter clip, padded
    return {
        "image": (torch.randn(n, 16, IN_DIMS["image"], generator=g), None, None),
        "audio": (torch.randn(n, 20, IN_DIMS["audio"], generator=g), audio_mask, None),
    }


@pytest.mark.parametrize("lora", [0, 4])
@torch.no_grad()
def test_no_modality_is_bit_identical(lora):
    base, omni = tiny_models(lora=lora)
    b = batch()
    for mods in (None, {}):
        for got, want in zip(omni(**b, modalities=mods), base(**b)):
            assert torch.equal(got, want)


@pytest.mark.parametrize("lora", [0, 4])
@torch.no_grad()
def test_rows_without_modality_match_laya_in_a_mixed_batch(lora):
    base, omni = tiny_models(lora=lora)
    b, m = batch(), feats()
    present = torch.tensor([True, False, True])  # row 1 has neither image nor audio
    m = {k: (f, mask, present) for k, (f, mask, _) in m.items()}
    got, want = omni(**b, modalities=m), base(**b)
    # Same tokens at the same positions, only extra padding: equal up to kernel rounding.
    assert torch.allclose(got[0][1], want[0][1], atol=1e-5) and torch.allclose(got[1][1], want[1][1], atol=1e-5)
    assert not torch.allclose(got[0][0], want[0][0], atol=1e-3)


@torch.no_grad()
def test_splice_keeps_text_positions_and_moves_markers():
    _, omni = tiny_models()
    b, m = batch(), feats()
    present = torch.tensor([True, False, True])
    m = {k: (f, mask, present) for k, (f, mask, _) in m.items()}
    embeds, att, markers = omni.splice(b["input_ids"], b["attention_mask"], b["marker_pos"], m)
    text = omni.encoder.embeddings.tok_embeddings(b["input_ids"])
    n0 = 16 + 12  # row 0: 16 image tokens and a 12-frame clip
    assert torch.equal(markers[0], b["marker_pos"][0] + n0)
    assert torch.equal(embeds[0, 1 + n0 : 1 + n0 + 23], text[0, 1:])
    assert torch.equal(markers[1], b["marker_pos"][1])  # nothing inserted in row 1
    assert torch.equal(embeds[1, :24], text[1]) and att[1, 18:].sum() == 0
    assert int(att[0].sum()) == 24 + n0


@pytest.mark.parametrize("names", [("image",), ("audio",), ("image", "audio")])
@torch.no_grad()
def test_each_modality_is_optional_and_changes_output(names):
    base, omni = tiny_models()
    b, m = batch(), feats()
    logits, _ = omni(**b, modalities={k: m[k] for k in names})
    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits, base(**b)[0], atol=1e-3)


@torch.no_grad()
def test_multiple_images_per_row():
    base, omni = tiny_models()
    b = batch()
    g = torch.Generator().manual_seed(3)
    imgs = torch.randn(3, 2, 16, IN_DIMS["image"], generator=g)
    two = torch.tensor([[True, True], [True, False], [False, False]])  # 2 images, 1 image, none
    logits, _ = omni(**b, modalities={"image": (imgs, None, two)})
    assert torch.allclose(logits[2], base(**b)[0][2], atol=1e-5)
    # Row 1's missing second image must not matter: same as giving it one image.
    single, _ = omni(**b, modalities={"image": (imgs[:, 0], None, two[:, 0])})
    assert torch.allclose(logits[1], single[1], atol=1e-5)
    # Item embeddings tell the images apart: swapping row 0's two images changes the answer.
    swapped, _ = omni(**b, modalities={"image": (imgs.flip(1), None, two)})
    assert not torch.allclose(swapped[0], logits[0], atol=1e-4)


def test_only_fusion_trains():
    _, omni = tiny_models()
    for name, p in omni.named_parameters():
        p.requires_grad_(name.startswith("fusion."))
    logits, act = omni(**batch(), modalities=feats())
    (logits.logsumexp(-1).sum() + act.sum()).backward()
    grads = {n for n, p in omni.named_parameters() if p.grad is not None}
    assert grads and all(n.startswith("fusion.") for n in grads)
    assert omni.fusion.projectors["image"][1].weight.grad.abs().sum() > 0


def test_lora_hooks_the_encoder_and_head_and_trains():
    _, omni = tiny_models(lora=4)
    names = omni.fusion.lora_layers
    assert any(n.startswith("encoder.") for n in names) and any(n.startswith("head.") for n in names)
    assert omni.fusion.row_mask is None
    # Laya's own parameter names are untouched: its checkpoint still loads strictly.
    assert not any("lora" in n for n, _ in omni.named_parameters() if not n.startswith("fusion."))
    for name, p in omni.named_parameters():
        p.requires_grad_(name.startswith("fusion."))
    logits, _ = omni(**batch(), modalities=feats())
    logits.logsumexp(-1).sum().backward()
    assert all(m.A.grad is not None and m.A.grad.abs().sum() > 0 for m in omni.fusion.lora.values())
    assert omni.fusion.row_mask is None  # cleared after the forward


@torch.no_grad()
def test_image_grid_positions_pool_like_the_features():
    torch.manual_seed(0)
    fusion = OmniFusion(128, IN_DIMS, image_grid=4)
    full, pooled = fusion.positions("image", 16), fusion.positions("image", 4)
    grid = full.T.reshape(128, 4, 4)
    assert torch.allclose(pooled, torch.nn.functional.avg_pool2d(grid[None], 2)[0].flatten(1).T)
    assert torch.equal(fusion.positions("audio", 5), fusion.frame_emb[:5])  # audio keeps frame indices
