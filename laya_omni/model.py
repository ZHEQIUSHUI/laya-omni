"""Laya's DecisionModel in PyTorch, plus optional image / audio fusion.

The text path reproduces upstream Laya (see NOTICE): a ModernBERT encoder, a
question-type embedding, a small pre-norm transformer decision head, a marker
scorer and an act head. Parameter names match the upstream checkpoint, so its
model.safetensors loads strictly without conversion.

Fusion adds, per modality, a Perceiver resampler that turns frozen encoder
features into a fixed number of tokens, and after every decision-head layer a
Flamingo-style gated cross-attention block. All gates start at zero. When no
modality is given the fusion blocks are skipped entirely, so the output is the
original Laya output bit for bit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadLayer(nn.Module):
    """Pre-norm TransformerEncoderLayer with ReLU, named like the upstream checkpoint."""

    def __init__(self, dims):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dims, max(1, dims // 64), batch_first=True)
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.linear1 = nn.Linear(dims, 4 * dims)
        self.linear2 = nn.Linear(4 * dims, dims)

    def forward(self, x, key_padding_mask):
        y = self.norm1(x)
        x = x + self.self_attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class DecisionHead(nn.Module):
    def __init__(self, dims, count):
        super().__init__()
        self.layers = nn.ModuleList(HeadLayer(dims) for _ in range(count))


class FeedForward(nn.Sequential):
    def __init__(self, dims, mult=4):
        super().__init__(nn.LayerNorm(dims), nn.Linear(dims, mult * dims), nn.GELU(), nn.Linear(mult * dims, dims))


class Resampler(nn.Module):
    """Perceiver resampler: a variable number of feature frames -> `num_latents` tokens."""

    def __init__(self, in_dims, dims, num_latents=32, depth=2, max_frames=4096):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dims), nn.Linear(in_dims, dims))
        self.pos = nn.Parameter(torch.zeros(max_frames, dims))
        self.latents = nn.Parameter(torch.randn(num_latents, dims) * 0.02)
        heads = max(1, dims // 64)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "norm_x": nn.LayerNorm(dims),
                    "norm_l": nn.LayerNorm(dims),
                    "attn": nn.MultiheadAttention(dims, heads, batch_first=True),
                    "ff": FeedForward(dims),
                }
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dims)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, feats, mask=None):
        """feats: (B, T, in_dims); mask: (B, T) bool, True = valid frame."""
        x = self.proj(feats) + self.pos[: feats.shape[1]]
        lat = self.latents.expand(feats.shape[0], -1, -1)
        pad = None
        if mask is not None:
            # Latents attend to themselves too, so a row is never fully masked.
            pad = torch.cat([~mask, mask.new_zeros(lat.shape[:2])], dim=1)
        for layer in self.layers:
            kv = torch.cat([layer["norm_x"](x), layer["norm_l"](lat)], dim=1)
            lat = lat + layer["attn"](layer["norm_l"](lat), kv, kv, key_padding_mask=pad, need_weights=False)[0]
            lat = lat + layer["ff"](lat)
        return self.norm(lat)


class GatedCrossAttention(nn.Module):
    """Text tokens read modality tokens; tanh gates start at 0 so training starts from Laya."""

    def __init__(self, dims):
        super().__init__()
        self.norm = nn.LayerNorm(dims)
        self.attn = nn.MultiheadAttention(dims, max(1, dims // 64), batch_first=True)
        self.ff = FeedForward(dims)
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, memory, memory_pad):
        y = self.attn(self.norm(x), memory, memory, key_padding_mask=memory_pad, need_weights=False)[0]
        x = x + torch.tanh(self.attn_gate) * y
        return x + torch.tanh(self.ff_gate) * self.ff(x)


MODALITIES = ("image", "audio")


class OmniFusion(nn.Module):
    """All trainable multimodal parameters; saved and shipped separately from Laya."""

    def __init__(self, dims, head_layers, in_dims, num_latents=32, depth=2):
        super().__init__()
        self.in_dims = dict(in_dims)
        self.resamplers = nn.ModuleDict(
            {m: Resampler(d, dims, num_latents, depth) for m, d in self.in_dims.items()}
        )
        self.modality_emb = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(dims)) for m in self.in_dims}
        )
        self.blocks = nn.ModuleList(GatedCrossAttention(dims) for _ in range(head_layers))

    def memory(self, inputs):
        """inputs: {modality: (feats, mask or None, present or None)} -> (tokens, key padding).

        `present` is a (B,) bool marking which rows actually have this modality, so a
        batch can mix rows with and without it; absent rows are masked out entirely.
        """
        tokens, pads = [], []
        for name, (feats, mask, present) in inputs.items():
            t = self.resamplers[name](feats, mask) + self.modality_emb[name]
            p = torch.zeros(t.shape[:2], dtype=torch.bool, device=t.device)
            if present is not None:
                p |= ~present[:, None]
            tokens.append(t)
            pads.append(p)
        return torch.cat(tokens, 1), torch.cat(pads, 1)


class DecisionModel(nn.Module):
    def __init__(self, encoder, agent_config, fusion=None):
        super().__init__()
        dims = encoder.config.hidden_size
        self.encoder = encoder
        self.head = DecisionHead(dims, agent_config.get("head_layers", 2))
        self.type_emb = nn.Embedding(3, dims)
        self.scorer = nn.Sequential(nn.LayerNorm(dims), nn.Linear(dims, dims), nn.GELU(), nn.Linear(dims, 1))
        self.act_head = nn.Sequential(
            nn.Linear(dims + 4, 256), nn.GELU(), nn.Linear(256, len(agent_config.get("act_costs", {})) + 1)
        )
        self.register_buffer("temperature", torch.ones(3))
        self.fusion = fusion

    def encode(self, input_ids, attention_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return h + self.type_emb(qtype)[:, None, :]

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, modalities=None, hidden=None):
        """modalities: optional {"image"|"audio": (feats, mask, present)}; None or {} = pure Laya.

        `hidden` lets training pass cached encoder outputs (the encoder is frozen).
        """
        h = self.encode(input_ids, attention_mask, qtype) if hidden is None else hidden
        pad = ~attention_mask.bool()
        memory = None
        if modalities and self.fusion is not None:
            memory, memory_pad = self.fusion.memory(modalities)
            # A row with no modality at all would have an all-masked cross-attention;
            # route it through a zero update instead.
            empty = memory_pad.all(dim=1)
            if empty.any():
                memory_pad = memory_pad.clone()
                memory_pad[empty, 0] = False
        for i, layer in enumerate(self.head.layers):
            h = layer(h, pad)
            if memory is not None:
                fused = self.fusion.blocks[i](h, memory, memory_pad)
                h = torch.where(empty[:, None, None], h, fused) if empty.any() else fused
        rows = torch.arange(h.shape[0], device=h.device)[:, None]
        markers = h[rows, marker_pos.clamp(min=0)]
        logits = self.scorer(markers).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask.bool(), -1e4)
        p = logits.softmax(-1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        entropy = -(p * p.clamp(min=1e-9).log()).sum(-1) / k.log()
        top = p.sort(-1).values[:, -2:]
        features = torch.stack([top[:, 1], top[:, 1] - top[:, 0], entropy, k / 255.0], -1)
        pooled = torch.cat([h[:, 0].float(), features], -1)
        action = self.act_head(pooled.to(self.act_head[0].weight.dtype)).float()
        return logits, action
