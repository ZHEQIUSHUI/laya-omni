"""Laya's DecisionModel in PyTorch, plus optional image / audio inputs.

The text path reproduces upstream Laya (see NOTICE): a ModernBERT encoder, a
question-type embedding, a small pre-norm transformer decision head, a marker
scorer and an act head. Parameter names match the upstream checkpoint, so its
model.safetensors loads strictly without conversion.

Images and audio enter the way LLaVA feeds a frozen language model: a small MLP
maps each frozen feature frame to one token in the encoder's word-embedding
space, and those tokens are placed right after [CLS], ahead of the question.
The frozen encoder then reads them together with the question, options and
state, and the decision head scores the options as usual. (Gated cross-attention
into the decision head, Flamingo-style, was tried first and could not learn:
the frozen head's residual stream is two orders of magnitude larger than the
updates, and early on every option marker reads the same thing.)

With no image and no audio nothing is inserted and the output is the original
Laya output bit for bit.
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


class Projector(nn.Sequential):
    """One feature frame -> one token in the encoder's word-embedding space."""

    def __init__(self, in_dims, dims):
        super().__init__(nn.LayerNorm(in_dims), nn.Linear(in_dims, dims), nn.GELU(), nn.Linear(dims, dims))


MODALITIES = ("image", "audio")


class OmniFusion(nn.Module):
    """All trainable multimodal parameters; saved and shipped separately from Laya."""

    def __init__(self, dims, in_dims, max_items=8, max_frames=1024):
        super().__init__()
        self.in_dims = dict(in_dims)
        self.max_items = max_items
        self.projectors = nn.ModuleDict({m: Projector(d, dims) for m, d in self.in_dims.items()})
        self.modality_emb = nn.ParameterDict({m: nn.Parameter(torch.zeros(dims)) for m in self.in_dims})
        # Which image (or clip) a token came from, so a question can say "the second image",
        # and where in it: the encoder's rotary positions only see the flat sequence.
        self.item_emb = nn.Parameter(torch.zeros(max_items, dims))
        self.frame_emb = nn.Parameter(torch.zeros(max_frames, dims))
        nn.init.normal_(self.item_emb, std=0.02)
        nn.init.normal_(self.frame_emb, std=0.02)

    def tokens(self, inputs):
        """inputs: {modality: (feats, mask or None, present or None)} -> (tokens, valid).

        One item per row: feats (B, T, D), mask (B, T), present (B,).
        Several items per row: feats (B, N, T, D), mask (B, N, T), present (B, N).
        `present` marks which rows (items) really exist, so a batch can mix rows with
        different numbers of images, or none. Returns (B, M, dims) tokens and a
        (B, M) bool marking the real ones.
        """
        tokens, valid = [], []
        for name, (feats, mask, present) in inputs.items():
            if feats.dim() == 3:
                feats = feats[:, None]
                mask = None if mask is None else mask[:, None]
                present = None if present is None else present[:, None]
            b, n, t, _ = feats.shape
            if n > self.max_items or t > self.frame_emb.shape[0]:
                raise ValueError(f"{name}: at most {self.max_items} items of {self.frame_emb.shape[0]} frames per row")
            tok = self.projectors[name](feats) + self.modality_emb[name]
            tok = tok + self.item_emb[:n, None, :] + self.frame_emb[:t]
            ok = torch.ones(b, n, t, dtype=torch.bool, device=tok.device)
            if mask is not None:
                ok &= mask.bool()
            if present is not None:
                ok &= present.bool()[:, :, None]
            tokens.append(tok.reshape(b, n * t, -1))
            valid.append(ok.reshape(b, n * t))
        return torch.cat(tokens, 1), torch.cat(valid, 1)


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

    def splice(self, input_ids, attention_mask, marker_pos, modalities):
        """Embed text and place each row's real modality tokens right after [CLS].

        Masked-out modality slots (missing images, padded frames) move to the end of
        the row as padding, so a row without any modality keeps every text token at
        its original position. Returns (embeds, attention mask, marker positions).
        """
        text = self.encoder.embeddings.tok_embeddings(input_ids)
        extra, valid = self.fusion.tokens(modalities)
        extra = extra.to(text.dtype)
        b, m = valid.shape
        seq = torch.cat([text[:, :1], extra, text[:, 1:]], 1)
        att = torch.cat([attention_mask[:, :1].bool(), valid, attention_mask[:, 1:].bool()], 1)
        # Stable order: [CLS], real modality tokens, text; then everything masked.
        pos = torch.arange(seq.shape[1], device=seq.device).expand(b, -1)
        is_extra = (pos >= 1) & (pos < 1 + m)
        key = torch.where(is_extra & ~att, pos + 2 * seq.shape[1], pos)
        order = key.argsort(dim=1, stable=True)
        seq = seq.gather(1, order[:, :, None].expand(-1, -1, seq.shape[-1]))
        att = att.gather(1, order)
        shift = valid.sum(1, keepdim=True)
        return seq, att.long(), marker_pos + shift

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, modalities=None):
        """modalities: optional {"image"|"audio": (feats, mask, present)}; None or {} = pure Laya."""
        if modalities and self.fusion is not None:
            embeds, attention_mask, marker_pos = self.splice(input_ids, attention_mask, marker_pos, modalities)
            h = self.encoder(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
        else:
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        pad = ~attention_mask.bool()
        for layer in self.head.layers:
            h = layer(h, pad)
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
