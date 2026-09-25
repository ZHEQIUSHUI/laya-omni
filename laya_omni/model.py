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

Optionally (lora_rank > 0) the frozen encoder and decision head get LoRA deltas
that act only on rows carrying an image or audio: a frozen projector alone
learns simple visual mappings (game frames) but underfits general visual
questions, and gating the deltas by row keeps text-only rows exactly Laya's.

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
# Linear layers that get a LoRA delta: ModernBERT's attention and MLP projections,
# and the decision head's feed-forward layers.
LORA_TARGETS = ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo", "linear1", "linear2")


class LoRA(nn.Module):
    """Low-rank delta B @ A for one frozen linear layer; B starts at zero."""

    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_features) / in_features**0.5)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return (x.to(self.A.dtype) @ self.A.T @ self.B.T) * self.scale


class OmniFusion(nn.Module):
    """All trainable multimodal parameters; saved and shipped separately from Laya.

    With `lora_rank`, the frozen encoder and decision head also get LoRA deltas,
    applied only to rows that carry an image or audio: Laya itself stays frozen,
    and text-only rows compute exactly what Laya computes.
    """

    def __init__(self, dims, in_dims, max_items=8, max_frames=1024, lora_rank=0, lora_alpha=None, lora_layers=None,
                 image_grid=0):
        super().__init__()
        self.in_dims = dict(in_dims)
        self.max_items = max_items
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha or 2 * max(lora_rank, 1)
        # Filled by attach(): module path -> LoRA ("." is not allowed in ModuleDict keys).
        self.lora = nn.ModuleDict()
        self.lora_layers = list(lora_layers or [])
        self.row_mask = None  # set per forward: which rows carry a modality
        self._hooks = []
        self.projectors = nn.ModuleDict({m: Projector(d, dims) for m, d in self.in_dims.items()})
        self.modality_emb = nn.ParameterDict({m: nn.Parameter(torch.zeros(dims)) for m in self.in_dims})
        # Which image (or clip) a token came from, so a question can say "the second image",
        # and where in it: the encoder's rotary positions only see the flat sequence.
        self.item_emb = nn.Parameter(torch.zeros(max_items, dims))
        self.frame_emb = nn.Parameter(torch.zeros(max_frames, dims))
        nn.init.normal_(self.item_emb, std=0.02)
        nn.init.normal_(self.frame_emb, std=0.02)
        # Image tokens get a 2-D position on a side x side grid instead of a frame index,
        # so 256 patch tokens and their 2x2-pooled 64 share positions: the grid is
        # pooled the same way the features are.
        self.image_grid = image_grid
        if image_grid:
            self.image_pos = nn.Parameter(torch.randn(image_grid, image_grid, dims) * 0.02)

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
            tok = tok + self.item_emb[:n, None, :] + self.positions(name, t).to(tok.dtype)
            ok = torch.ones(b, n, t, dtype=torch.bool, device=tok.device)
            if mask is not None:
                ok &= mask.bool()
            if present is not None:
                ok &= present.bool()[:, :, None]
            tokens.append(tok.reshape(b, n * t, -1))
            valid.append(ok.reshape(b, n * t))
        return torch.cat(tokens, 1), torch.cat(valid, 1)

    def positions(self, name, t):
        """(t, dims) position embeddings for t tokens of one item."""
        side = int(round(t**0.5))
        if name == "image" and self.image_grid and side * side == t and self.image_grid % side == 0:
            grid = self.image_pos.permute(2, 0, 1)[None]  # (1, dims, g, g)
            k = self.image_grid // side
            if k > 1:
                grid = F.avg_pool2d(grid, k)
            return grid[0].flatten(1).T
        return self.frame_emb[:t]

    def attach(self, model):
        """Hook LoRA deltas onto the model's target linear layers (creating them the
        first time). Hooks leave parameter names alone, so Laya's checkpoint still
        loads strictly and the deltas live, and are saved, with the fusion."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        if not self.lora_rank:
            return
        names = self.lora_layers or [
            n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.startswith(("encoder.", "head.")) and n.endswith(LORA_TARGETS)
        ]
        self.lora_layers = names
        modules = dict(model.named_modules())
        for name in names:
            key = name.replace(".", "__")
            linear = modules[name]
            if key not in self.lora:
                self.lora[key] = LoRA(linear.in_features, linear.out_features, self.lora_rank, self.lora_alpha).to(
                    linear.weight.device
                )
            self._hooks.append(linear.register_forward_hook(self._hook(self.lora[key])))

    def _hook(self, lora):
        def hook(module, inputs, output):
            if self.row_mask is None:
                return None
            if output.dim() != 3 or output.shape[0] != self.row_mask.shape[0]:
                raise RuntimeError("LoRA rows need padded (batch, length, dims) activations")
            delta = lora(inputs[0]).to(output.dtype)
            return output + delta * self.row_mask[:, None, None].to(output.dtype)

        return hook


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

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "fusion" and value is not None:
            value.attach(self)  # hook the fusion's LoRA deltas onto this model's layers

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
        fused = bool(modalities) and self.fusion is not None
        try:
            if fused:
                n_before = attention_mask.sum(1)
                embeds, attention_mask, marker_pos = self.splice(input_ids, attention_mask, marker_pos, modalities)
                # LoRA deltas apply only to rows that actually got modality tokens.
                self.fusion.row_mask = (attention_mask.sum(1) > n_before).float()
                h = self.encoder(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
            else:
                h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            h = h + self.type_emb(qtype)[:, None, :]
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, pad)
        finally:
            if fused:
                self.fusion.row_mask = None
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
