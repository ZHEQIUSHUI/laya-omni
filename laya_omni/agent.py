"""Inference runtime: upstream Laya prompts and outputs, with optional image / audio."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .common import QTYPES, build_sequence, confidence_from_probs, render_options, temp_bucket
from .model import DecisionModel, OmniFusion

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
# Upstream refuses calibration temperatures outside this range (some shipped buckets
# sharpen a coin flip into a certainty), so the same clamp applies here.
TEMP_MIN, TEMP_MAX = 0.5, 5.0


def clamp_temperature(t):
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 if not math.isfinite(t) else min(TEMP_MAX, max(TEMP_MIN, t))


def load_tokenizer(path):
    """AutoTokenizer, or, for configs newer transformers rejects, the same fix upstream
    applies (list-valued extra_special_tokens, TokenizersBackend class) done in memory
    instead of rewriting the checkpoint's files."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(path)
    except Exception:
        cfg = json.loads((Path(path) / "tokenizer_config.json").read_text())
        special = {k: cfg[k] for k in ("cls_token", "sep_token", "mask_token", "pad_token", "unk_token", "bos_token", "eos_token") if isinstance(cfg.get(k), str)}
        return PreTrainedTokenizerFast(tokenizer_file=str(Path(path) / "tokenizer.json"), **special)


def build_encoder(encoder_dir):
    from transformers import ModernBertConfig, ModernBertModel

    cfg = ModernBertConfig.from_pretrained(encoder_dir)
    cfg._attn_implementation = "sdpa"
    return ModernBertModel(cfg)


def collate_items(items, pad_id):
    n, length = len(items), max(len(item["ids"]) for item in items)
    count = max(2, max(len(item["markers"]) for item in items))
    batch = {
        "input_ids": torch.full((n, length), pad_id, dtype=torch.long),
        "attention_mask": torch.zeros((n, length), dtype=torch.long),
        "marker_pos": torch.zeros((n, count), dtype=torch.long),
        "marker_mask": torch.zeros((n, count), dtype=torch.bool),
        "qtype": torch.tensor([item["qtype"] for item in items], dtype=torch.long),
    }
    for i, item in enumerate(items):
        length, count = len(item["ids"]), len(item["markers"])
        batch["input_ids"][i, :length] = torch.tensor(item["ids"])
        batch["attention_mask"][i, :length] = 1
        batch["marker_pos"][i, :count] = torch.tensor(item["markers"])
        batch["marker_mask"][i, :count] = True
    return batch


def to_internal(qdef):
    if not isinstance(qdef, dict) or qdef.get("type") not in QTYPES or "instructions" not in qdef:
        raise ValueError("Each question needs type (choice/score/noul) and instructions")
    kind, criteria = qdef["type"], qdef.get("criteria")
    if kind == "choice" and isinstance(criteria, list):
        criteria = dict.fromkeys(criteria)
    if kind == "choice" and not (isinstance(criteria, dict) and criteria):
        raise ValueError("Choice criteria must be a nonempty dictionary or list")
    if kind == "score" and not (isinstance(criteria, list) and criteria):
        raise ValueError("Score criteria must be a nonempty list")
    ins = qdef["instructions"]
    return {"t": kind, "ins": ins if isinstance(ins, str) else json.dumps(ins), "crit": criteria}


class Agent:
    """Load a Laya checkpoint directory and, optionally, a laya-omni fusion checkpoint."""

    def __init__(self, laya_dir, fusion_path=None, *, device=None, dtype="float32", batch_size=16):
        self.laya_dir = Path(laya_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = DTYPES[dtype]
        self.batch_size = batch_size
        self.cfg = json.loads((self.laya_dir / "rl_agent_config.json").read_text())
        self.max_len = self.cfg.get("max_len", 512)
        self.head_max_len = self.cfg.get("head_max_len", 192)
        self.temperature = [clamp_temperature(t) for t in self.cfg.get("temperature", [1.0, 1.0, 1.0])]
        self.temperature_by_options = {k: clamp_temperature(v) for k, v in self.cfg.get("temperature_by_options", {}).items()}
        self.tok = load_tokenizer(self.laya_dir / "tokenizer")
        self.model = DecisionModel(build_encoder(self.laya_dir / "encoder"), self.cfg)
        missing, unexpected = self.model.load_state_dict(
            load_file(self.laya_dir / "model.safetensors"), strict=False
        )
        if unexpected or [k for k in missing if not k.startswith("fusion.")]:
            raise ValueError(f"Checkpoint mismatch: missing={missing} unexpected={unexpected}")
        self.fusion_cfg = None
        if fusion_path:
            self.load_fusion(fusion_path)
        self.model.to(self.device, self.dtype).eval()

    def load_fusion(self, path):
        path = Path(path)
        self.fusion_cfg = json.loads((path / "fusion_config.json").read_text())
        fusion = OmniFusion(
            self.model.encoder.config.hidden_size,
            self.fusion_cfg["in_dims"],
            self.fusion_cfg.get("max_items", 8),
            self.fusion_cfg.get("max_frames", 1024),
            lora_rank=self.fusion_cfg.get("lora_rank", 0),
            lora_alpha=self.fusion_cfg.get("lora_alpha"),
            lora_layers=self.fusion_cfg.get("lora_layers"),
        )
        self.model.fusion = fusion  # attaching creates the LoRA modules the weights fill
        fusion.load_state_dict(load_file(path / "fusion.safetensors"), strict=True)
        fusion.to(self.device, self.dtype).eval()

    def prepare(self, state, questions):
        items, internal = [], []
        for qid, definition in questions.items():
            q = to_internal(definition)
            ids, markers = build_sequence(self.tok, state, q, self.max_len, self.head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError(f"Question {qid!r} has too many options for the token budget")
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            internal.append(q)
        return items, internal

    def _features(self, feats):
        if isinstance(feats, torch.Tensor):
            return feats.detach().to(self.device, self.dtype)
        return torch.as_tensor(np.asarray(feats), dtype=self.dtype, device=self.device)

    def _modalities(self, n, image, audio):
        """Each argument is encoder features, or None: one (T, D) array or tensor, or a
        list of them for several images / clips (lengths may differ)."""
        out = {}
        for name, feats in (("image", image), ("audio", audio)):
            if feats is None:
                continue
            if self.model.fusion is None or name not in self.model.fusion.in_dims:
                raise ValueError(f"No fusion weights loaded for {name!r}")
            if isinstance(feats, (list, tuple)):
                items = [self._features(f) for f in feats]
                t = max(f.shape[0] for f in items)
                stacked = torch.zeros(len(items), t, items[0].shape[-1], dtype=self.dtype, device=self.device)
                mask = torch.zeros(len(items), t, dtype=torch.bool, device=self.device)
                for i, f in enumerate(items):
                    stacked[i, : f.shape[0]] = f
                    mask[i, : f.shape[0]] = True
                out[name] = (stacked[None].expand(n, -1, -1, -1), mask[None].expand(n, -1, -1), None)
            else:
                f = self._features(feats)
                out[name] = (f[None].expand(n, -1, -1), None, None)
        return out

    @torch.inference_mode()
    def predict(self, state, questions, *, image=None, audio=None):
        """Upstream Laya output schema. image/audio: optional frozen-encoder features."""
        items, internal = self.prepare(state, questions)
        combo = "+".join(m for m, v in (("image", image), ("audio", audio)) if v is not None) or "text"
        answers, qids = {}, list(questions)
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            batch = {k: v.to(self.device) for k, v in collate_items(chunk, self.tok.pad_token_id).items()}
            mods = self._modalities(len(chunk), image, audio)
            logits, act = self.model(**batch, modalities=mods or None)
            logits, act = logits.cpu().numpy(), act.softmax(-1).cpu().numpy()
            if not np.isfinite(logits).all():
                raise FloatingPointError("Non-finite model outputs; retry with dtype='float32'")
            for row, item in enumerate(chunk):
                q, k, qt = internal[start + row], len(item["markers"]), item["qtype"]
                scale = self._temperature(qt, k, combo)
                z = logits[row, :k] / max(1e-3, float(scale))
                p = np.exp(z - z.max())
                p /= p.sum()
                answer = {
                    "type": q["t"],
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "action": {"act_probability": round(float(act[row, 0]), 4)},
                }
                if q["t"] == "choice":
                    labels = list(q["crit"])
                    answer.update(
                        choice=labels[int(p.argmax())],
                        probabilities={label: round(float(v), 4) for label, v in zip(labels, p)},
                    )
                elif q["t"] == "score":
                    answer.update(
                        score=round(float((np.arange(k) * p).sum()), 4),
                        legend={str(i): v for i, v in enumerate(q["crit"])},
                        probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
                    )
                else:
                    answer.update(noul=round(float(p[1]), 4), confidence=round(max(float(p[1]), 1 - float(p[1])), 4))
                answers[qids[start + row]] = answer
        return {
            "model": "laya-omni",
            "modalities": combo,
            "answers": answers,
            "usage": {"input_tokens": sum(len(i["ids"]) for i in items), "output_tokens": 0},
        }

    def _temperature(self, qtype, k, combo):
        if combo != "text" and self.fusion_cfg:
            t = self.fusion_cfg.get("temperature_by_modality", {}).get(f"{combo}:{temp_bucket(qtype, k)}")
            if t is not None:
                return clamp_temperature(t)
        return self.temperature_by_options.get(temp_bucket(qtype, k), self.temperature[qtype])


def load(laya_dir, fusion_path=None, **kwargs):
    return Agent(laya_dir, fusion_path, **kwargs)
