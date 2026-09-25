# Architecture

laya-omni adds optional image and audio inputs to [Laya](https://github.com/NandhaKishorM/laya),
a bidirectional typed-decision model: it answers `choice`, `score` and `noul`
questions about a state in one forward pass, with probabilities and no
generated tokens.

```
image ──► SigLIP 2 (frozen) ──► 256 patch features ──► pool to 64 (default) or keep 256
audio ──► Qwen3-ASR audio encoder (frozen) ──► ~13 frames per second
                         │
                         ▼
             projector (MLP, trained) + modality / item / 2-D position embeddings
                         │   one token per feature frame
                         ▼
[CLS] <image and audio tokens> <question> [SEP] [MASK] option … [SEP] <state> [SEP]
                         │
       mmBERT encoder (frozen) + LoRA on rows with image/audio (trained)
                         │
       Laya decision head (frozen) + LoRA on its feed-forward layers (trained)
                         │
       Laya's scorer on each option marker (frozen) ──► probabilities
```

## Parts

| Part | Size | Trained |
|---|---:|---|
| Laya multilingual: mmBERT-base encoder, decision head, scorer, act head | ~310M | no |
| SigLIP 2 base/16, vision tower | ~93M | no |
| Qwen3-ASR-0.6B audio encoder (AuT) | 186M | no |
| Image and audio projectors (2-layer MLP each) | ~2.5M | yes |
| Modality, item and position embeddings | ~1M | yes |
| LoRA rank 16 on mmBERT's attention and MLP projections and the head's feed-forward | 3.6M | yes |

Everything trained lives in one `fusion.safetensors`, shipped next to an
unchanged Laya checkpoint.

## Design choices

**Tokens in the encoder input, not cross-attention.** Each feature frame
becomes one token in mmBERT's word-embedding space, placed right after `[CLS]`,
the way LLaVA feeds a frozen language model. Laya already relates a state to
the question and options; image and audio tokens are read the same way. A
Flamingo-style gated cross-attention into the decision head was tried first
and could not learn (see [results](results.md#what-did-not-work-ablations)).

**Row-gated LoRA.** A projector alone learns simple visual mappings but
underfits general questions, so the frozen encoder and head get LoRA deltas.
They are forward hooks that act only on rows that carry an image or audio clip:
Laya's own parameter names are untouched (its checkpoint loads strictly), and
a text-only question computes exactly what Laya computes.

**Text-only output is Laya's, bit for bit.** With no image and no audio
nothing is inserted and no hook fires. `tests/test_identity.py` checks this
with and without LoRA, and that rows without media in a mixed batch match
Laya. `scripts/parity.py` checks the PyTorch port against the official `laya`
package (identical fp32 logits).

**Scalable image tokens.** SigLIP's 16x16 patch grid is either average-pooled
to 8x8 (64 tokens, the default) or kept (256 tokens, `detail=True`). Training
draws one of the two per batch, and image tokens get a learned 2-D position
grid pooled the same way as the features, so one model serves both.

**Several images or clips.** An item embedding marks which image a token came
from, so questions can refer to "the left image".

**Frozen, swappable encoders.** The audio encoder is Qwen3-ASR's audio tower on
transformers' own `Qwen3ASREncoder` (`scripts/extract_audio_encoder.py`; checked
against qwen-asr by `scripts/check_audio_encoder.py`). A Qwen3.5 vision tower
is available as an image encoder (`scripts/extract_vision_encoder.py`,
`QwenImageEncoder`), with native resolution; it was not better on average and
costs more tokens, so SigLIP 2 is the default.

**Calibration.** Training uses cross-entropy over the option markers (a
proper scoring rule). A temperature per (modality, question type, option
count), fitted by `scripts/evaluate.py --calibrate`, is stored in
`fusion_config.json` and applied by `Agent.predict`.

## Code map

| File | What it holds |
|---|---|
| `laya_omni/model.py` | DecisionModel (Laya in PyTorch), OmniFusion (projectors, embeddings, LoRA hooks), token splicing |
| `laya_omni/agent.py` | Loading Laya and a fusion; Laya's prompt format, calibration and output schema |
| `laya_omni/encoders.py` | SigLIP 2, Qwen3-ASR and Qwen3.5 encoders |
| `laya_omni/pipeline.py` | `Omni`: files in, decisions out |
| `laya_omni/data.py` | Training data: sources, batching in DataLoader workers, on-GPU image encoding |
| `laya_omni/server.py`, `web/index.html` | The web demo |
| `scripts/convert_*.py` | Public datasets to typed questions |
| `scripts/train.py`, `scripts/evaluate.py` | Training (one or several GPUs) and evaluation / calibration |
