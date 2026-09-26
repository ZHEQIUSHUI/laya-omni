---
license: apache-2.0
language:
  - multilingual
  - en
  - zh
library_name: laya-omni
pipeline_tag: any-to-any
tags:
  - laya
  - decision-model
  - multimodal
  - audio
  - vision
base_model:
  - convaiinnovations/laya-multilingual
  - google/siglip2-base-patch16-256
  - Qwen/Qwen3-ASR-0.6B
---

# laya-omni (formal-v2)

[Laya](https://github.com/NandhaKishorM/laya) answers typed decisions (choose
one option, give a score, answer yes/no) with calibrated probabilities.
**laya-omni** adds optional image and audio inputs. Without an image or audio
clip, the output is bit-identical to Laya.

- Code, docs, fine-tuning tool: <https://github.com/ZHEQIUSHUI/laya-omni>
- This repository: the trained fusion (about 7M parameters, 28 MB) and the
  Qwen3-ASR-0.6B audio encoder extracted from the full ASR model (`audio_encoder/`).
- Frozen models it runs on: Laya multilingual (mmBERT-base, ~310M), SigLIP 2
  base/16-256 (~93M), Qwen3-ASR audio encoder (186M).

## Use

```bash
pip install "laya-omni @ git+https://github.com/ZHEQIUSHUI/laya-omni"
hf download zheqiushui/laya-omni --local-dir laya-omni
hf download convaiinnovations/laya-multilingual --local-dir laya-multilingual
```

```python
from laya_omni import Omni

omni = Omni.load("laya-omni", laya="laya-multilingual",
                 image_encoder="google/siglip2-base-patch16-256",
                 audio_encoder="laya-omni/audio_encoder")

omni.predict("Image.", {
    "sport": {"type": "choice", "instructions": "What game are they playing?",
              "criteria": ["baseball", "tennis", "soccer"]},
    "crowd": {"type": "noul", "instructions": "Is there an audience?"},
}, image="photo.jpg")

omni.predict("Audio clip.", {
    "intent": {"type": "choice", "instructions": "What does the caller want to do?",
               "criteria": ["check the balance", "freeze the card", "pay a bill"]},
}, audio="call.wav")
```

Several images: `image=[a, b]`. `detail=True` uses 256 image tokens instead of
64 (better for text, documents and spatial questions). One question with an
image or a 10 s clip takes 30-120 ms on an RTX 4090, encoders included.

## Results

Test questions never trained on, against Laya asked the same question without
the image or audio (it can only use the wording):

| | Laya (text only) | laya-omni |
|---|---:|---:|
| Photo questions (VQAv2 / GQA) | 0.45 / 0.55 | **0.78 / 0.78** |
| Reading text in images (TextVQA / OCR-VQA) | 0.36 / 0.44 | **0.82 / 0.89** |
| Charts and documents (ChartQA / DocVQA) | 0.36 / 0.40 | **0.82 / 0.72** |
| Science diagrams (ScienceQA / AI2D) | 0.41 / 0.25 | **0.80 / 0.62** |
| Counting (TallyQA) | 0.26 | **0.88** |
| Sound events (ESC-50 / FSD50K / VGGSound) | 0.43 / 0.45 / 0.43 | **0.96 / 0.86 / 0.84** |
| Spoken intents (SLURP / MINDS-14, 14 languages) | 0.34 / 0.41 | **0.96 / 0.95** |
| Image + audio together (OmniInstruct) | 0.46 | **0.85** |
| Held out: TQA / VQA-RAD / GTZAN | 0.26 / 0.43 / 0.41 | **0.39 / 0.52 / 0.64** |
| Zero-shot: Mini-ImageNet / MMAU / Song Describer | 0.52 / 0.29 / 0.46 | **0.79 / 0.45 / 0.68** |

Most sets were converted to Laya's multiple-choice and yes/no questions with
distractor options, so these numbers are not directly comparable to generative
VLM leaderboards. All 43 sets, calibration and ablations:
[docs/results.md](https://github.com/ZHEQIUSHUI/laya-omni/blob/main/docs/results.md)
(`eval.json` here has the raw numbers).

## Adapting to a new scenario

A few hundred labelled images or clips, one folder per answer, one command:
[docs/finetune.md](https://github.com/ZHEQIUSHUI/laya-omni/blob/main/docs/finetune.md).

## Limits

Weak on science diagrams (AI2D, TQA), medical images, spatial relations, and
questions about what happens next (game moves) rather than what is there.
Confident when wrong on unfamiliar diagram questions (TQA). Probabilities are
calibrated per modality and question type (`temperature_by_modality` in
`fusion_config.json`).

## License

Apache-2.0. Built on Laya (Convai Innovations, Apache-2.0), SigLIP 2 (Google,
Apache-2.0) and Qwen3-ASR (Qwen, Apache-2.0); `audio_encoder/` holds weights
from Qwen/Qwen3-ASR-0.6B unchanged. Trained on public datasets, each under its
own license; see
[docs/training.md](https://github.com/ZHEQIUSHUI/laya-omni/blob/main/docs/training.md).
