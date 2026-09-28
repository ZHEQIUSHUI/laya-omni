---
license: other
license_name: research-non-commercial
license_link: LICENSE.md
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

Standard benchmarks, same questions, one RTX 4090, one question at a time from
the image or audio file to the answer (accuracy / median latency):

| Benchmark | laya-omni | Qwen3.5-0.8B | Qwen3.5-2B | Valen-Preview-0923 |
|---|---:|---:|---:|---:|
| MMBench-EN dev | 0.63 / **21 ms** | 0.75 / 42 ms | **0.84** / 46 ms | 0.73 / 70 ms |
| MMBench-CN dev | 0.58 / **21 ms** | 0.75 / 42 ms | **0.82** / 45 ms | 0.73 / 70 ms |
| MMStar | 0.36 / **21 ms** | 0.46 / 42 ms | **0.53** / 46 ms | 0.44 / 70 ms |
| MME (yes / no) | 0.66 / **23 ms** | 0.76 / 48 ms | **0.82** / 62 ms | 0.61 / 105 ms |
| POPE | 0.83 / **22 ms** | 0.88 / 44 ms | **0.90** / 51 ms | – |
| SEED-Bench (image) | 0.54 / **26 ms** | **0.72** / 61 ms | – | – |
| OmniBench (image + audio) | 0.40 | – | – | – |
| MMAU test-mini (audio) | 0.45 | – | – | – |

About twice as fast as the smallest Qwen3.5, with audio input, and 5-18 points
less accurate; distillation from a larger VLM is in progress. Protocol,
CircularEval, overlap checks and the in-distribution test splits:
[docs/results.md](https://github.com/ZHEQIUSHUI/laya-omni/blob/main/docs/results.md).

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

- Code: Apache-2.0 ([repository](https://github.com/ZHEQIUSHUI/laya-omni)).
- `fusion.safetensors`: trained on public datasets that are not redistributed,
  some of which are for non-commercial research only (e.g. ScienceQA
  CC BY-NC-SA 4.0, ESC-50 CC BY-NC 3.0, Hateful Memes research license). Use it
  for research and non-commercial purposes; for commercial use, retrain with the
  repository's recipes on data whose licenses allow it.
- `audio_encoder/`: the audio encoder of
  [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), unchanged,
  Apache-2.0 (license text and notice in that folder).
- Runs on [Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations) and
  [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-256) (Google),
  both Apache-2.0, downloaded separately.
