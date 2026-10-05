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

# laya-omni (formal-v3)

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

Several images: `image=[a, b]`. Images use 256 tokens by default (`detail=False`:
64). At load time the LoRA deltas are folded into a copy of Laya's encoder for
image / audio requests (same outputs, about a quarter faster; `merge=False` to
skip). One question with an image takes 14-20 ms on an RTX 4090, encoders included.

The previous release is kept under the tag `formal-v2`
(`hf download zheqiushui/laya-omni --revision formal-v2`).

## Results

Standard benchmarks, same questions, one RTX 4090, one question at a time from
the image or audio file to the answer (accuracy / median latency):

| Benchmark | laya-omni | Qwen3.5-0.8B | Qwen3.5-2B | Valen-Preview-0923 | Qwen2.5-Omni-3B |
|---|---:|---:|---:|---:|---:|
| MMBench-EN dev | 0.66 / **14 ms** | 0.75 / 42 ms | **0.84** / 46 ms | 0.73 / 70 ms | – |
| MMBench-CN dev | 0.60 / **14 ms** | 0.75 / 42 ms | **0.82** / 45 ms | 0.73 / 70 ms | – |
| MMStar | 0.37 / **14 ms** | 0.46 / 42 ms | **0.53** / 46 ms | 0.44 / 70 ms | – |
| MME (yes / no) | 0.68 / **17 ms** | 0.76 / 48 ms | **0.82** / 62 ms | 0.61 / 105 ms | – |
| POPE | 0.84 / **16 ms** | 0.88 / 44 ms | **0.90** / 51 ms | 0.82 / 88 ms | – |
| SEED-Bench (image) | 0.57 / **20 ms** | 0.72 / 61 ms | **0.77** / 82 ms | 0.66 / 222 ms | – |
| OmniBench (image + audio) | 0.38 / **50 ms** | – | – | – | **0.42** / 321 ms |
| MMAU test-mini (audio) | 0.45 / **20 ms** | – | – | – | **0.67** / 91 ms |

About three times as fast as the smallest Qwen3.5 and 4-15 points less accurate
on the image benchmarks; Qwen2.5-Omni-3B is well ahead on audio. On held-out COCO
images it tells where an object is (0.85, 4 options), left / right / above
(0.85), which is bigger (0.94) and which objects are present (0.96). Protocol,
CircularEval, overlap checks and ablations:
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
  plus questions generated from COCO 2017 boxes (CC BY 4.0) and option
  probabilities from Qwen3.8-27B (Apache-2.0); some datasets are for
  non-commercial research only (e.g. ScienceQA
  CC BY-NC-SA 4.0, ESC-50 CC BY-NC 3.0, Hateful Memes research license). Use it
  for research and non-commercial purposes; for commercial use, retrain with the
  repository's recipes on data whose licenses allow it.
- `audio_encoder/`: the audio encoder of
  [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), unchanged,
  Apache-2.0 (license text and notice in that folder).
- Runs on [Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations) and
  [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-256) (Google),
  both Apache-2.0, downloaded separately.
