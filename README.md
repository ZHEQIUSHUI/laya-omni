# laya-omni

**Laya typed decisions with optional image and audio inputs.**

[Laya](https://github.com/NandhaKishorM/laya) answers constrained questions
(`choice`, `score`, `noul`) about a text state in one forward pass: calibrated
probabilities, no generated tokens. laya-omni lets the state also include
images, audio clips, or both, and keeps everything else about Laya: the same
questions, the same output, and, with no image or audio, the same numbers bit
for bit.

[中文说明](README.zh-CN.md) · [Architecture](docs/architecture.md) · [Results](docs/results.md) · [Fine-tuning](docs/finetune.md) · [Reproducing](docs/training.md)

## What it can do

Standard benchmarks, compared with small VLMs on the same questions, the same
RTX 4090, one question at a time from the image or audio file to the answer.
Each cell is accuracy / median latency:

| Benchmark (chance) | laya-omni | Qwen3.5-0.8B | Qwen3.5-2B | Valen (Qwen3.5-2B) |
|---|---:|---:|---:|---:|
| MMBench-EN dev (0.25) | 0.63 / **21 ms** | 0.75 / 42 ms | **0.84** / 46 ms | 0.73 / 70 ms |
| MMBench-CN dev (0.25) | 0.58 / **21 ms** | 0.75 / 42 ms | **0.82** / 45 ms | 0.73 / 70 ms |
| MMStar (0.25) | 0.36 / **21 ms** | 0.46 / 42 ms | **0.53** / 46 ms | 0.44 / 70 ms |
| MME, yes / no (0.5) | 0.66 / **23 ms** | 0.76 / 48 ms | **0.82** / 62 ms | 0.61 / 105 ms |
| POPE, object hallucination (0.5) | 0.83 / **22 ms** | 0.88 / 44 ms | **0.90** / 51 ms | pending |
| SEED-Bench, image (0.25) | 0.54 / **26 ms** | **0.72** / 61 ms | pending | pending |
| Valen-Eval-General-5k (Valen's own set) | 0.61 / **26 ms** | 0.70 / 50 ms | **0.74** / 61 ms | **0.74** / 207 ms* |
| OmniBench, image + audio (0.25) | 0.40 | pending (Qwen2.5-Omni-3B) | – | – |
| MMAU test-mini, audio (~0.25) | 0.45 | pending (Qwen2.5-Omni-3B) | – | – |

laya-omni answers about twice as fast as the smallest Qwen3.5 and adds audio,
but it is 5-18 points less accurate on these benchmarks; closing that gap
(distillation from a larger VLM) is the current work. Notes:

- Qwen3.5 answers by the logit of the option letter (or Yes / No) after one
  forward pass, with no decoding, which is the fastest way to run it.
- Valen is [Valen-Preview-0923](https://huggingface.co/Valen-Team/Valen-Preview-0923),
  its only public checkpoint (tuned further on Sokoban), run in its pinned
  environment; our harness reproduces its own evaluation (0.738 vs 0.737).
  *Measured before its fast linear-attention kernels were installed.
  About 12% of Valen's set also appears in laya-omni's training data.
- Latencies are medians from runs that shared the machine with other jobs; a
  clean timing pass is pending.
- Reproduce with `scripts/convert_bench.py` and `scripts/bench.py`; details in
  [docs/results.md](docs/results.md#standard-benchmarks).

Full tables, calibration, few-shot adaptation and what did not work:
[docs/results.md](docs/results.md).

## Size and speed

| | Parameters |
|---|---:|
| Laya multilingual (frozen) | ~310M |
| SigLIP 2 base/16 image encoder (frozen) | ~93M |
| Qwen3-ASR-0.6B audio encoder (frozen) | 186M |
| laya-omni fusion (trained) | ~7M |

One question with an image or a 10 s clip takes 30-120 ms on an RTX 4090,
encoders included. Images use 64 tokens by default; `detail=True` uses 256,
which helps with text-heavy images.

## Use

```python
from laya_omni import Omni

omni = Omni.load(
    "runs/formal-v2", laya="models/laya-multilingual",
    image_encoder="models/siglip2-base-patch16-256",
    audio_encoder="models/qwen3-asr-0.6b-audio-encoder",
)

omni.predict("Image.", {
    "sport": {"type": "choice", "instructions": "What game are they playing?",
              "criteria": ["baseball", "tennis", "soccer"]},
    "crowd": {"type": "noul", "instructions": "Is there an audience?"},
}, image="photo.jpg")

omni.predict("Audio clip.", {
    "intent": {"type": "choice", "instructions": "What does the caller want to do?",
               "criteria": ["check the balance", "freeze the card", "pay a bill"]},
}, audio="call.wav")

omni.predict(state, questions)   # no image, no audio: exactly Laya
```

Several images: `image=[a, b]`. The output follows Laya's schema (`choice`,
`probabilities`, `score`, `noul`, `confidence`, `action`).

## Web demo

```bash
pip install -e '.[web]'
laya-omni serve --fusion runs/formal-v2 --laya models/laya-multilingual \
    --image-encoder models/siglip2-base-patch16-256 \
    --audio-encoder models/qwen3-asr-0.6b-audio-encoder --examples data --port 8030
```

Upload images or audio (or record from the microphone), write questions, and
compare the answer with and without the image or audio.

## Adapt to your scenario

Put examples in one folder per answer and run one command; it trains from the
general fusion with early stopping and calibration, in minutes for a few
hundred examples:

```bash
python scripts/finetune.py --folders my_task/ --question "Is the part defective?" \
    --base runs/formal-v2 --laya models/laya-multilingual \
    --image-encoder models/siglip2-base-patch16-256 --out runs/my-task
```

Tutorial and results: [docs/finetune.md](docs/finetune.md).

## How it works

Frozen encoders turn each image into 64 (or 256) feature frames and each
second of audio into about 13; a small MLP maps every frame to one token in
Laya's input, right after `[CLS]`. Laya's frozen encoder and decision head read
them with the question and options, helped by LoRA deltas that apply only to
rows carrying an image or audio clip. Details and the reasons for each choice:
[docs/architecture.md](docs/architecture.md).

## Status

Early research release. Weights: [zheqiushui/laya-omni](https://huggingface.co/zheqiushui/laya-omni)
(fusion + audio encoder; `hf download zheqiushui/laya-omni --local-dir laya-omni`). Known weak spots:
science diagrams (AI2D, TQA), spatial relations (VSR), and tasks that need
reasoning about what happens next (game moves) rather than what is there.

## License and credits

The code is Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The
published weights were trained on public datasets that are not redistributed
here, some of which are for non-commercial research only (e.g. ScienceQA,
ESC-50, Hateful Memes), so use the weights for research and non-commercial
purposes; for commercial use, retrain with the recipes on data whose licenses
allow it. Dataset list and licenses: [docs/training.md](docs/training.md).

Built on [Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations),
[SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-256) and
[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), all Apache-2.0.
