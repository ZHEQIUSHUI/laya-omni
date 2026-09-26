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

Test accuracy of `formal-v2` on questions it never trained on, against Laya
asked the same question without the image or audio:

| | Laya (text only) | laya-omni |
|---|---:|---:|
| Photo questions (VQAv2 / GQA) | 0.45 / 0.55 | **0.78 / 0.78** |
| Reading text in images (TextVQA / OCR-VQA) | 0.36 / 0.44 | **0.82 / 0.89** |
| Charts and documents (ChartQA / DocVQA) | 0.36 / 0.40 | **0.82 / 0.72** |
| Science diagrams (ScienceQA / AI2D) | 0.41 / 0.25 | **0.80 / 0.62** |
| Counting (TallyQA) | 0.26 | **0.88** |
| Sound events (ESC-50 / VGGSound) | 0.43 / 0.43 | **0.96 / 0.84** |
| Spoken intents (SLURP 101 intents / MINDS-14, 14 languages) | 0.34 / 0.41 | **0.96 / 0.95** |
| Audio captions (AudioCaps) | 0.46 | **0.91** |
| Image + audio together (OmniInstruct) | 0.46 | **0.85** |
| Zero-shot: Mini-ImageNet / MMAU / Song Describer | 0.52 / 0.29 / 0.46 | **0.79 / 0.45 / 0.68** |

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

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Built on
[Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations),
[SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-256) and
[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), and trained on public
datasets listed in [docs/training.md](docs/training.md), each under its own
license; datasets are not redistributed here.
