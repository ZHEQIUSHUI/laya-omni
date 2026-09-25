# Adapting to your scenario

[中文](finetune.zh-CN.md)

The general fusion answers what is in a picture or a clip. Real scenarios have
their own criteria: is this part defective, is this camera frame unusual, what
kind of ticket is this call. A few hundred to a few thousand labelled examples
are usually enough to adapt to one.

Fine-tuning trains only the ~7M-parameter fusion; Laya, SigLIP 2 and Qwen3-ASR
stay frozen. Each scenario is a 28 MB directory, and all scenarios share the
same base models.

## 1. Data

The simplest layout is **one folder per answer**:

```
my_task/
├── normal/        <- the folder names are the options
│   ├── 0001.jpg
│   └── ...
└── defect/
    ├── 0101.jpg
    └── ...
```

- Images (jpg, png, webp, bmp) or audio (wav, flac, mp3, ogg, m4a); use one kind per task.
- With exactly two folders named `yes / no` (or `true / false`, `是 / 否`),
  the question is asked as a yes/no (noul) question.
- **How many**: at least 100 per answer; 300-1000 is safer. Keep the answers
  roughly balanced.

A CSV works too (columns `file,label`, optional `question` and `state`, so
each row can ask its own question), or a laya-omni jsonl (see the end).

## 2. One command

```bash
python scripts/finetune.py \
    --folders my_task/ --question "Is the part defective?" \
    --base runs/formal-v1-stage2 --laya models/laya-multilingual \
    --image-encoder models/siglip2-base-patch16-256 \
    --audio-encoder models/qwen3-asr-0.6b-audio-encoder \
    --out runs/my-task
```

It will:

1. hold out 20% per answer for validation (`--val-fraction`);
2. cache audio features, for audio tasks;
3. train from the general fusion `--base` (8 epochs by default), evaluating
   on the validation set after every epoch and **keeping the epoch with the
   lowest NLL** (early stopping);
4. fit temperatures on the validation set (calibration) and write them to
   `runs/my-task/fusion_config.json`;
5. print validation accuracy, next to the accuracy without the image or audio
   as a baseline.

A few hundred examples take a few minutes on one RTX 4090.

### Keeping general ability: replay

If the same fusion should still answer general questions, mix general data in:

```bash
    --replay 1000 \
    --replay-jsonl data/cauldron/vqav2.jsonl --replay-jsonl data/audio/esc50.jsonl \
    --audio-features cache/qwen3-asr
```

About 1000 general questions per epoch. Skip it if the fusion is only used for
this scenario.

## 3. Use

```python
from laya_omni import Omni

omni = Omni.load("runs/my-task", laya="models/laya-multilingual",
                 image_encoder="models/siglip2-base-patch16-256")
omni.predict("Image.", {"defect": {"type": "noul", "instructions": "Is the part defective?"}},
             image="new_part.jpg")
```

Asking the question the way it was trained works best; other questions still
work (better with replay).

## 4. What to expect

Snake frames from a game that was never in general training (4 options,
chance 0.25), same settings, best test accuracy:

| Training frames | from the general fusion | from scratch |
|---:|---:|---:|
| 100 | 0.36 | 0.38 |
| 500 | **0.55** | 0.35 |
| 2000 | **0.68** | 0.26 |

With `finetune.py`:

- snake, 500 frames (400 train / 100 validation) with VQAv2 and ESC-50
  replay: validation 0.55, ESC-50 still 0.96 (0.955 before), so general
  ability is kept;
- cough vs laughter, 200 clips (160 train / 40 validation): validation 1.00.

Tasks about what is in the picture or clip adapt fast; tasks that need
reasoning about rules (the snake's next move) need more examples.

## 5. Troubleshooting

- **Validation accuracy does not improve**: compare with the no-image
  baseline. If they are close, the answer may not be visible, or it needs
  reasoning about what happens next; such tasks need more data.
- **Good accuracy but high NLL**: small sets get overconfident. Early stopping
  already picks the epoch with the best NLL and calibration corrects the rest;
  more examples help.
- **Small text, tables, screens**: pass `detail=True` (256 tokens) at inference.
- **Several judgements in one scenario**: give each row its own question (CSV
  `question` column or jsonl); one fusion covers them all.

## Appendix: jsonl rows

One question per line:

```json
{"state": "Image.", "question": {"type": "choice", "instructions": "Which part is this?", "criteria": ["gear", "bearing", "nut"]}, "label": 1, "image": "imgs/0001.jpg"}
```

- `label` is the index of the right option (noul: 1 = yes, 0 = no);
- audio: `"audio": "clips/0001.wav"`; several images: `"images": ["a.jpg", "b.jpg"]`;
- add `"split": "train"` or `"test"` to choose the validation set yourself;
- `state` can carry text context (the ticket, sensor readings); the model
  weighs it together with the image or audio.
