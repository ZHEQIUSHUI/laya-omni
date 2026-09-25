# Results

All numbers are for `formal-v1` (recipe: [`recipes/formal_v1.sh`](../recipes/formal_v1.sh)):
Laya multilingual (mmBERT-base), SigLIP 2 base/16 at 256 px, the Qwen3-ASR-0.6B
audio encoder, a fusion of about 7M trainable parameters (projectors, position
and modality embeddings, row-gated LoRA rank 16). Two stages on two RTX 4090s:
alignment (1 epoch, about 1 h) and the question mixture (3 epochs, about 5 h).

Each set is evaluated on up to 1000 test questions that were never trained on
(held-out and zero-shot sets: any 1000 of their rows). *Laya (text only)* asks
the same question without the image or audio: it can only use the wording of
the question and options, so the gap is what the image or audio adds. Chance is
0.25-0.5 depending on the number of options.

## Accuracy

| Set | Laya (text only) | laya-omni, 64 tokens | laya-omni, 256 tokens | NLL (64) |
|---|---:|---:|---:|---:|
| **Image: general questions** | | | | |
| vqav2 | 0.447 | **0.770** | 0.765 | 0.462 |
| gqa | 0.547 | **0.771** | 0.772 | 0.424 |
| cocoqa | 0.459 | **0.915** | 0.930 | 0.222 |
| visual7w | 0.284 | **0.741** | 0.765 | 0.634 |
| aokvqa | 0.246 | **0.567** | 0.581 | 1.083 |
| tallyqa | 0.261 | **0.883** | 0.886 | 0.317 |
| vsr | 0.486 | **0.584** | 0.562 | 0.673 |
| hateful_memes | 0.656 | **0.725** | 0.749 | 0.527 |
| **Image: text, documents, screens** | | | | |
| textvqa | 0.364 | **0.813** | 0.841 | 0.453 |
| st_vqa | 0.379 | **0.819** | 0.860 | 0.436 |
| ocrvqa | 0.441 | **0.901** | 0.899 | 0.234 |
| docvqa | 0.401 | **0.713** | 0.733 | 0.591 |
| infographic_vqa | 0.466 | **0.710** | 0.712 | 0.597 |
| screen2words | 0.460 | **0.894** | 0.917 | 0.289 |
| textcaps | 0.498 | **0.969** | 0.972 | 0.098 |
| **Image: charts, diagrams, maps** | | | | |
| chartqa | 0.359 | **0.798** | 0.814 | 0.456 |
| ai2d | 0.247 | **0.579** | 0.549 | 0.997 |
| scienceqa | 0.410 | **0.755** | 0.683 | 0.565 |
| iconqa | 0.361 | **0.792** | 0.795 | 0.389 |
| mapqa | 0.440 | **0.750** | 0.764 | 0.429 |
| **Image: several images** | | | | |
| nlvr2 | 0.487 | **0.686** | 0.671 | 0.556 |
| spot_the_diff | 0.517 | **0.772** | 0.768 | 0.472 |
| **Image: caption matching** | | | | |
| cc3m | 0.415 | **0.877** | 0.877 | 0.304 |
| **Audio: sounds and voices** | | | | |
| esc50 | 0.430 | **0.955** | – | 0.105 |
| vggsound | 0.426 | **0.836** | – | 0.404 |
| vocalsound | 0.416 | **0.953** | – | 0.144 |
| cremad | 0.434 | **0.878** | – | 0.310 |
| **Audio: captions and questions** | | | | |
| audiocaps | 0.458 | **0.906** | – | 0.243 |
| clotho | 0.426 | **0.828** | – | 0.397 |
| clothoaqa | 0.456 | **0.777** | – | 0.487 |
| avqa | 0.365 | **0.824** | – | 0.476 |
| **Audio: spoken intents** | | | | |
| slurp | 0.336 | **0.952** | – | 0.130 |
| minds14 | 0.409 | **0.960** | – | 0.127 |
| **Audio: music** | | | | |
| musicbench | 0.373 | **0.716** | – | 0.666 |
| **Image + audio** | | | | |
| omniinstruct | 0.456 | **0.844** | 0.842 | 0.395 |
| **Held out (never trained on)** | | | | |
| tqa | 0.261 | **0.346** | 0.330 | 1.597 |
| vqarad | 0.434 | **0.513** | 0.520 | 0.955 |
| gtzan | 0.405 | **0.601** | – | 0.976 |
| **Zero-shot benchmarks** | | | | |
| imagenet | 0.519 | **0.799** | 0.804 | 0.542 |
| mmau | 0.288 | **0.439** | – | 1.603 |
| songdescriber | 0.461 | **0.695** | – | 0.833 |

Reading the table:

- Every set gains from its image or audio. The largest gains are on counting
  (TallyQA +0.62), spoken intents (SLURP +0.62, MINDS-14 +0.55), reading text
  in images (TextVQA +0.45, OCR-VQA +0.46) and sound events (ESC-50 +0.53).
- The zero-shot sets were never seen in any form: object classification on
  Mini-ImageNet (+0.28), the MMAU audio benchmark (+0.15) and music caption
  matching on Song Describer (+0.23). GTZAN music genres gained +0.20 without
  any genre training, from MusicBench's captions.
- **256 tokens** (`detail=True`) help reading tasks (ST-VQA +0.04, TextVQA
  +0.03, screens, documents) and hurt science diagrams (ScienceQA -0.07,
  AI2D -0.03). 64 tokens stays the default.
- Weak spots: science diagrams (AI2D 0.58, the held-out TQA 0.35), spatial
  relations (VSR 0.58), and memes whose meaning lies in the image-text pairing
  (Hateful Memes, +0.07).

## Calibration

`scripts/evaluate.py --calibrate` fits one temperature per (modality, question
type, option count) on half of the in-distribution test rows and reports the
other half. Most temperatures are 1.2-1.5 (the model is somewhat
overconfident). Mean NLL on the unseen half:

| | before | after |
|---|---:|---:|
| 35 in-distribution sets | 0.434 | 0.425 |
| 6 held-out and zero-shot sets | 1.043 | **0.955** |

Accuracy does not change (a temperature keeps the ranking of options).

## Games, zero-shot and few-shot

Frames from the three generated games (`scripts/make_game_data.py`) were never
in formal training. Zero-shot, the model is near chance: games ask what the
rules imply next (will this move trap the snake, where will the ball land), not
what is in the picture.

| Game (options) | text only | laya-omni |
|---|---:|---:|
| snake (4) | 0.244 | 0.292 |
| bird (2) | 0.539 | 0.563 |
| bricks (3) | 0.485 | 0.331 |

Few-shot adaptation on snake, same settings (lr 5e-4, LoRA 1e-4), starting from
the formal fusion or from a fresh one; best test accuracy over the run:

| Training frames | from formal-v1 | from scratch |
|---:|---:|---:|
| 100 | 0.36 | 0.38 |
| 500 | **0.55** | 0.35 |
| 2000 | **0.68** (5 epochs) | 0.26 (did not start learning) |

The trained fusion learns a new visual task from a few hundred examples that a
fresh fusion cannot. Small sets get overconfident quickly (test NLL 4-6 after
many epochs), so few-shot use needs early stopping and a calibration pass.

## What did not work (ablations)

| Idea | Result |
|---|---|
| Flamingo-style gated cross-attention after the decision-head layers | Collapses to uniform probabilities: the head's residual stream runs at RMS 140-280 against O(1) updates, and every option marker reads the same memory at the start. Scaling updates by token RMS did not help. |
| Perceiver resampler (32 learned queries) in front of the encoder | Same start-up problem; a per-frame MLP projector overfits 256 frames to 99.6% in 400 steps where the resampler stays at chance. |
| Projector only (encoder and head frozen, no LoRA) | Enough for game frames, but underfits general visual questions (train loss stays near 1.1; A-OKVQA 0.32). Row-gated LoRA raised the mean over ten sets from 0.43 to 0.59. |
| Two image encoders (SigLIP 2 + Qwen3.5 vision) | Qwen3.5's tower is better at text and charts, SigLIP 2 at photos and diagrams; the mean was a tie (0.59 vs 0.58) and running both doubles the image cost, so laya-omni keeps SigLIP 2. |
| Learning rate 1e-3 / LoRA 2e-4 on the full mixture | One seed stopped using the images for AI2D, ChartQA and ScienceQA in its first epoch and never recovered (AI2D 0.44 vs 0.67); another seed was fine. The formal recipe uses 5e-4 / 1e-4, 1000 warm-up steps, and probes every 2000 steps. |
| FigureQA (synthetic charts, 1.3M questions) | Chance at 64 and at 256 tokens; dropped from training. |
