# Results

The released fusion is `formal-v2`: Laya multilingual (mmBERT-base), SigLIP 2
base/16 at 256 px, the Qwen3-ASR-0.6B audio encoder, and a fusion of about 7M
trainable parameters (projectors, position and modality embeddings, row-gated
LoRA rank 16), trained on two RTX 4090s in three steps:

1. `formal-v1` stage 1, alignment on caption matching and audio captions (1 epoch, about 1 h);
2. `formal-v1` stage 2, the question mixture (3 epochs, about 5 h)
   ([`recipes/formal_v1.sh`](../recipes/formal_v1.sh));
3. `formal-v2`, continued from `formal-v1` with CLEVR and FSD50K added, science
   diagrams and spatial relations drawn 2-4x as often, and lower learning rates
   (2 epochs, about 6 h; [`recipes/formal_v2.sh`](../recipes/formal_v2.sh)).
   The first of the two epochs had the lower validation NLL and is the release;
   the second was 0.006 more accurate on average but overconfident on the
   held-out sets (TQA NLL 2.1 to 3.4).

Each set is evaluated on up to 1000 test questions that were never trained on
(held-out and zero-shot sets: any 1000 of their rows). *Laya (text only)* asks
the same question without the image or audio: it can only use the wording of
the question and options, so the gap is what the image or audio adds. Chance is
0.25-0.5 depending on the number of options.

## Accuracy

| Set | Laya (text only) | v1 | **v2**, 64 tokens | v2, 256 tokens | NLL (v2, 64) |
|---|---:|---:|---:|---:|---:|
| **Image: general questions** | | | | | |
| vqav2 | 0.447 | 0.770 | **0.777** | 0.777 | 0.451 |
| gqa | 0.547 | 0.771 | **0.782** | 0.803 | 0.403 |
| cocoqa | 0.459 | 0.915 | **0.922** | 0.935 | 0.203 |
| visual7w | 0.284 | 0.741 | **0.753** | 0.777 | 0.607 |
| aokvqa | 0.246 | 0.567 | **0.562** | 0.601 | 1.071 |
| tallyqa | 0.261 | 0.883 | **0.884** | 0.900 | 0.314 |
| vsr | 0.486 | 0.584 | **0.616** | 0.654 | 0.683 |
| hateful_memes | 0.656 | 0.725 | **0.730** | 0.751 | 0.535 |
| clevr | 0.400 | – | **0.729** | 0.721 | 0.457 |
| **Image: text, documents, screens** | | | | | |
| textvqa | 0.364 | 0.813 | **0.817** | 0.845 | 0.434 |
| st_vqa | 0.379 | 0.819 | **0.827** | 0.859 | 0.406 |
| ocrvqa | 0.441 | 0.901 | **0.894** | 0.909 | 0.227 |
| docvqa | 0.401 | 0.713 | **0.715** | 0.748 | 0.575 |
| infographic_vqa | 0.466 | 0.710 | **0.707** | 0.712 | 0.614 |
| screen2words | 0.460 | 0.894 | **0.870** | 0.902 | 0.340 |
| textcaps | 0.498 | 0.969 | **0.970** | 0.971 | 0.098 |
| **Image: charts, diagrams, maps** | | | | | |
| chartqa | 0.359 | 0.798 | **0.816** | 0.821 | 0.434 |
| ai2d | 0.247 | 0.579 | **0.622** | 0.616 | 1.220 |
| scienceqa | 0.410 | 0.755 | **0.803** | 0.786 | 0.440 |
| iconqa | 0.361 | 0.792 | **0.830** | 0.827 | 0.334 |
| mapqa | 0.440 | 0.750 | **0.773** | 0.781 | 0.389 |
| **Image: several images** | | | | | |
| nlvr2 | 0.487 | 0.686 | **0.701** | 0.713 | 0.535 |
| spot_the_diff | 0.517 | 0.772 | **0.770** | 0.778 | 0.470 |
| **Image: caption matching** | | | | | |
| cc3m | 0.415 | 0.877 | **0.879** | 0.884 | 0.297 |
| **Audio: sounds and voices** | | | | | |
| esc50 | 0.430 | 0.955 | **0.964** | – | 0.095 |
| fsd50k | 0.447 | – | **0.864** | – | 0.347 |
| vggsound | 0.426 | 0.836 | **0.837** | – | 0.412 |
| vocalsound | 0.416 | 0.953 | **0.947** | – | 0.155 |
| cremad | 0.434 | 0.878 | **0.886** | – | 0.297 |
| **Audio: captions and questions** | | | | | |
| audiocaps | 0.458 | 0.906 | **0.907** | – | 0.222 |
| clotho | 0.426 | 0.828 | **0.842** | – | 0.363 |
| clothoaqa | 0.456 | 0.777 | **0.797** | – | 0.462 |
| avqa | 0.365 | 0.824 | **0.816** | – | 0.493 |
| **Audio: spoken intents** | | | | | |
| slurp | 0.336 | 0.952 | **0.956** | – | 0.122 |
| minds14 | 0.409 | 0.960 | **0.954** | – | 0.145 |
| **Audio: music** | | | | | |
| musicbench | 0.373 | 0.716 | **0.711** | – | 0.650 |
| **Image + audio** | | | | | |
| omniinstruct | 0.456 | 0.844 | **0.845** | 0.849 | 0.389 |
| **Held out (never trained on)** | | | | | |
| tqa | 0.261 | 0.346 | **0.385** | 0.375 | 2.146 |
| vqarad | 0.434 | 0.513 | **0.521** | 0.504 | 0.952 |
| gtzan | 0.405 | 0.601 | **0.636** | – | 0.950 |
| **Zero-shot benchmarks** | | | | | |
| imagenet | 0.519 | 0.799 | **0.793** | 0.803 | 0.584 |
| mmau | 0.288 | 0.439 | **0.446** | – | 1.540 |
| songdescriber | 0.461 | 0.695 | **0.681** | – | 0.797 |

Reading the table:

- Mean over the 41 sets both versions share: formal-v1 0.771, formal-v2
  **0.779**. The targeted sets gained most: ScienceQA +0.05, AI2D +0.04,
  IconQA +0.04, TQA (held out) +0.04, VSR +0.03, GTZAN (held out) +0.035.
  Nothing dropped by more than 0.025 (Screen2Words).
- Every set gains from its image or audio. The largest gains are on counting
  (TallyQA +0.62), spoken intents (SLURP +0.62, MINDS-14 +0.55), reading text
  in images (TextVQA +0.45, OCR-VQA +0.45) and sound events (ESC-50 +0.53).
- The zero-shot sets were never seen in any form: object classification on
  Mini-ImageNet (+0.27), the MMAU audio benchmark (+0.16) and music caption
  matching on Song Describer (+0.22). GTZAN music genres gained +0.23 without
  any genre training, from MusicBench's captions.
- **256 tokens** (`detail=True`) help reading tasks (DocVQA +0.03, ST-VQA
  +0.03, TextVQA +0.03, Screen2Words +0.03), spatial relations (VSR +0.04)
  and open questions (A-OKVQA +0.04); science diagrams are slightly worse
  (ScienceQA -0.02). 64 tokens stays the default.
- Weak spots: science diagrams (AI2D 0.62, the held-out TQA 0.39), medical
  images (the held-out VQA-RAD 0.52), spatial relations (VSR 0.62), and memes
  whose meaning lies in the image-text pairing (Hateful Memes, +0.07).

## Calibration

`scripts/evaluate.py --calibrate` fits one temperature per (modality, question
type, option count) on half of the in-distribution test rows and reports the
other half. Most temperatures are 1.2-1.4 (the model is somewhat
overconfident). Mean NLL on the unseen half:

| | formal-v1 before | after | formal-v2 before | after |
|---|---:|---:|---:|---:|
| in-distribution sets (35 / 37) | 0.434 | 0.425 | 0.432 | 0.422 |
| 6 held-out and zero-shot sets | 1.043 | 0.955 | 1.115 | **1.015** |

formal-v2 is more accurate on the held-out sets but more confident when it is
wrong, mostly on TQA (NLL 1.93 after calibration). Accuracy does not change (a temperature keeps the ranking of options).

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

Few-shot adaptation on snake (measured with formal-v1), same settings (lr 5e-4, LoRA 1e-4), starting from
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
