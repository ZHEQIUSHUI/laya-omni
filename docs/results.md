# Results

The released fusion is `formal-v3` (formal-v2 plus COCO questions and teacher soft labels, below): Laya multilingual (mmBERT-base), SigLIP 2
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

## Standard benchmarks

Benchmarks nobody trained on (converted by `scripts/convert_bench.py`, run by
`scripts/bench.py`). Every model answers the same questions one at a time on
one RTX 4090, timed from the image or audio file to the answer. Qwen3.5 sees
the options as lettered lines and the standard instruction; its answer is the
option letter (or Yes / No) with the highest next-token logit, one forward pass,
no decoding. Valen is its public checkpoint Valen-Preview-0923 in its pinned
environment (torch 2.6, transformers 5.4); through this harness it scores
0.738 on its own set, where its evaluation script gives 0.737.
*Laya (text only)* answers without the image or audio: near chance, so these
sets do need the picture or clip.

Accuracy / median latency (ms). laya-omni formal-v3 runs at 256 image tokens with
the LoRA deltas merged (its default); formal-v2 at 64 tokens as first released:

| Benchmark | Laya (text only) | formal-v2 | **formal-v3** | Qwen3.5-0.8B | Qwen3.5-2B | Valen | Qwen2.5-Omni-3B |
|---|---:|---:|---:|---:|---:|---:|---:|
| MMBench-EN dev | 0.283 | 0.629 / 21 | 0.664 / 14 | 0.754 / 42 | **0.837** / 46 | 0.734 / 70 | – |
| &nbsp; CircularEval | 0.129 | 0.559 | 0.595 | 0.571 | **0.722** | 0.623 | – |
| MMBench-CN dev | 0.277 | 0.582 / 21 | 0.598 / 14 | 0.747 / 42 | **0.820** / 45 | 0.731 / 70 | – |
| &nbsp; CircularEval | 0.104 | 0.493 | 0.517 | 0.580 | **0.710** | 0.640 | – |
| MMStar | 0.249 | 0.358 / 21 | 0.372 / 14 | 0.456 / 42 | **0.528** / 46 | 0.437 / 70 | – |
| MME (yes / no) | 0.503 | 0.655 / 23 | 0.677 / 17 | 0.755 / 48 | **0.815** / 62 | 0.606 / 105 | – |
| &nbsp; acc+ (both questions on an image) | 0.043 | 0.349 | 0.398 | 0.525 | **0.644** | 0.212 | – |
| POPE | 0.500 | 0.828 / 22 | 0.840 / 16 | 0.879 / 44 | **0.900** / 51 | 0.818 / 88 | – |
| SEED-Bench (image) | 0.282 | 0.536 / 26 | 0.573 / 20 | 0.719 / 61 | **0.767** / 82 | 0.664 / 222 | – |
| Valen-Eval-General-5k | 0.315 | 0.614 / 26 | 0.629 / 19 | 0.704 / 50 | **0.743** / 61 | 0.738 / 207 | – |
| OmniBench (image + audio) | 0.271 | 0.398 / 70 | 0.384 / 50 | – | – | – | **0.419** / 321 |
| MMAU test-mini (audio) | 0.289 | 0.448 / 30 | 0.449 / 20 | – | – | – | **0.668** / 91 |
| COCO location, held-out images (4 options) | – | 0.295 | **0.845** / 16 | – | – | – | – |
| COCO left / right / above | – | 0.493 | **0.851** / 16 | – | – | – | – |
| COCO which is bigger | – | 0.489 | **0.941** / 16 | – | – | – | – |
| COCO which object is present | – | 0.713 | **0.959** / 16 | – | – | – | – |

Some comparison runs shared the machine with other jobs, so latencies are indicative.

- laya-omni formal-v3 is about three times as fast as Qwen3.5-0.8B and 4-15 points less
  accurate on the image benchmarks; on audio, Qwen2.5-Omni-3B is well ahead (MMAU).
  The gap is widest on fine-grained perception (SEED-Bench, MMBench-CN).
- Valen's MME answers lean heavily to "no" (acc+ 0.21), and on MMBench it
  scores 0.73 where its own base model, Qwen3.5-2B, scores 0.84. Its public
  checkpoint was tuned further on Sokoban; the general checkpoint behind its
  reported 78.4% is not released.
- Overlap with training: 415 MMBench dev questions come from ScienceQA, which
  laya-omni trained on; without them laya-omni scores 0.610 on MMBench-EN
  (Qwen3.5 and Valen change by under 0.01). Valen-Eval-General-5k draws its
  questions from the training splits of VQAv2, GQA, TextVQA, DocVQA and others;
  611 of its 5000 questions (same image and question) are in laya-omni's
  training data, another 206 share only the image (`scripts/check_overlap.py`).

## In-distribution test splits

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
| A larger image encoder (SigLIP 2 so400m/16 at 256 px) | Same short from-scratch run as SigLIP 2 base: MMBench-EN 0.534 vs 0.514, SEED 0.423 vs 0.444, MMStar 0.307 vs 0.307, POPE 0.773 vs 0.765. No gain overall, 15% slower; the encoder is not the bottleneck. (Base at 384 px diverged in the same run and is not conclusive.) |
| Larger LoRA rank (64, 128 instead of 16) | Short runs from scratch: standard-benchmark mean 0.449 (r64) and 0.471 (r128) against 0.518 (r16). |
| Fully fine-tuning a copy of Laya's encoder and head (125M trainable instead of 7M) | One epoch from formal-v3: benchmark mean 0.623 against 0.621 for LoRA, same in-distribution accuracy; trainable capacity is not the bottleneck. |
| Tiles (whole image + 2x2 crops, 64 tokens each) | Within a point of 256 tokens on every benchmark, in both directions, and 3-6 ms slower. |
| Teacher soft labels on the training questions (Qwen3.8-27B) | No accuracy change against the same run without them (the teacher agrees with the labels on 90%+); lower held-out NLL in the probe (1.20 vs 1.30), not after calibration in formal-v3. |
| Questions from COCO boxes, one epoch | No change on the standard benchmarks (seed noise is about a point), but location 0.31 to 0.68, size 0.55 to 0.94 and object identity 0.71 to 0.95 on held-out COCO images; kept in formal-v3. |
