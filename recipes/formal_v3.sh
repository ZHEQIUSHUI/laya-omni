#!/bin/bash
# formal-v3: continue formal-v2 with COCO location / object / size questions and teacher soft labels.
#   + CLEVR (spatial and compositional reasoning), FSD50K (200 sound classes)
#   + science diagrams and spatial relations drawn 2-4x as often
#   + a higher per-source cap, lower learning rates (continued training)
# Usage: W=<work dir> bash recipes/formal_v2.sh
set -e
cd "$(dirname "$0")/.."
W=${W:-$HOME/laya-omni-work}; D=$W/data
IMG="vqav2 vsr ai2d iconqa scienceqa st_vqa chartqa aokvqa infographic_vqa spot_the_diff hateful_memes visual7w cocoqa textvqa screen2words textcaps tallyqa docvqa mapqa nlvr2 ocrvqa clevr"
AUD="esc50 cremad vocalsound clotho audiocaps clothoaqa minds14 slurp vggsound musicbench avqa fsd50k"
T=""
for s in $IMG; do T="$T --jsonl $D/cauldron/$s.jsonl"; done
T="$T --jsonl $D/gqa/gqa.jsonl --jsonl $D/omni/omniinstruct.jsonl --jsonl $D/captions/cc3m.jsonl"
for a in $AUD; do T="$T --jsonl $D/audio/$a.jsonl"; done
EVAL="--holdout-jsonl $D/cauldron/tqa.jsonl --holdout-jsonl $D/cauldron/vqarad.jsonl --holdout-jsonl $D/audio/gtzan.jsonl"
EVAL="$EVAL --holdout-jsonl $D/eval/imagenet.jsonl --holdout-jsonl $D/eval/mmau.jsonl --holdout-jsonl $D/eval/songdescriber.jsonl"
WEIGHTS="--weight ai2d=4 --weight scienceqa=4 --weight vsr=4 --weight iconqa=2"
# COCO questions from scripts/make_spatial.py (where things are, which objects, which is bigger)
for k in location relation identity size; do T="$T --jsonl $D/coco_spatial/coco_$k.jsonl"; done
# image-teacher soft labels (scripts/teacher_label.py, Qwen3.8-27B): KL on rows the teacher answers correctly
for f in $D/teacher/*.jsonl; do T="$T --teacher $f"; done

env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. $W/venv/bin/torchrun --nproc_per_node 2 scripts/train.py \
  --laya $W/models/laya-multilingual --raw-images --image-encoder $W/models/siglip2-base-patch16-256 \
  --image-tokens 64,256 --audio-features $W/cache/qwen3-asr --lora 16 --batch 16 --accum 2 --workers 8 \
  --lr 2e-4 --lora-lr 4e-5 --warmup 500 --probe-every 3000 --probe-limit 200 --eval-limit 1000 \
  $T $EVAL $WEIGHTS --cap 80000 --epochs 2 --init $W/runs/formal-v2 --out $W/runs/formal-v3
