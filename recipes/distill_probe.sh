#!/bin/bash
# Does distilling a large VLM teacher help? Two runs continue formal-v2 for one epoch on
# the same mixture, one with the teacher's option probabilities (scripts/teacher_label.py
# output in $W/data/teacher) and one without.
# Usage: W=<work dir> bash recipes/distill_probe.sh <control|distill> <gpu>
set -e
cd "$(dirname "$0")/.."
W=${W:-$HOME/laya-omni-work}; D=$W/data
MODE=$1; GPU=${2:-0}
IMG="vqav2 vsr ai2d iconqa scienceqa st_vqa chartqa aokvqa infographic_vqa spot_the_diff hateful_memes visual7w cocoqa textvqa screen2words textcaps tallyqa docvqa mapqa nlvr2 ocrvqa clevr"
AUD="esc50 cremad vocalsound clotho audiocaps clothoaqa minds14 slurp vggsound musicbench avqa fsd50k"
T=""
for s in $IMG; do T="$T --jsonl $D/cauldron/$s.jsonl"; done
T="$T --jsonl $D/gqa/gqa.jsonl --jsonl $D/omni/omniinstruct.jsonl --jsonl $D/captions/cc3m.jsonl"
for a in $AUD; do T="$T --jsonl $D/audio/$a.jsonl"; done
EVAL="--holdout-jsonl $D/cauldron/tqa.jsonl --holdout-jsonl $D/cauldron/vqarad.jsonl --holdout-jsonl $D/audio/gtzan.jsonl"
EVAL="$EVAL --holdout-jsonl $D/eval/imagenet.jsonl --holdout-jsonl $D/eval/mmau.jsonl --holdout-jsonl $D/eval/songdescriber.jsonl"
TEACH=""
if [ "$MODE" = distill ]; then for f in $D/teacher/*.jsonl; do TEACH="$TEACH --teacher $f"; done; fi

env CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. $W/venv/bin/python scripts/train.py \
  --laya $W/models/laya-multilingual --raw-images --image-encoder $W/models/siglip2-base-patch16-256 \
  --image-tokens 64,256 --audio-features $W/cache/qwen3-asr --lora 16 --batch ${BATCH:-16} --accum ${ACCUM:-2} --workers 12 \
  --lr 2e-4 --lora-lr 4e-5 --warmup 300 --probe-every 3000 --probe-limit 200 --eval-limit 1000 \
  $T $EVAL $TEACH --cap 40000 --epochs 1 --seed 0 --init $W/runs/formal-v2 --out $W/runs/probe-$MODE
