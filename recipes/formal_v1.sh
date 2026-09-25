#!/bin/bash
# The first full laya-omni training run, on two GPUs.
#   stage 1 (alignment): image-caption and audio-caption matching, sound classes
#   stage 2 (mixture):   every question dataset, initialised from stage 1
# Usage: W=<work dir> bash recipes/formal_v1.sh [stage1|stage2|all]
set -e
cd "$(dirname "$0")/.."
W=${W:-$HOME/laya-omni-work}; D=$W/data
IMG="vqav2 vsr ai2d iconqa scienceqa st_vqa chartqa aokvqa infographic_vqa spot_the_diff hateful_memes visual7w cocoqa textvqa screen2words textcaps tallyqa docvqa mapqa nlvr2 ocrvqa"
AUD="esc50 cremad vocalsound clotho audiocaps clothoaqa minds14 slurp vggsound musicbench avqa"

EVAL="--holdout-jsonl $D/cauldron/tqa.jsonl --holdout-jsonl $D/cauldron/vqarad.jsonl --holdout-jsonl $D/audio/gtzan.jsonl"
EVAL="$EVAL --holdout-jsonl $D/eval/imagenet.jsonl --holdout-jsonl $D/eval/mmau.jsonl --holdout-jsonl $D/eval/songdescriber.jsonl"
COMMON="--laya $W/models/laya-multilingual --raw-images --image-encoder $W/models/siglip2-base-patch16-256
  --image-tokens 64,256 --audio-features $W/cache/qwen3-asr --lora 16 --batch 16 --accum 2 --workers 8
  --lr 5e-4 --lora-lr 1e-4 --warmup 1000 --probe-every 2000 --probe-limit 200 --eval-limit 1000"
RUN="env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. $W/venv/bin/torchrun --nproc_per_node 2 scripts/train.py"

stage1() {
  T="--jsonl $D/captions/cc3m.jsonl"
  for a in audiocaps clotho vggsound musicbench; do T="$T --jsonl $D/audio/$a.jsonl"; done
  $RUN $COMMON $T $EVAL --cap 200000 --epochs 1 --out $W/runs/formal-v1-stage1
}

stage2() {
  T=""
  for s in $IMG; do T="$T --jsonl $D/cauldron/$s.jsonl"; done
  T="$T --jsonl $D/gqa/gqa.jsonl --jsonl $D/omni/omniinstruct.jsonl --jsonl $D/captions/cc3m.jsonl"
  for a in $AUD; do T="$T --jsonl $D/audio/$a.jsonl"; done
  $RUN $COMMON $T $EVAL --cap 40000 --epochs 3 --init $W/runs/formal-v1-stage1 --out $W/runs/formal-v1-stage2
}

case "${1:-all}" in
  stage1) stage1 ;;
  stage2) stage2 ;;
  all) stage1; stage2 ;;
esac
