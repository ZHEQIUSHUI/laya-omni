# Reproducing formal-v2

Everything below ran on one machine with two RTX 4090s (24 GB), 128 CPU cores
and about 1 TB of free disk. Paths use `W=~/laya-omni-work` for downloads,
converted data and runs; run the scripts from the repository root with
`PYTHONPATH=.`.

## 1. Environment

```bash
python -m venv $W/venv && . $W/venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[train,web,dev]'
```

transformers 5.x is required (it ships `Qwen3ASREncoder`). Behind hf-mirror,
set `HF_ENDPOINT=https://hf-mirror.com` and `HF_HUB_DISABLE_XET=1` (the Xet
backend bypasses the mirror and fails with 401).

## 2. Models

```bash
hf download convaiinnovations/laya-multilingual --local-dir $W/models/laya-multilingual
hf download google/siglip2-base-patch16-256 --local-dir $W/models/siglip2-base-patch16-256
hf download Qwen/Qwen3-ASR-0.6B --local-dir $W/models/Qwen3-ASR-0.6B
python scripts/extract_audio_encoder.py $W/models/Qwen3-ASR-0.6B $W/models/qwen3-asr-0.6b-audio-encoder
```

Checks: `pytest tests/`, `python scripts/parity.py $W/models/laya-multilingual`
(needs `pip install laya`).

## 3. Data

Download into `$W/datasets/<repo name>` (`hf download --repo-type dataset ...`).
Datasets are not redistributed; only the converters ship.

| Repository | Files | Used for |
|---|---|---|
| HuggingFaceM4/the_cauldron | folders: vqav2 vsr ai2d iconqa scienceqa st_vqa chartqa aokvqa infographic_vqa spot_the_diff hateful_memes visual7w cocoqa textvqa screen2words textcaps tallyqa docvqa mapqa nlvr2 ocrvqa (training); tqa vqarad (held out) | image questions |
| HuggingFaceM4/the_cauldron `clevr` (from ModelScope `AI-ModelScope/the_cauldron` if the mirror redirects it) | folder: clevr | spatial and compositional questions (formal-v2) |
| lmms-lab/GQA | `train_balanced_*`, `testdev_balanced_*` | compositional image questions |
| pixparse/cc3m-wds | `cc3m-train-00[0-9][0-9].tar`, `cc3m-validation-*.tar` | image-caption matching |
| m-a-p/OmniInstruct | `data/*.parquet` | image + audio questions |
| ashraq/esc50, AbstractTTS/CREMA-D, lmms-lab/vocalsound, marsyas/gtzan | all | sounds, emotion, vocal sounds; genres (GTZAN, held out) |
| CLAPv2/Clotho, gijs/audiocaps, lmms-lab/ClothoAQA | all | audio captions and questions |
| txya900619/vggsound-16k, baijs/AudioSetCaps (`Dataset/*`) | all | sound classes, captions, speech/music presence |
| amaai-lab/MusicBench | all | music captions, tempo, key, metre |
| Fhrozen/FSD50k | `clips/`, `labels/` | 200 sound classes (formal-v2) |
| qmeeus/slurp, PolyAI/minds14 | all | spoken intents |
| gijs/avqa-processed | all | audio questions |
| timm/mini-imagenet (test), lmms-lab-audio/mmau (test_mini), renumics/song-describer-dataset | as listed | zero-shot evaluation |

Convert (image files and audio clips are written next to each jsonl):

```bash
D=$W/datasets; O=$W/data
python scripts/convert_cauldron.py $D/the_cauldron --out $O/cauldron   # add --subsets clevr for a separately downloaded CLEVR
python scripts/convert_gqa.py $D/GQA --out $O/gqa
python scripts/convert_captions.py $D/cc3m-wds --name cc3m --out $O/captions \
    --train "cc3m-train-*.tar" --test "cc3m-validation-*.tar" --encoder $W/models/siglip2-base-patch16-256
python scripts/convert_omni.py $D/OmniInstruct --out $O/omni
for a in esc50:esc50 cremad:CREMA-D vocalsound:vocalsound gtzan:gtzan clotho:Clotho audiocaps:audiocaps \
         clothoaqa:ClothoAQA vggsound:vggsound-16k musicbench:MusicBench slurp:slurp minds14:minds14 avqa:avqa-processed \
         fsd50k:FSD50k; do
  python scripts/convert_audio.py ${a%%:*} $D/${a#*:} --out $O/audio
done
python scripts/convert_audio.py mmau $D/mmau --out $O/eval
python scripts/convert_audio.py songdescriber $D/song-describer-dataset --out $O/eval
python scripts/convert_vqa.py imagenet <mini-imagenet test parquet dir> --out $O/eval --label-names <names.json>
```

Mini-ImageNet's class names come from timm's synset table
(`timm/data/_info/imagenet_synset_to_lemma.txt`).

Audio features are cached once (images are encoded during training):

```bash
for f in $O/audio/*.jsonl $O/omni/omniinstruct.jsonl $O/eval/mmau.jsonl $O/eval/songdescriber.jsonl; do
  python scripts/cache_features.py --modality audio --data $f --root $(dirname $f) \
      --encoder $W/models/qwen3-asr-0.6b-audio-encoder --out $W/cache/qwen3-asr/$(basename $f .jsonl)
done
```

## 4. Train

```bash
W=$W bash recipes/formal_v1.sh all      # stage 1 then stage 2, both on two GPUs
W=$W bash recipes/formal_v2.sh          # continues formal-v1 (adds CLEVR and FSD50K)
```

On consumer GPUs without peer-to-peer access (RTX 4090), `train.py` sets
`NCCL_P2P_DISABLE=1` when it runs under torchrun. `--probe-every` prints each
source's gain over text only during an epoch; a source stuck near zero early
on is a run worth restarting.

## 5. Evaluate and calibrate

```bash
python scripts/evaluate.py --laya $W/models/laya-multilingual --fusion $W/runs/formal-v2 \
    --image-encoder $W/models/siglip2-base-patch16-256 --audio-features $W/cache/qwen3-asr \
    --jsonl <each training set> --holdout <tqa, vqarad, gtzan, imagenet, mmau, songdescriber> \
    --tokens 64,256 --limit 1000 --calibrate
```

The temperatures go into the run's `fusion_config.json`.

## 6. Adapting to a new task

Use `scripts/finetune.py`: it builds the task data from folders, a CSV or a
jsonl, trains from the formal fusion with early stopping, and calibrates. See
[finetune.md](finetune.md).
