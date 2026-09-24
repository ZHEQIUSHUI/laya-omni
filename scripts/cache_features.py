"""Run a frozen encoder once over every image or audio clip in a dataset and store the features.

Images: one float16 .npy per dataset, (N, T, D). Audio clips have different
lengths, so their frames are concatenated into one (total frames, D) array and
the .json index records where each clip starts and ends. Training reads either
by memory map instead of decoding and encoding every epoch.

    python scripts/cache_features.py --data data/games/snake.jsonl --root data/games \
        --encoder models/siglip2-base-patch16-256 --out cache/siglip2/snake
    python scripts/cache_features.py --modality audio --data data/audio/esc50.jsonl --root data/audio \
        --encoder models/qwen3-asr-0.6b-audio-encoder --out cache/qwen3-asr/esc50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from laya_omni.encoders import AudioEncoder, image_encoder, load_audio


def cache_spans(args, paths, enc, load):
    """Variable-length features (audio, native-resolution images): frames concatenated,
    with a [start, end) span per file."""
    chunks, spans, start = [], {}, 0
    for i in range(0, len(paths), args.batch):
        chunk = paths[i : i + args.batch]
        for p, h in zip(chunk, enc([load(Path(args.root) / p) for p in chunk])):
            chunks.append(h.float().cpu().numpy().astype(np.float16))
            spans[p] = [start, start + h.shape[0]]
            start += h.shape[0]
        print(f"\r{i + len(chunk)}/{len(paths)}", end="", flush=True)
    feats = np.concatenate(chunks)
    out = Path(args.out)
    np.save(out.with_suffix(".npy"), feats)
    meta = {"encoder": args.encoder, "shape": list(feats.shape), "spans": spans}
    out.with_suffix(".json").write_text(json.dumps(meta))
    print(f"\n{out.with_suffix('.npy')} {feats.shape}, {len(paths)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl rows with an 'image' or 'audio' path")
    ap.add_argument("--modality", choices=("image", "audio"), default="image")
    ap.add_argument("--audio-output", choices=("hidden", "projected"), default="hidden")
    ap.add_argument("--root", default=".", help="directory image paths are relative to")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True, help="output prefix: <out>.npy and <out>.json")
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.modality == "audio":
        paths = sorted({p for r in rows for p in ([r["audio"]] if r.get("audio") else r.get("audios", []))})
        return cache_spans(args, paths, AudioEncoder(args.encoder, device=args.device, output=args.audio_output), load_audio)
    paths = sorted({p for r in rows for p in ([r["image"]] if r.get("image") else r.get("images", []))})
    enc = image_encoder(args.encoder, device=args.device, pool=args.pool)
    if not hasattr(enc, "pool"):  # native resolution: token counts differ per image
        return cache_spans(args, paths, enc, lambda p: Image.open(p).convert("RGB"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats = None
    for start in range(0, len(paths), args.batch):
        chunk = paths[start : start + args.batch]
        images = [Image.open(Path(args.root) / p).convert("RGB") for p in chunk]
        h = enc(images).float().cpu().numpy().astype(np.float16)
        if feats is None:
            feats = np.lib.format.open_memmap(out.with_suffix(".npy"), "w+", np.float16, (len(paths), *h.shape[1:]))
        feats[start : start + len(chunk)] = h
        print(f"\r{start + len(chunk)}/{len(paths)}", end="", flush=True)
    feats.flush()
    meta = {"encoder": args.encoder, "pool": args.pool, "shape": list(feats.shape), "index": {p: i for i, p in enumerate(paths)}}
    out.with_suffix(".json").write_text(json.dumps(meta))
    print(f"\n{out.with_suffix('.npy')} {feats.shape}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
