"""Run a frozen encoder once over every image in a dataset and store the features.

Features go to one float16 .npy per dataset (N, T, D) plus a row index, so
training reads them by memory map instead of decoding and encoding images
every epoch.

    python scripts/cache_features.py --data data/games/snake.jsonl --root data/games \
        --encoder models/siglip2-base-patch16-256 --out cache/siglip2/snake
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from laya_omni.encoders import ImageEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl rows with an 'image' path")
    ap.add_argument("--root", default=".", help="directory image paths are relative to")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True, help="output prefix: <out>.npy and <out>.json")
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data)]
    paths = sorted({r["image"] for r in rows if r.get("image")})
    enc = ImageEncoder(args.encoder, device=args.device, pool=args.pool)
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
