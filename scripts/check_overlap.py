"""Which evaluation rows did laya-omni see in training? Same image (perceptual hash,
robust to re-encoding) and, for those, the same question text.

    python scripts/check_overlap.py --eval data/bench/valen_general.jsonl \\
        --train data/cauldron/vqav2.jsonl data/gqa/gqa.jsonl ... --out data/bench/valen_general.overlap.json

Writes {row id: "question" | "image"} for rows whose image appears in a training
row (training splits only): "question" when the question text matches too.
"""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def dhash(path, size=16):
    try:
        img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    except Exception:
        return None
    a = np.asarray(img, dtype=np.int16)
    return np.packbits(a[:, 1:] > a[:, :-1]).tobytes().hex()


def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def rows(path, train_only):
    root = Path(path).parent
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if train_only and r.get("split") != "train":
            continue
        imgs = r.get("images") or ([r["image"]] if r.get("image") else [])
        yield r, [str(root / p) for p in imgs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    questions = {}  # image hash -> normalized training questions
    paths, texts = [], []
    for p in args.train:
        for r, imgs in rows(p, train_only=True):
            for img in imgs:
                paths.append(img)
                texts.append(norm(r["question"]["instructions"]))
    unique = sorted(set(paths))
    with ProcessPoolExecutor(args.workers) as ex:
        hashes = dict(zip(unique, ex.map(dhash, unique, chunksize=256)))
    for path, text in zip(paths, texts):
        h = hashes.get(path)
        if h:
            questions.setdefault(h, set()).add(text)
    print(f"{len(unique)} training images, {len(questions)} distinct hashes")

    evals = list(rows(args.eval, train_only=False))
    with ProcessPoolExecutor(args.workers) as ex:
        eh = list(ex.map(dhash, [imgs[0] if imgs else "" for _, imgs in evals], chunksize=64))
    seen = {}
    for (r, _), h in zip(evals, eh):
        if h and h in questions:
            seen[str(r["id"])] = "question" if norm(r["question"]["instructions"]) in questions[h] else "image"
    Path(args.out).write_text(json.dumps(seen, indent=0) + "\n")
    n_q = sum(v == "question" for v in seen.values())
    print(f"{len(evals)} eval rows: {n_q} same image and question, {len(seen) - n_q} same image only")


if __name__ == "__main__":
    main()
