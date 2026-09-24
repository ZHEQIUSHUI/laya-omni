"""Turn image-caption pairs (CLIP-style data, e.g. CC3M webdataset shards) into
laya-omni matching questions: which caption belongs to this image (choice), and
does this caption describe it (noul).

Random wrong captions are too easy ("a cat" vs "a truck"), so half the negatives
are hard: captions whose SigLIP text embedding is close to the true one ("a white
cat on a sofa" vs "a white dog on a sofa"). CC3M's captions are hypernymised
("actor attends the premiere"), so near-copies of the true caption would be
false negatives: neighbours with the same normalised text or a cosine above
`--max-sim` are skipped, and hard negatives come from ranks 5-30.

    python scripts/convert_captions.py datasets/cc3m-wds --name cc3m --out data/captions \\
        --train "cc3m-train-*.tar" --test "cc3m-validation-*.tar" --encoder models/siglip2-base-patch16-256
"""

import argparse
import glob
import io
import json
import random
import re
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

STATE = "Image."
PICK_ASK = [
    "Which caption describes this image?",
    "Which of these fits the picture?",
    "What does the image show?",
    "哪一条描述符合这张图？",
]
MATCH_ASK = [
    "Does this caption describe the image?",
    "Is this an accurate caption for the picture?",
    "这条描述和图片相符吗？",
]


def normalise(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def read_shards(pattern, src, out_dir, name, limit):
    """(image path, caption) pairs from webdataset tars, images resized and saved."""
    pairs = []
    for shard in sorted(glob.glob(str(Path(src) / pattern))):
        current = {}
        with tarfile.open(shard, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                key, _, ext = member.name.rpartition(".")
                if ext not in ("jpg", "jpeg", "png", "webp", "txt"):
                    continue
                current.setdefault(key, {})[ext] = tar.extractfile(member).read()
                item = current[key]
                image = item.get("jpg") or item.get("jpeg") or item.get("png") or item.get("webp")
                if image is not None and "txt" in item:
                    del current[key]
                    caption = item["txt"].decode("utf-8", "ignore").strip()
                    if len(caption.split()) < 3:
                        continue
                    dest = out_dir / f"{Path(shard).stem}_{Path(key).name}.jpg"
                    try:
                        if not dest.exists():
                            img = Image.open(io.BytesIO(image)).convert("RGB")
                            img.thumbnail((384, 384))
                            img.save(dest, quality=90)
                    except Exception:
                        continue
                    pairs.append((f"{name}/{dest.name}", caption))
        print(f"\r{shard}: {len(pairs)} pairs", end="", flush=True)
        if limit and len(pairs) >= limit:
            break
    print()
    return pairs[:limit] if limit else pairs


@torch.inference_mode()
def text_embeddings(captions, encoder, device, batch=1024):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder, torch_dtype=torch.float16).to(device).eval()
    out = []
    for i in range(0, len(captions), batch):
        # SigLIP was trained on max_length padding; shorter padding shifts its embeddings.
        t = tok(captions[i : i + batch], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        e = model.get_text_features(input_ids=t["input_ids"].to(device))
        e = e if isinstance(e, torch.Tensor) else e.pooler_output  # transformers 5 returns an output object
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).half())
    return torch.cat(out)


@torch.inference_mode()
def neighbours(emb, captions, k=30, max_sim=0.9, chunk=4096):
    """For each caption, up to k other captions that are close but not near-copies."""
    norm = [normalise(c) for c in captions]
    result = []
    for i in range(0, emb.shape[0], chunk):
        sims = emb[i : i + chunk] @ emb.T
        vals, idx = sims.float().topk(min(k * 3, emb.shape[0]), dim=1)
        for row, (v, j) in enumerate(zip(vals.tolist(), idx.tolist())):
            me = i + row
            keep = [b for a, b in zip(v, j) if b != me and a < max_sim and norm[b] != norm[me]]
            result.append(keep[:k])
    return result


def questions(pairs, near, rng):
    captions = [c for _, c in pairs]
    norm = [normalise(c) for c in captions]
    for i, (image, caption) in enumerate(pairs):
        hard = near[i][4:] or near[i]

        def negative():
            if hard and rng.random() < 0.5:
                return captions[rng.choice(hard)]
            while True:
                j = rng.randrange(len(captions))
                if norm[j] != norm[i]:
                    return captions[j]

        options = list(dict.fromkeys(negative() for _ in range(rng.randint(1, 3))))
        options = [o for o in options if o != caption] + [caption]
        rng.shuffle(options)
        yield image, {"type": "choice", "instructions": rng.choice(PICK_ASK), "criteria": options}, options.index(caption)
        text = caption if rng.random() < 0.5 else negative()
        yield image, {"type": "noul", "instructions": f"{rng.choice(MATCH_ASK)} \"{text}\""}, int(text == caption)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="data/captions")
    ap.add_argument("--train", required=True, help="glob of training shards")
    ap.add_argument("--test", required=True, help="glob of test shards")
    ap.add_argument("--encoder", required=True, help="SigLIP checkpoint for mining hard negatives")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=10000)
    ap.add_argument("--max-sim", type=float, default=0.9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    img_dir = out / args.name
    img_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with open(out / f"{args.name}.jsonl", "w") as f:
        for split, pattern, limit in (("train", args.train, args.max_train), ("test", args.test, args.max_test)):
            pairs = read_shards(pattern, args.src, img_dir, args.name, limit)
            if not pairs:
                continue
            emb = text_embeddings([c for _, c in pairs], args.encoder, args.device)
            near = neighbours(emb, [c for _, c in pairs], max_sim=args.max_sim)
            for image, q, y in questions(pairs, near, rng):
                row = {"state": STATE, "question": q, "label": y, "image": image, "split": split}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[f"{split}:{q['type']}"] += 1
    print(args.name, dict(counts))


if __name__ == "__main__":
    main()
