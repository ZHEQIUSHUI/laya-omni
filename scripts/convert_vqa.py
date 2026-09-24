"""Convert public image-question datasets into laya-omni rows.

Each dataset becomes <out>/<name>.jsonl plus extracted images under <out>/<name>/.
Rows look like {"state", "question", "label", "image", "split"}; see laya_omni/data.py.
Datasets are downloaded separately (Hugging Face parquet); only this converter ships.

    python scripts/convert_vqa.py aokvqa datasets/A-OKVQA --out data/vqa
"""

import argparse
import glob
import hashlib
import io
import json
import random
import re
import zlib
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

PHOTO = "Image."


def parquet_rows(src, pattern="**/*.parquet"):
    for path in sorted(glob.glob(str(Path(src) / pattern), recursive=True)):
        split = "test" if re.search(r"(val|test)", Path(path).name) else "train"
        table = pq.read_table(path)
        for row in table.to_pylist():
            yield split, path, row


def save_image(value, out_dir):
    """HF image cells are {"bytes", "path"}; store a 384px-max JPEG (SigLIP sees 256).

    Files are named by content, so an image shared by many questions is stored and
    encoded once.
    """
    if value is None:
        return None
    data = value.get("bytes") if isinstance(value, dict) else value
    if not data:
        return None
    path = out_dir / f"{hashlib.md5(data).hexdigest()[:16]}.jpg"
    if not path.exists():
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((384, 384))
        img.save(path, quality=90)
    return path


def aokvqa(src, rng):
    for split, _, r in parquet_rows(src):
        if r.get("correct_choice_idx") is None:
            continue
        yield split, r["image"], {
            "state": PHOTO,
            "question": {"type": "choice", "instructions": r["question"], "criteria": list(r["choices"])},
            "label": int(r["correct_choice_idx"]),
        }


def scienceqa(src, rng):
    for split, _, r in parquet_rows(src):
        if not r.get("image") or not (2 <= len(r["choices"]) <= 5):
            continue
        yield split, r["image"], {
            "state": r.get("hint") or PHOTO,
            "question": {"type": "choice", "instructions": r["question"], "criteria": list(r["choices"])},
            "label": int(r["answer"]),
        }


NUMBER_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}


def vqav2(src, rng):
    rows = list(parquet_rows(src))
    pools = defaultdict(set)  # question type -> answers seen, for plausible distractors
    for _, _, r in rows:
        if r.get("answer_type") == "other":
            pools[r.get("question_type", "")].add(r["multiple_choice_answer"])
    for split, _, r in rows:
        answer, kind = r["multiple_choice_answer"], r.get("answer_type")
        # VQA splits are by image already; keep a tenth of the val images for our test split.
        split = "test" if zlib.crc32(str(r.get("image_id")).encode()) % 10 == 0 else "train"
        if kind == "yes/no" and answer in ("yes", "no"):
            q = {"type": "noul", "instructions": r["question"]}
            yield split, r["image"], {"state": PHOTO, "question": q, "label": int(answer == "yes")}
        elif kind == "number":
            n = int(answer) if answer.isdigit() else NUMBER_WORDS.get(answer)
            if n is None or n > 20:
                continue
            near = [m for m in range(max(0, n - 3), n + 4) if m != n]
            options = rng.sample(near, rng.randint(1, 3)) + [n]
            rng.shuffle(options)
            q = {"type": "choice", "instructions": r["question"], "criteria": [str(m) for m in options]}
            yield split, r["image"], {"state": PHOTO, "question": q, "label": options.index(n)}
        elif kind == "other":
            pool = sorted(pools[r.get("question_type", "")] - {answer})
            if len(pool) < 3 or len(answer) > 40:
                continue
            options = rng.sample(pool, rng.randint(1, 3)) + [answer]
            rng.shuffle(options)
            q = {"type": "choice", "instructions": r["question"], "criteria": options}
            yield split, r["image"], {"state": PHOTO, "question": q, "label": options.index(answer)}


CLASSIFY = [
    "What is shown in this image?",
    "Which of these is in the picture?",
    "What is the main object in the photo?",
    "Identify the subject of the image.",
    "图里是什么？",
    "这张照片的主体是哪一个？",
]


def imagenet(src, rng, names=None):
    """Classification as choice: the true class plus 1-3 random others, varied phrasing."""
    names = names or {}
    rows = list(parquet_rows(src))
    labels = sorted({r["label"] for _, _, r in rows})
    for split, _, r in rows:
        others = rng.sample([l for l in labels if l != r["label"]], rng.randint(1, 3))
        options = others + [r["label"]]
        rng.shuffle(options)
        crit = [names.get(l, str(l)) for l in options]
        q = {"type": "choice", "instructions": rng.choice(CLASSIFY), "criteria": crit}
        yield split, r["image"], {"state": PHOTO, "question": q, "label": options.index(r["label"])}


CONVERTERS = {"aokvqa": aokvqa, "scienceqa": scienceqa, "vqav2": vqav2, "imagenet": imagenet}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=CONVERTERS)
    ap.add_argument("src")
    ap.add_argument("--out", default="data/vqa")
    ap.add_argument("--label-names", help="json list or dict of class names (imagenet)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    img_dir = out / args.name
    img_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if args.label_names:
        names = json.loads(Path(args.label_names).read_text())
        kwargs["names"] = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    counts = defaultdict(int)
    with open(out / f"{args.name}.jsonl", "w") as f:
        for i, (split, image, row) in enumerate(CONVERTERS[args.name](args.src, rng, **kwargs)):
            if args.limit and i >= args.limit:
                break
            path = save_image(image, img_dir)
            if path is None:
                continue
            row.update(image=str(path.relative_to(out)), split=split)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[(split, row["question"]["type"])] += 1
    print(args.name, dict(counts))


if __name__ == "__main__":
    main()
