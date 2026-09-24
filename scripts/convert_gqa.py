"""Convert GQA (lmms-lab/GQA) into laya-omni questions.

GQA asks compositional questions about scene graphs: relations ("what is on the
white wall?"), attributes, comparisons, spatial logic. The balanced train split
trains and the balanced testdev split tests.

- yes / no answers                  -> noul
- "choose" questions ("Is the sky blue or gray?") -> choice between the two named
- other answers                     -> choice; wrong options are answers to other
  questions of the same detailed GQA type (relation queries get objects, colour
  queries get colours), so they are plausible

    python scripts/convert_gqa.py datasets/GQA --out data/gqa
"""

import argparse
import glob
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

STATE = "Image."
SPLITS = {"train": "train_balanced", "test": "testdev_balanced"}


def save_images(src, name, out_dir):
    saved = {}
    for path in sorted(glob.glob(str(Path(src) / f"{name}_images" / "*.parquet"))):
        for r in pq.read_table(path).to_pylist():
            dest = out_dir / f"{r['id']}.jpg"
            if not dest.exists():
                data = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img.thumbnail((512, 512))
                img.save(dest, quality=90)
            saved[r["id"]] = f"images/{dest.name}"
    return saved


MODIFIERS = {"light", "dark", "bright", "pale", "deep"}


def clean_option(text):
    """'the woman' -> 'woman'; 'to the right of the cup' -> 'right'."""
    text = re.sub(r"^(to the|to|the|a|an)\s+", "", text.strip())
    return re.split(r"\s+(?:of|than|in|on|at|from)\s+", text)[0].strip()


def alternatives(question, answer):
    """'Is the sky blue or gray?' -> ['blue', 'gray'] when the answer is one of the two."""
    m = re.search(r"(\w[\w -]*?) or (\w[\w -]*?)\?$", question)
    if not m:
        return None
    words = m.group(1).split()
    left = words[-2:] if len(words) > 1 and words[-2] in MODIFIERS else words[-1:]
    left, right = clean_option(" ".join(left)), clean_option(m.group(2))
    if answer in (left, right) and left != right and left and right:
        return [left, right]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="data/gqa")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with open(out / "gqa.jsonl", "w") as f:
        for split, name in SPLITS.items():
            images = save_images(args.src, name, out / "images")
            cols = ["imageId", "question", "answer", "types"]
            rows = [r for p in sorted(glob.glob(str(Path(args.src) / f"{name}_instructions" / "*.parquet")))
                    for r in pq.read_table(p, columns=cols).to_pylist()]
            pool = defaultdict(set)
            for r in rows:
                pool[(r["types"] or {}).get("detailed")].add(r["answer"])
            for r in rows:
                image = images.get(r["imageId"])
                answer, question = r["answer"].strip(), r["question"].strip()
                if image is None or not answer:
                    counts["skipped"] += 1
                    continue
                if answer in ("yes", "no"):
                    q, y = {"type": "noul", "instructions": question}, int(answer == "yes")
                else:
                    options = alternatives(question, answer)
                    if options is None:
                        others = sorted(pool[(r["types"] or {}).get("detailed")] - {answer})
                        if len(others) < 3:
                            counts["skipped"] += 1
                            continue
                        options = rng.sample(others, rng.randint(1, 3)) + [answer]
                    rng.shuffle(options)
                    q, y = {"type": "choice", "instructions": question, "criteria": options}, options.index(answer)
                row = {"state": STATE, "question": q, "label": y, "image": image, "split": split,
                       "gqa_type": (r["types"] or {}).get("detailed")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[f"{split}:{q['type']}"] += 1
            print(split, len(images), "images", flush=True)
    print("gqa", dict(counts))


if __name__ == "__main__":
    main()
