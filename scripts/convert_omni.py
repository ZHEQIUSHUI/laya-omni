"""Convert m-a-p/OmniInstruct into laya-omni rows that carry an image and an audio clip.

Each OmniInstruct row is a video frame, the video's audio, a question and a short
answer. The answer becomes a choice among 2-4 options; the wrong ones are answers
to the same kind of question elsewhere in the dataset (the first four, three or two
words of the question). A twentieth of the videos, by id, are the test split.

    python scripts/convert_omni.py datasets/OmniInstruct --out data/omni
"""

import argparse
import glob
import hashlib
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from convert_audio import SR, decode  # noqa: E402
from convert_cauldron import question_kinds  # noqa: E402

STATE = "A video frame and its audio."
NAME = "omniinstruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="data/omni")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    src, out = Path(args.src), Path(args.out)
    (out / NAME).mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(src / "data" / "*.parquet")))

    answers = defaultdict(set)
    for path in files:
        for r in pq.read_table(path, columns=["question", "answer"]).to_pylist():
            for kind in question_kinds(r["question"]):
                answers[kind].add(r["answer"].strip())

    counts, saved = Counter(), {}

    def store(cell, kind):
        data = cell.get("bytes") if isinstance(cell, dict) else None
        if not data:
            return None
        digest = hashlib.md5(data).hexdigest()[:16]
        key = (kind, digest)
        if key not in saved:
            if kind == "image":
                dest = out / NAME / f"{digest}.jpg"
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img.thumbnail((512, 512))
                img.save(dest, quality=90)
            else:
                dest = out / NAME / f"{digest}.wav"
                sf.write(dest, decode(cell), SR)
            saved[key] = str(dest.relative_to(out))
        return saved[key]

    with open(out / f"{NAME}.jsonl", "w") as f:
        for path in files:
            for r in pq.read_table(path).to_pylist():
                answer, question = r["answer"].strip(), r["question"].strip()
                pool = []
                for kind in question_kinds(question):
                    pool = sorted(answers.get(kind, set()) - {answer})
                    if len(pool) >= 3:
                        break
                if len(pool) < 3 or not answer:
                    counts["skipped"] += 1
                    continue
                try:
                    image, audio = store(r["image"], "image"), store(r["audio"], "audio")
                except Exception:
                    counts["undecodable"] += 1
                    continue
                if not image or not audio:
                    counts["missing media"] += 1
                    continue
                options = rng.sample(pool, rng.randint(1, 3)) + [answer]
                rng.shuffle(options)
                split = "test" if r["video_id"] % 20 == 0 else "train"
                row = {
                    "state": STATE,
                    "question": {"type": "choice", "instructions": question, "criteria": options},
                    "label": options.index(answer),
                    "image": image,
                    "audio": audio,
                    "split": split,
                    "category": r.get("category"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[split] += 1
    print(NAME, dict(counts), len(saved), "media files")


if __name__ == "__main__":
    main()
