"""Standard benchmarks -> laya-omni rows, for scripts/bench.py.

    python scripts/convert_bench.py mmbench datasets/MMBench --out data/bench
    python scripts/convert_bench.py mmstar datasets/MMStar --out data/bench
    ...

Sets and splits (all with public answers, none used in training):
  mmbench   MMBench dev, English and Chinese (mmbench_en, mmbench_cn), with CircularEval
            rotations: a question counts only if every rotation of its options is right
  mmstar    MMStar (vision-indispensable questions)
  mme       MME perception + cognition (yes / no), with MME's acc+ pairs
  pope      POPE adversarial / popular / random (yes / no, object hallucination)
  seed      SEED-Bench image questions (dimensions 1-9)
  omnibench OmniBench (image + audio together)
  airbench  AIR-Bench foundation (audio), the downloaded tasks
Rows carry "category" for per-dimension scores; "context" is extra text (MMBench hints)
that goes into Laya's state and ahead of the question for chat models.
"""

import argparse
import glob
import hashlib
import io
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


def nan(x):
    return x is None or (isinstance(x, float) and x != x) or str(x).strip().lower() in ("", "nan", "none")


class Writer:
    def __init__(self, out, name):
        self.dir = Path(out) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.f = open(Path(out) / f"{name}.jsonl", "w", encoding="utf-8")
        self.name, self.n = name, 0

    def media(self, data, ext):
        """Write image / audio bytes once (content-addressed); return the path relative to the jsonl."""
        key = hashlib.sha1(data).hexdigest()[:16]
        path = self.dir / f"{key}{ext}"
        if not path.exists():
            path.write_bytes(data)
        return f"{self.name}/{path.name}"

    def image(self, cell):
        data = cell["bytes"] if isinstance(cell, dict) else cell
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        if img.format not in ("JPEG", "PNG"):
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "PNG")
            data = buf.getvalue()
            return self.media(data, ".png")
        return self.media(data, ".jpg" if img.format == "JPEG" else ".png")

    def write(self, row):
        row.setdefault("split", "test")
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.n += 1

    def close(self):
        self.f.close()
        print(f"{self.name}: {self.n} rows")


def state_for(modality, context=""):
    base = {"image": "Image.", "audio": "Audio clip.", "image+audio": "Image and audio clip."}[modality]
    return f"{base} {context}".strip() if context else base


def choice_rows(w, base, options, answer, rotate=False):
    """One row, or one per rotation of the options (CircularEval)."""
    n = len(options)
    for rot in range(n if rotate else 1):
        opts = options[rot:] + options[:rot]
        row = dict(base, question=dict(base["question"], criteria=opts), label=opts.index(options[answer]))
        if rotate:
            row.update(group=base["id"], rot=rot)
        w.write(row)


def mmbench(src, out):
    for lang in ("en", "cn"):
        w = Writer(out, f"mmbench_{lang}")
        for path in sorted(glob.glob(f"{src}/{lang}/dev-*.parquet")):
            for r in pq.read_table(path).to_pylist():
                options = [str(r[k]).strip() for k in "ABCD" if not nan(r.get(k))]
                if r["answer"] not in "ABCD" or len(options) < 2:
                    continue
                context = "" if nan(r.get("hint")) else str(r["hint"]).strip()
                base = {"id": f"{lang}-{r['index']}", "state": state_for("image", context), "context": context,
                        "question": {"type": "choice", "instructions": r["question"].strip()},
                        "image": w.image(r["image"]), "category": r.get("L2-category") or r["category"],
                        "source": r.get("source")}
                choice_rows(w, base, options, "ABCD".index(r["answer"]), rotate=True)
        w.close()


def mmstar(src, out):
    w = Writer(out, "mmstar")
    for r in pq.read_table(f"{src}/mmstar.parquet").to_pylist():
        text = r["question"]
        m = re.search(r"\n?Options:\s*(.*)$", text, re.S)
        if m:  # "Options: A: x, B: y"
            parts = re.split(r"(?:^|,\s*|\n)\s*([A-F])[:.]\s", m.group(1).strip())
        else:  # MathVista style: "Hint: ...\nQuestion: ...\nChoices:\n(A) x\n(B) y"
            m = re.search(r"\n?Choices:\s*(.*)$", text, re.S)
            if not m:
                continue
            parts = re.split(r"(?:^|\n)\s*\(([A-F])\)\s*", m.group(1).strip())
        options = [p.strip().rstrip(",") for p in parts[2::2]]
        letters = parts[1::2]
        if letters != list("ABCDEF"[: len(letters)]) or r["answer"] not in letters:
            continue
        stem = text[: m.start()].strip()
        stem = re.sub(r"^Hint: Please answer the question and provide the correct option letter.*?\n(Question:\s*)?", "", stem, flags=re.S).strip()
        base = {"id": r["index"], "state": "Image.", "question": {"type": "choice", "instructions": stem},
                "image": w.image(r["image"]), "category": r["category"]}
        choice_rows(w, base, options, letters.index(r["answer"]))
    w.close()


def mme(src, out):
    w = Writer(out, "mme")
    for path in sorted(glob.glob(f"{src}/data/*.parquet")):
        for r in pq.read_table(path).to_pylist():
            q = re.sub(r"\s*Please answer (yes or no|this question with one word)\.?\s*$", "", r["question"].strip(), flags=re.I)
            ans = r["answer"].strip().lower()
            if ans not in ("yes", "no"):
                continue
            img = w.image(r["image"])
            w.write({"id": r["question_id"], "state": "Image.", "question": {"type": "noul", "instructions": q},
                     "label": int(ans == "yes"), "image": img, "category": r["category"],
                     "group": f"{r['category']}/{img}"})  # acc+: both questions about an image right
    w.close()


def pope(src, out):
    w = Writer(out, "pope")
    for path in sorted(glob.glob(f"{src}/data/*.parquet")):
        for r in pq.read_table(path).to_pylist():
            ans = r["answer"].strip().lower()
            w.write({"id": f"{r['category']}-{r['question_id']}", "state": "Image.",
                     "question": {"type": "noul", "instructions": r["question"].strip()},
                     "label": int(ans == "yes"), "image": w.image(r["image"]), "category": r["category"]})
    w.close()


SEED_DIMS = {1: "scene understanding", 2: "instance identity", 3: "instance attributes", 4: "instance location",
             5: "instance counting", 6: "spatial relation", 7: "instance interaction", 8: "visual reasoning",
             9: "text recognition"}


def seed(src, out):
    w = Writer(out, "seed")
    for path in sorted(glob.glob(f"{src}/data/*.parquet")):
        t = pq.read_table(path)
        for r in t.to_pylist():
            if r.get("data_type") != "image" or r["answer"] not in "ABCD":
                continue
            img = r["image"][0] if isinstance(r["image"], list) else r["image"]
            options = [str(r[f"choice_{c}"]).strip() for c in "abcd"]
            base = {"id": r["question_id"], "state": "Image.", "question": {"type": "choice", "instructions": r["question"].strip()},
                    "image": w.image(img), "category": SEED_DIMS.get(int(r["question_type_id"]), str(r["question_type_id"]))}
            choice_rows(w, base, options, "ABCD".index(r["answer"]))
    w.close()


def omnibench(src, out):
    w = Writer(out, "omnibench")
    for path in sorted(glob.glob(f"{src}/data/*.parquet")):
        for r in pq.read_table(path).to_pylist():
            options = [str(o).strip() for o in r["options"]]
            if r["answer"].strip() not in options:
                continue
            audio = r["audio"]["bytes"] if isinstance(r["audio"], dict) else r["audio"]
            base = {"id": r["index"], "state": "Image and audio clip.",
                    "question": {"type": "choice", "instructions": r["question"].strip()},
                    "image": w.image(r["image"]), "audio": w.media(audio, Path((r["audio"] or {}).get("path") or "a.wav").suffix or ".wav"),
                    "category": r.get("task type") or r.get("task_type")}
            choice_rows(w, base, options, options.index(r["answer"].strip()))
    w.close()


def airbench(src, out):
    w = Writer(out, "airbench")
    meta = json.load(open(f"{src}/Foundation/Foundation_meta.json"))
    for r in meta:
        task = f"{r['task_name']}_{r['dataset_name']}"
        audio = Path(src) / "Foundation" / task / r["path"]
        if not audio.exists():
            continue
        options = [str(r[k]).strip() for k in ("choice_a", "choice_b", "choice_c", "choice_d") if not nan(r.get(k))]
        answer = str(r["answer_gt"]).strip()
        if answer not in options:
            continue
        rel = w.media(audio.read_bytes(), audio.suffix)
        base = {"id": r["uniq_id"], "state": "Audio clip.", "question": {"type": "choice", "instructions": r["question"].strip()},
                "audio": rel, "category": task}
        choice_rows(w, base, options, options.index(answer))
    w.close()


def valen(src, out):
    """Valen-Eval-General-5k (Valen's own held-out set; its images come from the training
    splits of VQAv2, GQA, TextVQA, ChartQA, DocVQA, CLEVR, RICO-ScreenQA, ShowUI, GameQA).
    The 18 score questions are asked as choices over their levels."""
    w = Writer(out, "valen_general")
    for line in open(Path(src) / "train.jsonl", encoding="utf-8"):
        r = json.loads(line)
        (q,) = r["request"]["questions"].values()
        (target,) = r["targets"].values()
        probs = target["probabilities"]
        content = [c for m in r["request"]["state"]["messages"] for c in m["content"]]
        context = " ".join(c["text"] for c in content if c["type"] == "text").strip()
        (image,) = [c["image_url"]["url"] for c in content if c["type"] == "image_url"]
        if q["type"] == "noul":
            question, label = {"type": "noul", "instructions": q["instructions"]}, int(probs["true"] == 1)
        else:
            keys = list(q["criteria"]) if isinstance(q["criteria"], dict) else [str(i) for i in range(len(q["criteria"]))]
            options = list(q["criteria"].values()) if isinstance(q["criteria"], dict) else list(q["criteria"])
            question, label = {"type": "choice", "instructions": q["instructions"], "criteria": options}, keys.index(max(probs, key=probs.get))
        w.write({"id": r["meta"]["record_id"], "state": state_for("image", context), "context": context,
                 "question": question, "label": label,
                 "image": w.media((Path(src) / image).read_bytes(), Path(image).suffix), "category": r["meta"]["source"]})
    w.close()


SETS = {"valen": valen, "mmbench": mmbench, "mmstar": mmstar, "mme": mme, "pope": pope, "seed": seed, "omnibench": omnibench, "airbench": airbench}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", choices=sorted(SETS))
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    SETS[args.set](args.src, args.out)


if __name__ == "__main__":
    main()
