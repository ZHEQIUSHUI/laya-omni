"""Adapt laya-omni to a new scenario with a few hundred to a few thousand examples.

One command: build the task data, train from a general fusion with early stopping,
calibrate, and write a fusion directory that Omni.load() takes.

Task data, any of:
  --folders DIR     one sub-folder per answer (images or audio clips):
                      DIR/normal/*.jpg  DIR/defect/*.jpg   ->  choice between "normal" and "defect"
  --csv FILE        columns: file (image or audio path), label; optional: question, state
  --jsonl FILE      laya-omni rows (see docs/finetune.md), with or without "split"

    python scripts/finetune.py --base runs/formal-v2 --laya models/laya-multilingual \\
        --image-encoder models/siglip2-base-patch16-256 \\
        --folders my_photos/ --question "Is the part defective?" --out runs/my-scenario

The question is asked as a choice among the answers (folder names or CSV labels);
with exactly two answers named yes/no (or true/false) it becomes a noul question.
--replay N mixes N questions per epoch from general data (--replay-jsonl) so the
adapter keeps its general ability.
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
YES_NO = [{"no", "yes"}, {"false", "true"}, {"否", "是"}]


def kind_of(path):
    ext = Path(path).suffix.lower()
    return "image" if ext in IMAGE_EXT else "audio" if ext in AUDIO_EXT else None


def examples_from_folders(root):
    out = []
    for sub in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for f in sorted(sub.rglob("*")):
            if f.is_file() and kind_of(f):
                out.append({"file": str(f.resolve()), "label": sub.name})
    return out


def examples_from_csv(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    base = Path(path).parent
    for r in rows:
        f = Path(r["file"])
        r["file"] = str((f if f.is_absolute() else base / f).resolve())
    return rows


def to_rows(examples, question, state, rng, root):
    """Examples -> laya-omni rows, media paths relative to `root` (the task jsonl's folder)."""
    labels = sorted({e["label"] for e in examples})
    if len(labels) < 2:
        raise SystemExit("need at least two different answers")
    noul = len(labels) == 2 and {l.lower() for l in labels} in YES_NO
    rows = []
    for e in examples:
        k = kind_of(e["file"])
        if k is None:
            continue
        q_text = e.get("question") or question
        if noul:
            q = {"type": "noul", "instructions": q_text}
            label = int(e["label"].lower() in ("yes", "true", "是"))
        else:
            q = {"type": "choice", "instructions": q_text, "criteria": labels}
            label = labels.index(e["label"])
        rows.append({"state": e.get("state") or state or ("Image." if k == "image" else "Audio clip."),
                     "question": q, "label": label, k: os.path.relpath(e["file"], root)})
    return rows, labels, noul


def split_rows(rows, val_fraction, rng):
    """Stratified train / validation split, unless the rows carry their own splits."""
    if all("split" in r for r in rows):
        return rows
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    for group in by_label.values():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction))
        for i, r in enumerate(group):
            r["split"] = "test" if i < n_val else "train"
    return rows


def run(cmd, env=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--folders")
    src.add_argument("--csv")
    src.add_argument("--jsonl")
    ap.add_argument("--question", default="What is this?", help="the question to ask about every example")
    ap.add_argument("--state", default="", help="state text (default: 'Image.' / 'Audio clip.')")
    ap.add_argument("--base", required=True, help="general fusion to start from (a run directory)")
    ap.add_argument("--laya", required=True)
    ap.add_argument("--image-encoder", default="")
    ap.add_argument("--audio-encoder", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora-lr", type=float, default=6e-5)
    ap.add_argument("--replay", type=int, default=0, help="general-data questions mixed in per epoch")
    ap.add_argument("--replay-jsonl", action="append", default=[], help="general datasets for --replay")
    ap.add_argument("--audio-features", default="", help="cached features of the replay audio sets")
    ap.add_argument("--device", default="0", help="CUDA device index")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    work = out / "task"
    work.mkdir(parents=True, exist_ok=True)

    if args.jsonl:
        rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
        root = Path(args.jsonl).resolve().parent
        for r in rows:  # media paths relative to the task jsonl written below
            for k in ("image", "audio"):
                if r.get(k):
                    r[k] = os.path.relpath((root / r[k]).resolve(), work.resolve())
        labels, noul = None, None
    else:
        examples = examples_from_folders(args.folders) if args.folders else examples_from_csv(args.csv)
        rows, labels, noul = to_rows(examples, args.question, args.state, rng, work.resolve())
    rows = split_rows(rows, args.val_fraction, rng)
    task = work / "task.jsonl"
    with open(task, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_train = sum(r["split"] == "train" for r in rows)
    print(f"{len(rows)} examples ({n_train} train, {len(rows) - n_train} validation)"
          + (f"; answers: {labels}" + (" (asked as yes/no)" if noul else "") if labels else ""))

    py, env = sys.executable, {"PYTHONPATH": str(Path(__file__).resolve().parents[1]), "CUDA_VISIBLE_DEVICES": args.device}
    feats = Path(args.audio_features) if args.audio_features else work / "features"
    if any(r.get("audio") for r in rows):
        if not args.audio_encoder:
            raise SystemExit("audio examples need --audio-encoder")
        feats_task = work / "features"
        run([py, "scripts/cache_features.py", "--modality", "audio", "--data", str(task), "--root", str(work),
             "--encoder", args.audio_encoder, "--out", str(feats_task / "task")], env)
        feats = f"{feats_task}:{args.audio_features}" if args.audio_features else feats_task

    n_val = len(rows) - n_train
    train = [py, "scripts/train.py", "--laya", args.laya, "--init", args.base, "--jsonl", str(task),
             "--epochs", str(args.epochs), "--batch", "16", "--lr", str(args.lr), "--lora-lr", str(args.lora_lr),
             "--warmup", "20", "--eval-limit", str(max(1000, n_val)), "--workers", "4", "--seed", str(args.seed),
             "--out", str(out)]
    if any(r.get("image") or r.get("images") for r in rows) or args.replay_jsonl:
        train += ["--raw-images", "--image-encoder", args.image_encoder, "--image-tokens", "64"]
    train += ["--audio-features", str(feats)]
    if args.replay:  # each epoch draws about `replay` general questions, split over the replay sets
        per_set = args.replay / max(1, len(args.replay_jsonl))
        for p in args.replay_jsonl:
            size = sum(1 for line in open(p, encoding="utf-8") if '"split": "test"' not in line)
            train += ["--jsonl", p, "--weight", f"{Path(p).stem}={min(1.0, per_set / max(1, size)):.6f}"]
    run(train, env)  # keeps the epoch with the best validation NLL (early stopping)

    evaluate = [py, "scripts/evaluate.py", "--laya", args.laya, "--fusion", str(out), "--jsonl", str(task),
                "--tokens", "64", "--limit", "100000", "--calibrate", "--workers", "4",
                "--audio-features", str(feats)] + (["--image-encoder", args.image_encoder] if args.image_encoder else [])
    run(evaluate, env)
    cfg = json.loads((out / "fusion_config.json").read_text())
    cfg.update({"task_answers": labels, "task_question": args.question, "base": str(args.base)})
    (out / "fusion_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    report = json.loads((out / "eval.json").read_text())
    for k, v in report.items():
        print(f"validation {k}: accuracy {v['with']['acc']:.3f} (without the image/audio {v['without']['acc']:.3f}), "
              f"NLL {v['with']['nll']:.3f} over {v['with']['n']} questions before calibration")
    print(f"done: Omni.load({str(out)!r}, laya=..., image_encoder=..., audio_encoder=...)")


if __name__ == "__main__":
    main()
