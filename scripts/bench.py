"""Standard benchmarks, laya-omni against small Qwen VLMs: accuracy and latency.

Every model answers the same rows (from scripts/convert_bench.py) one question at
a time, from the image / audio file to the answer, on the same GPU:

    python scripts/bench.py --model laya-omni --fusion runs/formal-v2 --laya models/laya-multilingual \\
        --image-encoder models/siglip2-base-patch16-256 --audio-encoder models/qwen3-asr-0.6b-audio-encoder \\
        --data data/bench/*.jsonl --out results/bench/laya-omni
    python scripts/bench.py --model qwen --qwen models/Qwen3.5-2B --data ... --out results/bench/qwen3.5-2b
    PYTHONPATH=<Valen repo> python scripts/bench.py --model valen --valen models/Valen-Preview-0923 \\
        --valen-base models/Qwen3.5-2B --data ... --out results/bench/valen

Qwen models see the options as lettered lines and the standard instruction
("Answer with the option's letter ..."); the answer is the letter (or Yes / No)
with the highest next-token logit, so they need one forward pass and no decoding.
Writes <out>/<set>.jsonl (one line per question) and <out>/summary.json.
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

LETTERS = "ABCDEFGHIJ"


def load_rows(path, limit):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    root = Path(path).parent
    for r in rows:
        for k in ("image", "audio"):
            if r.get(k):
                r[k] = str(root / r[k])
        if r.get("images"):
            r["images"] = [str(root / p) for p in r["images"]]
    return rows[:limit] if limit else rows


def images_of(row):
    return row.get("images") or ([row["image"]] if row.get("image") else [])


class LayaOmni:
    def __init__(self, args):
        from laya_omni import Omni

        self.omni = Omni.load(args.fusion, laya=args.laya, image_encoder=args.image_encoder or None,
                              audio_encoder=args.audio_encoder or None)
        self.text_only, self.detail = args.text_only, args.detail

    def answer(self, row):
        q = row["question"]
        kw = {}
        if not self.text_only:
            imgs = images_of(row)
            if imgs:
                kw["image"] = imgs if len(imgs) > 1 else imgs[0]
            if row.get("audio"):
                kw["audio"] = row["audio"]
        out = self.omni.predict(row["state"], {"q": q}, detail=self.detail, **kw)["answers"]["q"]
        if q["type"] == "noul":
            return int(out["noul"] >= 0.5)
        return q["criteria"].index(out["choice"])


class Qwen:
    SYSTEM_OMNI = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
                   "perceiving auditory and visual inputs, as well as generating text and speech.")

    def __init__(self, args):
        from transformers import AutoProcessor

        path = args.qwen
        cfg = json.loads((Path(path) / "config.json").read_text())
        self.omni = "omni" in cfg.get("model_type", "")
        if self.omni:
            from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

            self.processor = Qwen2_5OmniProcessor.from_pretrained(path)
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16)
        else:
            from transformers import AutoModelForImageTextToText

            self.processor = AutoProcessor.from_pretrained(path)
            self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16)
        self.model.to("cuda").eval()
        tok = self.processor.tokenizer
        first = lambda s: tok.encode(s, add_special_tokens=False)[0]
        self.letter_ids = [first(c) for c in LETTERS]
        self.yes_ids, self.no_ids = [first("Yes"), first("yes")], [first("No"), first("no")]
        self.text_only = args.text_only

    def prompt(self, row):
        q = row["question"]
        context = row.get("context", "")
        head = (context + "\n") if context else ""
        if q["type"] == "noul":
            return f"{head}{q['instructions']}\nAnswer the question using a single word: Yes or No."
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["criteria"]))
        return f"{head}{q['instructions']}\n{opts}\nAnswer with the option's letter from the given choices directly."

    @torch.inference_mode()
    def answer(self, row):
        from PIL import Image

        imgs = [] if self.text_only else [Image.open(p).convert("RGB") for p in images_of(row)]
        audio = None
        if row.get("audio") and not self.text_only:
            from laya_omni.encoders import load_audio

            audio = load_audio(row["audio"], 16000)
        content = [{"type": "image"} for _ in imgs] + ([{"type": "audio"}] if audio is not None else [])
        content.append({"type": "text", "text": self.prompt(row)})
        messages = [{"role": "user", "content": content}]
        kw = {}
        if self.omni:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_OMNI}]})
        else:
            kw["enable_thinking"] = False
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw)
        inputs = {"text": [text], "return_tensors": "pt"}
        if imgs:
            inputs["images"] = imgs
        if audio is not None:
            inputs["audio"] = [audio]
        batch = self.processor(**inputs).to("cuda")
        logits = self.model(**batch).logits[0, -1].float()
        q = row["question"]
        if q["type"] == "noul":
            return int(logits[self.yes_ids].max() > logits[self.no_ids].max())
        return int(logits[self.letter_ids[: len(q["criteria"])]].argmax())


class Valen:
    """Valen (github.com/Liuziyu77/Valen) through its own compiler and decision head.
    Run under Valen's pinned environment with its repository on PYTHONPATH."""

    def __init__(self, args):
        from valen.modeling.factory import build_compiler, build_model, normalize_model_config
        from valen.training.checkpoint import load_checkpoint

        ck = Path(args.valen)
        cfg = normalize_model_config(json.loads((ck / "config.json").read_text()))
        cfg.update(model_path=args.valen_base, device="cuda", gradient_checkpointing=False)
        self.model = build_model(cfg)
        load_checkpoint(str(ck), self.model)
        self.model.eval()
        self.compiler = build_compiler(cfg, "/")
        self.text_only = args.text_only

    @torch.inference_mode()
    def answer(self, row):
        from valen.data.schema import candidates, validate_record
        from valen.evaluation.inference import predict

        q = row["question"]
        content = [{"type": "text", "text": row["context"]}] if row.get("context") else []
        if not self.text_only:
            content += [{"type": "image_url", "image_url": {"url": p}} for p in images_of(row)]
        if not content:
            content = [{"type": "text", "text": row["state"]}]
        question = {"type": q["type"], "instructions": q["instructions"]}
        if q["type"] == "choice":
            question["criteria"] = {f"c{i}": o for i, o in enumerate(q["criteria"])}
        record = {"request": {"state": {"messages": [{"role": "user", "content": content}]}, "questions": {"q": question}},
                  "group_id": str(row.get("id", "x"))}
        validate_record(record, candidates)
        out = predict(self.model, self.compiler.compile(record))["answers"]["q"]
        if q["type"] == "noul":
            return int(out["noul"] >= 0.5)
        return int(out["choice"][1:])


def summarize(results):
    """Accuracy (and circular accuracy for rotated sets), per category, latency."""
    rot0 = [r for r in results if r.get("rot", 0) == 0]
    out = {"n": len(rot0), "acc": round(float(np.mean([r["correct"] for r in rot0])), 4)}
    groups = defaultdict(list)
    for r in results:
        if "group" in r:
            groups[r["group"]].append(r["correct"])
    if groups:
        out["circular_acc"] = round(float(np.mean([all(v) for v in groups.values()])), 4)
    cats = defaultdict(list)
    for r in rot0:
        cats[r.get("category") or "all"].append(r["correct"])
    if len(cats) > 1:
        out["by_category"] = {k: round(float(np.mean(v)), 4) for k, v in sorted(cats.items())}
    ms = [r["ms"] for r in results]
    out["latency_ms"] = {"mean": round(statistics.mean(ms), 1), "p50": round(statistics.median(ms), 1),
                         "p90": round(float(np.percentile(ms, 90)), 1)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["laya-omni", "qwen", "valen"], required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N rows of each set")
    ap.add_argument("--text-only", action="store_true", help="drop images and audio (what the wording alone gives)")
    ap.add_argument("--fusion")
    ap.add_argument("--laya")
    ap.add_argument("--image-encoder", default="")
    ap.add_argument("--audio-encoder", default="")
    ap.add_argument("--detail", action="store_true", help="laya-omni: 256 image tokens")
    ap.add_argument("--qwen")
    ap.add_argument("--valen", help="Valen checkpoint directory")
    ap.add_argument("--valen-base", help="its Qwen3.5 base model directory")
    args = ap.parse_args()

    model = {"laya-omni": LayaOmni, "qwen": Qwen, "valen": Valen}[args.model](args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    for path in args.data:
        name = Path(path).stem
        rows = load_rows(path, args.limit)
        if not rows:
            continue
        needs_audio = any(r.get("audio") for r in rows)
        if isinstance(model, LayaOmni) and needs_audio and model.omni.audio_encoder is None:
            print(f"skip {name}: no audio encoder")
            continue
        if (isinstance(model, Qwen) and needs_audio and not model.omni) or (isinstance(model, Valen) and needs_audio):
            print(f"skip {name}: {args.model} has no audio input")
            continue
        for r in rows[:3]:  # warm-up, not timed
            model.answer(r)
        results = []
        with open(out / f"{name}.jsonl", "w") as f:
            for i, r in enumerate(rows):
                torch.cuda.synchronize()
                t = time.perf_counter()
                pred = model.answer(r)
                torch.cuda.synchronize()
                res = {"id": r.get("id", i), "label": r["label"], "pred": pred, "correct": pred == r["label"],
                       "ms": (time.perf_counter() - t) * 1000}
                for k in ("category", "group", "rot"):
                    if k in r:
                        res[k] = r[k]
                results.append(res)
                f.write(json.dumps(res) + "\n")
                if (i + 1) % 500 == 0:
                    print(f"{name} {i + 1}/{len(rows)} acc {np.mean([x['correct'] for x in results]):.3f}", flush=True)
        summary[name] = summarize(results)
        print(name, json.dumps({k: v for k, v in summary[name].items() if k != "by_category"}), flush=True)
        summary["_peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
