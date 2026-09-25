"""Evaluate a trained fusion through the public Agent, and fit calibration temperatures.

Loads Laya plus a fusion checkpoint the way users do (laya_omni.Agent), runs the
test rows of each dataset (every row for --holdout sets) at each image token count,
and reports accuracy, NLL and Brier with and without the image/audio.

--calibrate fits one temperature per (modality combination, question-type bucket)
on half of the rows (even positions) of the --jsonl sets and reports the other
half of every set, held-out ones included, before and after;
the temperatures go into the run's fusion_config.json, where Agent.predict applies
them. Temperatures stay within Laya's [0.5, 5] clamp.

    python scripts/evaluate.py --laya models/laya-multilingual --fusion runs/joint-v2 \\
        --image-encoder models/siglip2-base-patch16-256 --audio-features cache/qwen3-asr \\
        --jsonl data/cauldron/vqav2.jsonl --holdout data/cauldron/tqa.jsonl --calibrate
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from laya_omni.agent import Agent, TEMP_MAX, TEMP_MIN
from laya_omni.common import temp_bucket
from laya_omni.data import RAW, Batches, FeatureStore, JsonlSource, seeded, to_device
from laya_omni.encoders import ImageEncoder

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_rows(paths, holdout, store, audio_features, limit):
    sources = []
    for path, all_rows in [(p, False) for p in paths] + [(p, True) for p in holdout]:
        first = json.loads(open(path).readline())
        stores = {}
        if first.get("image") or first.get("images"):
            stores["image"] = RAW + str(Path(path).parent)
        if first.get("audio") or first.get("audios"):
            name = f"audio:{Path(path).stem}"
            if name not in store.arrays:
                store.add(name, Path(audio_features) / Path(path).stem)
            stores["audio"] = name
        splits = ("train", "test") if all_rows else ("test",)
        rows = [r for s in splits for r in JsonlSource(path, s, store, stores, shuffle_options=False).rows]
        sources.append((Path(path).stem, rows[:limit]))
    return sources


@torch.no_grad()
def logits_for(agent, store, samples, encoder, tokens, device, workers):
    """Raw option logits with and without the modality features, plus labels."""
    data = Batches(samples, 64, agent.tok, agent.max_len, agent.head_max_len, store)
    loader = torch.utils.data.DataLoader(data, batch_size=None, num_workers=workers)
    with_m, without, labels = [], [], []
    for cpu in loader:
        b, mods, y = to_device(*cpu, device, encoder, tokens)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lw = agent.model(**b, modalities=mods)[0].float()
            lo = agent.model(**b)[0].float()
        for row_w, row_o, yi in zip(lw, lo, y):
            k = int((row_w > -1e3).sum())
            with_m.append(row_w[:k].cpu().numpy())
            without.append(row_o[:k].cpu().numpy())
            labels.append(int(yi))
    return with_m, without, labels


def score(logits, labels, temps=None, keys=None):
    n = acc = nll = brier = 0.0
    for i, (z, y) in enumerate(zip(logits, labels)):
        t = temps.get(keys[i], 1.0) if temps else 1.0
        p = np.exp((z - z.max()) / t)
        p /= p.sum()
        n += 1
        acc += int(p.argmax() == y)
        nll -= math.log(max(p[y], 1e-9))
        brier += float(((p - np.eye(len(p))[y]) ** 2).sum())
    return {"n": int(n), "acc": round(acc / n, 4), "nll": round(nll / n, 4), "brier": round(brier / n, 4)}


def fit_temperature(logits, labels):
    """The temperature in [TEMP_MIN, TEMP_MAX] with the lowest NLL (grid, then refine)."""
    grid = np.exp(np.linspace(math.log(TEMP_MIN), math.log(TEMP_MAX), 61))

    def nll(t):
        return -sum(float(z[y] / t - np.log(np.exp((z - z.max()) / t).sum()) - z.max() / t) for z, y in zip(logits, labels))

    best = min(grid, key=nll)
    fine = best * np.exp(np.linspace(-0.05, 0.05, 11))
    return float(min(fine, key=nll))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya", required=True)
    ap.add_argument("--fusion", required=True, help="training run directory")
    ap.add_argument("--image-encoder", required=True)
    ap.add_argument("--audio-features", default="")
    ap.add_argument("--jsonl", action="append", default=[], help="dataset: its test rows are evaluated")
    ap.add_argument("--holdout", action="append", default=[], help="dataset never trained on: all rows")
    ap.add_argument("--tokens", default="64,256")
    ap.add_argument("--limit", type=int, default=4000, help="rows per dataset")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="", help="write the report here (default: <fusion>/eval.json)")
    args = ap.parse_args()

    agent = Agent(args.laya, args.fusion, device=args.device)
    encoder = ImageEncoder(args.image_encoder, device=args.device, pool=1)
    store = FeatureStore()
    report, calib = {}, defaultdict(lambda: ([], []))
    held_out = {Path(p).stem for p in args.holdout}
    for name, rows in load_rows(args.jsonl, args.holdout, store, args.audio_features, args.limit):
        samples = [dict(r) for r in rows]
        modality = "+".join(m for m in ("image", "audio") if any(m in s["features"] for s in samples)) or "text"
        counts = [int(t) for t in args.tokens.split(",")] if "image" in modality else [0]
        for tokens in counts:
            with_m, without, labels = logits_for(agent, store, samples, encoder, tokens or 64, args.device, args.workers)
            tag = f"{name}@{tokens}" if tokens else name
            keys = [f"{modality}:{temp_bucket(qt, len(z))}" for z, qt in
                    zip(with_m, [{"choice": 0, "score": 1, "noul": 2}[s["question"]["type"]] for s in samples])]
            report[tag] = {"modality": modality, "with": score(with_m, labels), "without": score(without, labels)}
            if args.calibrate and tokens in (0, counts[0]):  # calibrate at the default token count
                # Held-out sets are deliberately unlike the training data: they check the
                # temperatures but do not fit them.
                if name not in held_out:
                    for i, (z, y, k) in enumerate(zip(with_m, labels, keys)):
                        if i % 2 == 0:
                            calib[k][0].append(z)
                            calib[k][1].append(y)
                report[tag]["_logits"] = (with_m, labels, keys)
            print(tag, json.dumps({k: v for k, v in report[tag].items() if not k.startswith("_")}), flush=True)
    if args.calibrate:
        temps = {k: round(fit_temperature(z, y), 4) for k, (z, y) in calib.items() if len(y) >= 50}
        print("temperatures:", temps)
        for tag, r in report.items():
            if "_logits" in r:
                z, y, k = r.pop("_logits")
                odd = [i for i in range(len(y)) if i % 2 == 1]
                pick = lambda xs: [xs[i] for i in odd]
                r["held_half_before"] = score(pick(z), pick(y))
                r["held_half_after"] = score(pick(z), pick(y), temps, pick(k))
                print(tag, "calibration on held half:", r["held_half_before"], "->", r["held_half_after"])
        cfg_path = Path(args.fusion) / "fusion_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["temperature_by_modality"] = temps
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    out = Path(args.out or Path(args.fusion) / "eval.json")
    out.write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in report.items()}, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
