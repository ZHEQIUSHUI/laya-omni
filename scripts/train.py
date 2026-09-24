"""Train the fusion adapter on cached features. Laya and the feature encoders stay frozen.

Loss is cross-entropy over the option markers (the log score, a proper scoring
rule), so probabilities stay calibrated rather than just ranked. Every source is
evaluated with and without its features: the gap is what the adapter adds.

    python scripts/train.py --laya models/laya-multilingual --features cache/siglip2 \
        --games data/games:snake,bird --holdout-games bricks \
        --jsonl data/vqa/aokvqa.jsonl --out runs/v1
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from laya_omni.agent import Agent
from laya_omni.data import FeatureStore, GameSource, JsonlSource, collate, mix, seeded
from laya_omni.model import OmniFusion


def build_sources(args, store, split):
    sources = []
    game_dir, _, names = args.games.partition(":") if args.games else ("", "", "")
    names = [n for n in names.split(",") if n]
    holdout = [n for n in args.holdout_games.split(",") if n]
    for game in names + holdout:
        if f"game:{game}" not in store.arrays:
            store.add(f"game:{game}", Path(args.features) / game)
    for game in names:
        sources.append(GameSource(game_dir, game, split, store, f"game:{game}", augment=not args.no_augment))
    if split == "test":
        for game in holdout:  # never trained on: all of its frames are test frames
            sources += [GameSource(game_dir, game, s, store, f"game:{game}", augment=not args.no_augment) for s in ("train", "test")]
            sources[-2].name = sources[-1].name = f"holdout:{game}"
    for path in args.jsonl:
        # The rows' own keys say which modality a dataset carries.
        with open(path) as f:
            first = json.loads(f.readline())
        stores = {}
        for modality, root in (("image", args.features), ("audio", args.audio_features)):
            if first.get(modality):
                name = f"{modality}:{Path(path).stem}"
                if name not in store.arrays:
                    store.add(name, Path(root) / Path(path).stem)
                stores[modality] = name
        sources.append(JsonlSource(path, split, store, stores, shuffle_options=split == "train"))
    return [s for s in sources if (s.rows if hasattr(s, "rows") else True)]


@torch.no_grad()
def evaluate(agent, store, sources, device, batch_size=256, limit=4000):
    model = agent.model
    model.eval()
    rng = seeded(1234)
    stats = {}
    samples = []
    for src in sources:
        s = src.samples(rng)
        samples += s[:limit]
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        b, mods, y = collate(agent, store, chunk, ("image", "audio"), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            runs = {"with": model(**b, modalities=mods)[0], "without": model(**b)[0]}
        for name, lg in runs.items():
            p = lg.softmax(-1)
            for s, row, pi, yi in zip(chunk, lg, p, y):
                k = int((row > -1e3).sum())
                st = stats.setdefault(s["source"], {}).setdefault(name, [0, 0, 0.0, 0.0])
                st[0] += 1
                st[1] += int(pi.argmax() == yi)
                st[2] -= float(pi[yi].clamp(min=1e-9).log())
                st[3] += float(((pi[:k] - F.one_hot(yi, k)) ** 2).sum())
    return {
        src: {m: {"n": n, "acc": round(a / n, 4), "nll": round(l / n, 4), "brier": round(br / n, 4)} for m, (n, a, l, br) in v.items()}
        for src, v in sorted(stats.items())
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya", required=True)
    ap.add_argument("--features", default="", help="directory of cached image features, one prefix per dataset")
    ap.add_argument("--audio-features", default="", help="directory of cached audio features, one prefix per dataset")
    ap.add_argument("--games", default="", help="data_dir:game1,game2 (trained on)")
    ap.add_argument("--holdout-games", default="", help="games evaluated zero-shot, never trained on")
    ap.add_argument("--jsonl", action="append", default=[], help="converted dataset (repeatable)")
    ap.add_argument("--cap", type=int, default=0, help="max samples per source per epoch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-augment", action="store_true", help="games: one phrasing, one label set, fixed order")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--checkpointing", action="store_true", help="recompute encoder activations to save memory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = seeded(args.seed)
    agent = Agent(args.laya, device=args.device)
    model = agent.model
    store = FeatureStore()
    train, test = build_sources(args, store, "train"), build_sources(args, store, "test")
    in_dims = {}
    for name in store.arrays:  # "game:x" stores are images; others are "<modality>:<dataset>"
        in_dims.setdefault("image" if name.startswith("game:") else name.split(":", 1)[0], store.dims(name))
    model.fusion = OmniFusion(model.encoder.config.hidden_size, in_dims).to(args.device)
    if args.checkpointing:
        model.encoder.gradient_checkpointing_enable()
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("fusion."))
    params = [p for p in model.parameters() if p.requires_grad]
    print("train", {s.name: len(s.rows) for s in train}, "test", {s.name: len(s.rows) for s in test})
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.2f}M")

    per_epoch = len(mix(train, seeded(0), args.cap or None))
    steps = args.epochs * math.ceil(per_epoch / args.batch)
    warm = max(1, min(300, steps // 10))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    log = open(out / "log.jsonl", "a")
    base = evaluate(agent, store, test, args.device)
    print(json.dumps({"epoch": -1, **base}, ensure_ascii=False), flush=True)
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.eval()  # frozen parts keep dropout off; the fusion has no dropout
        epoch_samples = mix(train, rng, args.cap or None)
        for start in range(0, len(epoch_samples), args.batch):
            b, mods, y = collate(agent, store, epoch_samples[start : start + args.batch], ("image", "audio"), args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(**b, modalities=mods)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"ep {epoch} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        metrics = evaluate(agent, store, test, args.device)
        print(json.dumps({"epoch": epoch, **metrics}, ensure_ascii=False), flush=True)
        log.write(json.dumps({"epoch": epoch, "step": step, "metrics": metrics}) + "\n")
        log.flush()
        save_file({k: v.detach().cpu().contiguous() for k, v in model.fusion.state_dict().items()}, out / "fusion.safetensors")
        (out / "fusion_config.json").write_text(
            json.dumps({"in_dims": in_dims, "max_items": model.fusion.max_items, "max_frames": model.fusion.frame_emb.shape[0], "laya": str(args.laya)}, indent=2) + "\n"
        )
    print("saved", out)


if __name__ == "__main__":
    main()
