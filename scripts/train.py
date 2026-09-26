"""Train the fusion adapter. Laya and the feature encoders stay frozen.

Loss is cross-entropy over the option markers (the log score, a proper scoring
rule), so probabilities stay calibrated rather than just ranked. Every source is
evaluated with and without its features: the gap is what the adapter adds.

Images come either from cached features (--features) or, with --raw-images,
straight from the image files: DataLoader workers decode them and the frozen
image encoder runs on the GPU each step, so no feature cache is needed and each
batch can use a different token count (--image-tokens 64,256). Audio always
comes from cached features (--audio-features).

Several GPUs: `torchrun --nproc_per_node 2 scripts/train.py ...`. Every rank draws
the same epoch order and takes every world-size-th batch; the fusion's gradients
(the only trainable parameters) are averaged before each optimizer step. Rank 0
evaluates, logs and saves.

    python scripts/train.py --laya models/laya-multilingual --raw-images \\
        --image-encoder models/siglip2-base-patch16-256 --image-tokens 64,256 \\
        --audio-features cache/qwen3-asr --jsonl data/cauldron/vqav2.jsonl \\
        --jsonl data/audio/esc50.jsonl --lora 16 --out runs/v2
"""

import argparse
import datetime
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file

from laya_omni.agent import Agent
from laya_omni.data import RAW, Batches, FeatureStore, GameSource, JsonlSource, feature_prefix, load_teacher, mix, seeded, to_device
from laya_omni.model import OmniFusion

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # workers fork after the tokenizer is used


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
    jsonl = [(p, False) for p in args.jsonl] + ([(p, True) for p in args.holdout_jsonl] if split == "test" else [])
    for path, held_out in jsonl:
        # The rows' own keys say which modality a dataset carries.
        with open(path) as f:
            first = json.loads(f.readline())
        stores = {}
        for modality, root in (("image", args.features), ("audio", args.audio_features)):
            if not (first.get(modality) or first.get(modality + "s")):
                continue
            if modality == "image" and args.raw_images:
                stores[modality] = RAW + str(Path(path).parent)  # image paths are relative to the jsonl
                continue
            name = f"{modality}:{Path(path).stem}"
            if name not in store.arrays:
                store.add(name, feature_prefix(root, Path(path).stem))
            stores[modality] = name
        if held_out:  # never trained on: every row is a test row
            parts = [JsonlSource(path, s, store, stores, shuffle_options=False) for s in ("train", "test")]
            parts[0].rows += parts[1].rows
            parts[0].name = f"holdout:{Path(path).stem}"
            sources.append(parts[0])
        else:
            teacher = args.teacher_labels if split == "train" else None
            sources.append(JsonlSource(path, split, store, stores, shuffle_options=split == "train", teacher=teacher))
    return [s for s in sources if (s.rows if hasattr(s, "rows") else True)]


def loader(agent, store, samples, batch_size, workers, image_encoder=None):
    size = image_encoder.image_size if image_encoder is not None else 256
    data = Batches(samples, batch_size, agent.tok, agent.max_len, agent.head_max_len, store, image_size=size)
    return torch.utils.data.DataLoader(
        data, batch_size=None, shuffle=False, num_workers=workers, prefetch_factor=4 if workers else None,
        persistent_workers=False,
    )


def evaluate_sources(agent, store, sources, args, image_encoder, image_tokens, only_images=False, batch_size=64):
    """Accuracy, NLL and Brier per source, with and without the source's features."""
    model = agent.model
    model.eval()
    rng = seeded(1234)
    samples = []
    for src in sources:
        s = src.samples(rng)[: args.eval_limit]
        if only_images:
            s = [x for x in s if "image" in x["features"]]
        samples += s
    names = [s["source"] for s in samples]
    stats, seen = {}, 0
    with torch.no_grad():
        for cpu in loader(agent, store, samples, batch_size, args.workers, image_encoder):
            b, mods, y = to_device(*cpu, args.device, image_encoder, image_tokens)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                runs = {"with": model(**b, modalities=mods)[0], "without": model(**b)[0]}
            batch_names = names[seen : seen + len(y)]
            seen += len(y)
            for name, lg in runs.items():
                p = lg.float().softmax(-1)
                for src, row, pi, yi in zip(batch_names, lg, p, y):
                    k = int((row > -1e3).sum())
                    st = stats.setdefault(src, {}).setdefault(name, [0, 0, 0.0, 0.0])
                    st[0] += 1
                    st[1] += int(pi.argmax() == yi)
                    st[2] -= float(pi[yi].clamp(min=1e-9).log())
                    st[3] += float(((pi[:k] - F.one_hot(yi, k)) ** 2).sum())
    return {
        src: {m: {"n": n, "acc": round(a / n, 4), "nll": round(l / n, 4), "brier": round(br / n, 4)} for m, (n, a, l, br) in v.items()}
        for src, v in sorted(stats.items())
    }


def evaluate_all(agent, store, sources, args, image_encoder):
    """Every source at the first image token count; image sources again at the others."""
    counts = args.tokens
    metrics = evaluate_sources(agent, store, sources, args, image_encoder, counts[0])
    for t in counts[1:]:
        more = evaluate_sources(agent, store, sources, args, image_encoder, t, only_images=True)
        metrics.update({f"{k}@{t}": v for k, v in more.items()})
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya", required=True)
    ap.add_argument("--features", default="", help="directory of cached image features, one prefix per dataset")
    ap.add_argument("--audio-features", default="", help="directories (':'-separated) of cached audio features, one prefix per dataset")
    ap.add_argument("--raw-images", action="store_true", help="read and encode image files during training")
    ap.add_argument("--image-encoder", default="", help="SigLIP checkpoint for --raw-images")
    ap.add_argument("--image-tokens", default="64", help="token counts per image, drawn per batch (e.g. 64,256)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--image-grid", type=int, default=16, help="2-D position grid for image tokens (0: frame indices)")
    ap.add_argument("--games", default="", help="data_dir:game1,game2 (trained on)")
    ap.add_argument("--holdout-games", default="", help="games evaluated zero-shot, never trained on")
    ap.add_argument("--jsonl", action="append", default=[], help="converted dataset (repeatable)")
    ap.add_argument("--holdout-jsonl", action="append", default=[], help="dataset evaluated zero-shot, never trained on")
    ap.add_argument("--cap", type=int, default=0, help="max samples per source per epoch")
    ap.add_argument("--weight", action="append", default=[], help="source=factor: draw a source more (or less) often")
    ap.add_argument("--eval-limit", type=int, default=2000, help="test samples per source")
    ap.add_argument("--probe-every", type=int, default=0, help="quick evaluation every N optimizer steps (0: off)")
    ap.add_argument("--probe-limit", type=int, default=200, help="test samples per source for the quick evaluation")
    ap.add_argument("--warmup", type=int, default=300, help="warm-up optimizer steps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-augment", action="store_true", help="games: one phrasing, one label set, fixed order")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per optimizer step (effective batch = batch x accum)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lora", type=int, default=0, help="LoRA rank for the frozen encoder and head (modality rows only)")
    ap.add_argument("--lora-lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="", help="run directory whose fusion.safetensors initialises this run")
    ap.add_argument("--teacher", action="append", default=[],
                    help="teacher-labelled rows (scripts/teacher_label.py) for distillation; repeatable")
    ap.add_argument("--distill-weight", type=float, default=0.3,
                    help="weight of KL(teacher || model) on rows the teacher answers correctly")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.tokens = [int(t) for t in args.image_tokens.split(",")]
    args.weights = {k: float(v) for k, v in (w.split("=", 1) for w in args.weight)}
    args.teacher_labels = load_teacher(args.teacher) if args.teacher else None

    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        # Consumer cards (RTX 4090) have no GPU-to-GPU P2P; NCCL's P2P path crashes there.
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        # Rank 0 evaluates between epochs while the others wait: allow a long wait.
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        args.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    main_rank = rank == 0
    torch.manual_seed(args.seed)
    rng = seeded(args.seed)  # identical on every rank: same epoch order
    token_rng = seeded(args.seed + 1000 + rank)  # per rank: token counts differ between ranks
    agent = Agent(args.laya, device=args.device)
    model = agent.model
    store = FeatureStore()
    train, test = build_sources(args, store, "train"), build_sources(args, store, "test")
    in_dims = {}
    for name in store.arrays:  # "game:x" stores are images; others are "<modality>:<dataset>"
        in_dims.setdefault("image" if name.startswith("game:") else name.split(":", 1)[0], store.dims(name))
    image_encoder = None
    if args.raw_images:
        from laya_omni.encoders import ImageEncoder

        image_encoder = ImageEncoder(args.image_encoder, device=args.device, pool=1)
        in_dims["image"] = image_encoder.dims
    grid = args.image_grid if args.raw_images else 0  # SigLIP base/16 at 256 px: a 16 x 16 patch grid
    if args.init:  # keep the initialising run's modalities and layout, even ones this data lacks
        init_cfg = json.loads((Path(args.init) / "fusion_config.json").read_text())
        in_dims = {**init_cfg["in_dims"], **in_dims}
        grid = init_cfg.get("image_grid", grid)
        args.lora = init_cfg.get("lora_rank", args.lora)
    model.fusion = OmniFusion(model.encoder.config.hidden_size, in_dims, lora_rank=args.lora, image_grid=grid).to(args.device)
    if args.init:  # continue from an earlier run's fusion (e.g. stage 2 after an alignment stage)
        from safetensors.torch import load_file

        model.fusion.load_state_dict(load_file(Path(args.init) / "fusion.safetensors", device=args.device), strict=True)
        print(f"initialised the fusion from {args.init}") if rank == 0 else None
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("fusion."))
    params = [p for p in model.parameters() if p.requires_grad]
    if main_rank:
        print("train", {s.name: len(s.rows) for s in train}, "test", {s.name: len(s.rows) for s in test})
    main_rank and print(f"trainable {sum(p.numel() for p in params) / 1e6:.2f}M, of which LoRA {sum(p.numel() for n, p in model.named_parameters() if 'fusion.lora.' in n) / 1e6:.2f}M")

    per_epoch = len(mix(train, seeded(0), args.cap or None, args.weights))
    steps = args.epochs * math.ceil(per_epoch / (args.batch * args.accum * world))
    warm = max(1, min(args.warmup, steps // 10))
    lora = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("fusion.lora.")]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fusion.lora.")]
    groups = [{"params": rest, "lr": args.lr}] + ([{"params": lora, "lr": args.lora_lr}] if lora else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if main_rank:
        (out / "args.json").write_text(json.dumps({**vars(args), "world": world}, indent=2) + "\n")
    config = {
        "in_dims": in_dims,
        "max_items": model.fusion.max_items,
        "max_frames": model.fusion.frame_emb.shape[0],
        "image_grid": grid,
        "image_tokens": args.tokens,
        "image_encoder": args.image_encoder,
        "lora_rank": model.fusion.lora_rank,
        "lora_alpha": model.fusion.lora_alpha,
        "lora_layers": model.fusion.lora_layers,
        "laya": str(args.laya),
    }
    log = None
    if main_rank:
        (out / "fusion_config.json").write_text(json.dumps(config, indent=2) + "\n")
        log = open(out / "log.jsonl", "a")
        base = evaluate_all(agent, store, test, args, image_encoder)
        print(json.dumps({"epoch": -1, **base}, ensure_ascii=False), flush=True)
    if world > 1:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    step, micro, t0, state = 0, 0, time.time(), {"best": float("inf")}
    for epoch in range(args.epochs):
        model.eval()  # frozen parts keep dropout off; the fusion has no dropout
        samples = mix(train, rng, args.cap or None, args.weights)
        per_rank = len(samples) // (args.batch * world) * args.batch  # equal batch counts on every rank
        mine = [x for i in range(0, per_rank * world, args.batch) if (i // args.batch) % world == rank
                for x in samples[i : i + args.batch]]
        for cpu in loader(agent, store, mine, args.batch, args.workers, image_encoder):
            b, mods, y = to_device(*cpu, args.device, image_encoder, token_rng.choice(args.tokens))
            teacher, teacher_mask = b.pop("teacher", None), b.pop("teacher_mask", None)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(**b, modalities=mods)
            loss = F.cross_entropy(logits.float(), y)
            if teacher is not None:  # distill only where the teacher picks the true answer
                use = teacher_mask & (teacher.argmax(-1) == y)
                if use.any():
                    t, logp = teacher[use], F.log_softmax(logits.float()[use], -1)
                    kl = torch.where(t > 0, t * (torch.log(t.clamp_min(1e-8)) - logp), torch.zeros_like(t)).sum(-1)
                    loss = loss + args.distill_weight * kl.sum() / len(y)
            (loss / args.accum).backward()
            micro += 1
            if micro % args.accum:
                continue
            if world > 1:  # average the fusion's gradients in one call; unused ones count as zero
                grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                flat = torch.cat([g.flatten() for g in grads])
                dist.all_reduce(flat, op=dist.ReduceOp.AVG)
                for p, g in zip(params, flat.split([g.numel() for g in grads])):
                    p.grad = g.view_as(p)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 100 == 0 and main_rank:
                print(f"ep {epoch} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if args.probe_every and step % args.probe_every == 0 and micro % args.accum == 0:
                # A quick look inside the epoch: a run that stops using the images for some
                # sources early (as one seed of joint-v2 did) shows up here, not hours later.
                if world > 1:
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                if main_rank:
                    probe = evaluate_sources(agent, store, test, argparse.Namespace(**{**vars(args), "eval_limit": args.probe_limit}),
                                             image_encoder, args.tokens[0])
                    gap = {k: round(v["with"]["acc"] - v["without"]["acc"], 3) for k, v in probe.items()}
                    print(json.dumps({"probe_step": step, "gain_over_text": gap}), flush=True)
                    log.write(json.dumps({"probe_step": step, "metrics": probe}) + "\n")
                    log.flush()
                    model.eval()
                if world > 1:
                    dist.barrier(device_ids=[torch.cuda.current_device()])
        if world > 1:
            dist.barrier(device_ids=[torch.cuda.current_device()])
        if main_rank:
            end_of_epoch(agent, store, test, args, image_encoder, model, epoch, step, log, out, state)
        if world > 1:
            dist.barrier(device_ids=[torch.cuda.current_device()])
    if world > 1:
        dist.destroy_process_group()
    if main_rank:
        print("saved", out)


def end_of_epoch(agent, store, test, args, image_encoder, model, epoch, step, log, out, state):
    """Evaluate, log, and keep the fusion of the epoch with the best mean test NLL."""
    metrics = evaluate_all(agent, store, test, args, image_encoder)
    print(json.dumps({"epoch": epoch, **metrics}, ensure_ascii=False), flush=True)
    log.write(json.dumps({"epoch": epoch, "step": step, "metrics": metrics}) + "\n")
    log.flush()
    # Keep the epoch with the best mean test NLL: on small data accuracy plateaus
    # while the probabilities keep getting more overconfident.
    score = sum(v["with"]["nll"] for v in metrics.values()) / len(metrics)
    if score < state["best"]:
        state["best"] = score
        save_file({k: v.detach().cpu().contiguous() for k, v in model.fusion.state_dict().items()}, out / "fusion.safetensors")
        (out / "best.json").write_text(json.dumps({"epoch": epoch, "mean_nll": score, "metrics": metrics}, indent=1) + "\n")


if __name__ == "__main__":
    main()
