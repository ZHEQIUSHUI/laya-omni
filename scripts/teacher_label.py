"""Label questions with a large VLM teacher's option probabilities (for distillation).

The teacher runs behind OpenAI-compatible servers (vLLM); each question is asked the
way scripts/bench.py asks Qwen: lettered options, one generated token, and the
probability of each option letter (Yes / No for noul) from its top log-probabilities.

    python scripts/teacher_label.py --data data/cauldron/vqav2.jsonl --out data/teacher/vqav2.jsonl \\
        --server http://spark0:8333/v1 --server http://spark1:8333/v1 --workers 32

Writes each input row plus "teacher": {"probs": [...], "model": ...}; rows already in
--out are skipped, so an interrupted run resumes. --modality picks image rows (default),
audio rows (clips sent as 16 kHz WAV, for an omni teacher) or both.
"""

import argparse
import base64
import io
import itertools
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LETTERS = "ABCDEFGHIJ"


def prompt(row):
    q = row["question"]
    context = row.get("context", "")
    head = (context + "\n") if context else ""
    if q["type"] == "noul":
        return f"{head}{q['instructions']}\nAnswer the question using a single word: Yes or No."
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["criteria"]))
    return f"{head}{q['instructions']}\n{opts}\nAnswer with the option's letter from the given choices directly."


def image_url(path, max_side):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def option_probs(top, row):
    """Top log-probabilities of the first token -> a distribution over the row's options."""
    q = row["question"]
    keys = [["no", "No", "NO"], ["yes", "Yes", "YES"]] if q["type"] == "noul" else [[c] for c in LETTERS[: len(q["criteria"])]]
    mass = []
    for variants in keys:
        p = sum(math.exp(t["logprob"]) for t in top if t["token"].strip() in variants)
        mass.append(p)
    total = sum(mass)
    if total <= 0:
        return None, 0.0
    return [m / total for m in mass], total


def audio_url(path, max_seconds):
    """A clip as a 16 kHz mono WAV data URL (decoded with laya_omni's loader, so any format works)."""
    import soundfile as sf

    from laya_omni.audio import load_audio

    wav = load_audio(path, 16000)[: int(max_seconds * 16000)]
    buf = io.BytesIO()
    sf.write(buf, wav, 16000, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def label(client, model, row, root, max_side, max_seconds=30):
    imgs = row.get("images") or ([row["image"]] if row.get("image") else [])
    content = [{"type": "image_url", "image_url": {"url": image_url(root / p, max_side)}} for p in imgs]
    clips = row.get("audios") or ([row["audio"]] if row.get("audio") else [])
    content += [{"type": "audio_url", "audio_url": {"url": audio_url(root / p, max_seconds)}} for p in clips]
    content.append({"type": "text", "text": prompt(row)})
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}], max_tokens=1, temperature=0,
        logprobs=True, top_logprobs=20, extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    top = [{"token": t.token, "logprob": t.logprob} for t in r.choices[0].logprobs.content[0].top_logprobs]
    return option_probs(top, row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server", action="append", required=True, help="OpenAI-compatible base URL; repeat for several")
    ap.add_argument("--model", default="teacher", help="served model name")
    ap.add_argument("--workers", type=int, default=32, help="requests in flight (all servers together)")
    ap.add_argument("--split", default="train", help="only rows of this split ('' for all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1024, help="downscale larger images before sending")
    ap.add_argument("--modality", choices=["image", "audio", "any"], default="image",
                    help="rows to send: with an image and no audio, with audio (image optional), or any")
    args = ap.parse_args()

    from openai import OpenAI

    root = Path(args.data).parent
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    has_img = lambda r: bool(r.get("image") or r.get("images"))
    has_aud = lambda r: bool(r.get("audio") or r.get("audios"))
    keep = {"image": lambda r: has_img(r) and not has_aud(r), "audio": has_aud, "any": lambda r: has_img(r) or has_aud(r)}[args.modality]
    rows = [r for r in rows if keep(r) and (not args.split or r.get("split", "train") == args.split)]
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    key = lambda r: json.dumps([r.get("id"), r.get("image") or r.get("images"), r.get("audio"), r["question"]], sort_keys=True)
    done = {key(json.loads(line)) for line in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [r for r in rows if key(r) not in done]
    print(f"{len(rows)} rows, {len(done)} already labelled, {len(todo)} to go", flush=True)

    clients = itertools.cycle([OpenAI(base_url=s, api_key="none", timeout=300) for s in args.server])
    lock, n, correct, failed = threading.Lock(), [0], [0], [0]
    stream = open(out, "a", encoding="utf-8")

    def work(row):
        with lock:
            client = next(clients)
        try:
            probs, mass = label(client, args.model, row, root, args.max_side)
        except Exception as e:  # keep going; the row is retried on the next run
            with lock:
                failed[0] += 1
                if failed[0] <= 5:
                    print("failed:", repr(e)[:200], flush=True)
            return
        if probs is None:
            return
        rec = dict(row, teacher={"probs": [round(p, 5) for p in probs], "mass": round(mass, 4), "model": args.model})
        with lock:
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n[0] += 1
            correct[0] += int(max(range(len(probs)), key=probs.__getitem__) == row["label"])
            if n[0] % 500 == 0:
                stream.flush()
                print(f"{n[0]}/{len(todo)} teacher accuracy {correct[0] / n[0]:.3f}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, todo))
    stream.close()
    print(f"done: {n[0]} labelled, {failed[0]} failed, teacher accuracy {correct[0] / max(1, n[0]):.3f}")


if __name__ == "__main__":
    main()
