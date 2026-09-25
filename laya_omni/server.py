"""Web demo: ask Laya typed questions about an uploaded image and/or audio clip.

    laya-omni serve --fusion runs/formal-v2 --laya models/laya-multilingual \\
        --image-encoder models/siglip2-base-patch16-256 \\
        --audio-encoder models/qwen3-asr-0.6b-audio-encoder --examples data --port 8030
"""

import io
import json
import random
import time
from pathlib import Path

import numpy as np

WEB = Path(__file__).parent / "web"
# Example sets: (jsonl relative to the examples root, label shown in the gallery).
EXAMPLE_SETS = [
    ("cauldron/vqav2.jsonl", "照片问答"),
    ("cauldron/textvqa.jsonl", "读字"),
    ("cauldron/chartqa.jsonl", "图表"),
    ("gqa/gqa.jsonl", "组合推理"),
    ("eval/imagenet.jsonl", "物体分类（零样本）"),
    ("audio/esc50.jsonl", "环境声"),
    ("audio/cremad.jsonl", "语音情绪"),
    ("audio/slurp.jsonl", "语音指令"),
    ("audio/minds14.jsonl", "客服意图（多语言）"),
    ("audio/musicbench.jsonl", "音乐"),
    ("omni/omniinstruct.jsonl", "画面 + 声音"),
]


def load_examples(root, per_set=6, seed=0):
    rng = random.Random(seed)
    out = []
    for rel, label in EXAMPLE_SETS:
        path = Path(root) / rel
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if r.get("split") == "test" and (r.get("image") or r.get("audio"))]
        for r in rng.sample(rows, min(per_set, len(rows))):
            base = str(Path(rel).parent)
            out.append({
                "set": label,
                "state": r.get("state", ""),
                "question": r["question"],
                "answer": r["label"],
                "image": f"{base}/{r['image']}" if r.get("image") else None,
                "audio": f"{base}/{r['audio']}" if r.get("audio") else None,
            })
    return out


def create_app(omni, examples_root=None):
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse
    from PIL import Image

    app = FastAPI(title="laya-omni")
    examples = load_examples(examples_root) if examples_root else []
    root = Path(examples_root).resolve() if examples_root else None

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB / "index.html").read_text()

    @app.get("/api/info")
    def info():
        return {
            "image": omni.image_encoder is not None,
            "audio": omni.audio_encoder is not None,
            "device": str(omni.agent.device),
            "examples": len(examples),
        }

    @app.get("/api/examples")
    def get_examples():
        return examples

    @app.get("/api/media")
    def media(path: str):
        if root is None:
            raise HTTPException(404)
        target = (root / path).resolve()
        if root not in target.parents or not target.is_file():
            raise HTTPException(404)
        return FileResponse(target)

    @app.post("/api/predict")
    async def predict(
        state: str = Form(""),
        questions: str = Form(...),
        images: list[UploadFile] = File(default=[]),
        audio: UploadFile | None = File(default=None),
        example_images: str = Form(""),
        example_audio: str = Form(""),
        detail: bool = Form(False),
        compare: bool = Form(True),
    ):
        try:
            qs = json.loads(questions)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"questions is not valid JSON: {e}")
        imgs = [Image.open(io.BytesIO(await f.read())).convert("RGB") for f in images if f.filename]
        for rel in filter(None, example_images.split("|")):
            target = (root / rel).resolve() if root else None
            if target is None or root not in target.parents:
                raise HTTPException(400, "unknown example image")
            imgs.append(Image.open(target).convert("RGB"))
        wav = None
        if audio is not None and audio.filename:
            wav = _decode_audio(await audio.read(), omni.audio_encoder.sampling_rate)
        elif example_audio:
            target = (root / example_audio).resolve() if root else None
            if target is None or root not in target.parents:
                raise HTTPException(400, "unknown example audio")
            from .encoders import load_audio

            wav = load_audio(target, omni.audio_encoder.sampling_rate)
        image = imgs if len(imgs) > 1 else (imgs[0] if imgs else None)
        t0 = time.perf_counter()
        try:
            result = omni.predict(state, qs, image=image, audio=wav, detail=detail)
        except ValueError as e:
            raise HTTPException(400, str(e))
        elapsed = (time.perf_counter() - t0) * 1000
        out = {"result": result, "ms": round(elapsed, 1), "tokens": 256 if detail else 64,
               "inputs": {"images": len(imgs), "audio_seconds": round(len(wav) / 16000, 2) if wav is not None else 0}}
        if compare and (image is not None or wav is not None):
            out["baseline"] = omni.predict(state, qs)
        return out

    return app


def _decode_audio(data, sampling_rate):
    """Uploaded audio of any format soundfile or librosa reads, as mono float32."""
    import soundfile as sf

    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        wav = wav.mean(1)
    except Exception:
        import librosa

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".audio") as f:
            f.write(data)
            f.flush()
            wav, sr = librosa.load(f.name, sr=None, mono=True)
    if sr != sampling_rate:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=sampling_rate)
    return np.asarray(wav, dtype=np.float32)


def main(argv=None):
    import argparse

    import uvicorn

    from .pipeline import Omni

    ap = argparse.ArgumentParser(prog="laya-omni serve")
    ap.add_argument("--fusion", required=True)
    ap.add_argument("--laya", required=True)
    ap.add_argument("--image-encoder")
    ap.add_argument("--audio-encoder")
    ap.add_argument("--examples", help="data root with converted datasets, for the example gallery")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8030)
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    omni = Omni.load(args.fusion, args.laya, args.image_encoder, args.audio_encoder, device=args.device)
    uvicorn.run(create_app(omni, args.examples), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
