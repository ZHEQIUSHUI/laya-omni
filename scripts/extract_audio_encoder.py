"""Extract the audio encoder from a Qwen3-ASR checkpoint into a standalone directory.

Only the audio tower is kept (Qwen3-ASR-0.6B: 186M of 938M parameters); the
text decoder is dropped. The result loads with transformers' own
Qwen3ASREncoder, so no qwen-asr package is needed:

    <out>/config.json               Qwen3ASREncoderConfig (the checkpoint's audio_config)
    <out>/model.safetensors         encoder.* and projector.* (proj1/proj2 -> linear_1/linear_2)
    <out>/preprocessor_config.json  Qwen3ASRFeatureExtractor, 128-bin log-mel at 16 kHz

    python scripts/extract_audio_encoder.py models/Qwen3-ASR-0.6B models/qwen3-asr-0.6b-audio-encoder
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "thinker.audio_tower."
RENAME = {"proj1.": "projector.linear_1.", "proj2.": "projector.linear_2."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Qwen3-ASR checkpoint directory")
    ap.add_argument("out")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import Qwen3ASREncoderConfig, Qwen3ASRFeatureExtractor

    audio = json.loads((src / "config.json").read_text())["thinker_config"]["audio_config"]
    audio = {k: v for k, v in audio.items() if k not in ("architectures", "_name_or_path", "model_type")}
    Qwen3ASREncoderConfig(**audio).save_pretrained(out)

    weights = {}
    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(shard, "pt") as f:
            for key in f.keys():
                if not key.startswith(PREFIX):
                    continue
                name = key[len(PREFIX) :]
                for old, new in RENAME.items():
                    if name.startswith(old):
                        name = new + name[len(old) :]
                        break
                else:
                    name = "encoder." + name
                weights[name] = f.get_tensor(key).contiguous()
    save_file(weights, out / "model.safetensors", metadata={"source": str(src.name)})

    Qwen3ASRFeatureExtractor(feature_size=audio.get("num_mel_bins", 128)).save_pretrained(out)
    for name in ("LICENSE", "README.md"):
        if (src / name).exists():
            shutil.copy(src / name, out / f"UPSTREAM_{name}")
    total = sum(t.numel() for t in weights.values())
    print(f"{len(weights)} tensors, {total / 1e6:.1f}M parameters -> {out}")


if __name__ == "__main__":
    main()
