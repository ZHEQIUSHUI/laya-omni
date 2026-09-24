"""Extract the vision tower from a Qwen3.5 checkpoint into a standalone directory.

Only `model.visual.*` is kept (Qwen3.5-0.8B: about 0.1B of 0.8B parameters). The
result loads with transformers' own Qwen3_5VisionModel:

    <out>/config.json               Qwen3_5VisionConfig (the checkpoint's vision_config)
    <out>/model.safetensors         the vision tower, including its 2x2 patch merger
    <out>/preprocessor_config.json  the checkpoint's image processor settings

    python scripts/extract_vision_encoder.py models/Qwen3.5-0.8B models/qwen3.5-0.8b-vision
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "model.visual."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Qwen3.5 checkpoint directory")
    ap.add_argument("out")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    vision = json.loads((src / "config.json").read_text())["vision_config"]
    vision = {k: v for k, v in vision.items() if k not in ("architectures", "_name_or_path", "model_type")}
    Qwen3_5VisionConfig(**vision).save_pretrained(out)

    weights = {}
    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(shard, "pt") as f:
            for key in f.keys():
                if key.startswith(PREFIX):
                    weights[key[len(PREFIX) :]] = f.get_tensor(key).contiguous()
    if not weights:
        raise SystemExit(f"No {PREFIX}* tensors in {src}")
    save_file(weights, out / "model.safetensors", metadata={"source": src.name})
    shutil.copy(src / "preprocessor_config.json", out / "preprocessor_config.json")
    if (src / "LICENSE").exists():
        shutil.copy(src / "LICENSE", out / "UPSTREAM_LICENSE")
    total = sum(t.numel() for t in weights.values())
    print(f"{len(weights)} tensors, {total / 1e6:.1f}M parameters -> {out}")


if __name__ == "__main__":
    main()
