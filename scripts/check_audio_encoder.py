"""Check laya-omni's AudioEncoder against the official qwen-asr implementation.

Two halves, because qwen-asr pins an older transformers:

    # in an environment with qwen-asr installed
    python scripts/check_audio_encoder.py reference models/Qwen3-ASR-0.6B ref.npz
    # in laya-omni's environment
    python scripts/check_audio_encoder.py compare models/qwen3-asr-0.6b-audio-encoder ref.npz

The reference half makes test clips (0.5 s to 45 s, so both the 100-frame chunks
and the 8 s attention windows are exercised), runs qwen-asr's own feature
extractor and audio tower one clip at a time as qwen-asr does, and stores the
tower's hidden states (before the projection) and outputs (after it).

Two known differences in qwen-asr itself, both handled here:
- Its transformers backend only applies the 8 s attention windows with
  flash-attention; with sdpa or eager it passes no mask and every frame attends
  to the whole clip (`_prepare_attention_mask` is defined but never called). Its
  vLLM backend, and transformers' own Qwen3ASREncoder, use the windows, so the
  reference applies qwen-asr's own window mask.
- A clip under 1 s is a single partial chunk. qwen-asr pads chunks to the longest
  chunk in the call, so such a clip alone is not padded to 100 frames, but it is
  when batched with a longer one; laya-omni always pads (the batched behaviour),
  which changes the last frame's convolution. Reported, not counted.
"""

import sys

import numpy as np
import torch

SR = 16000
# cuDNN picks TF32 convolutions by default (~1e-3 relative error) and the choice can
# differ between batch sizes; turn it off so the comparison sees the implementations.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def clips():
    rng = np.random.default_rng(0)
    out = {}
    for name, sec in (("short", 0.5), ("odd", 3.3), ("multi", 12.0), ("long", 45.0)):
        t = np.arange(int(SR * sec)) / SR
        f0 = 120 + 80 * np.sin(2 * np.pi * 0.7 * t)  # a voice-like wandering pitch
        wav = sum(np.sin(2 * np.pi * k * np.cumsum(f0) / SR) / k for k in range(1, 6))
        wav *= 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2  # syllable-rate envelope
        wav = 0.2 * wav / np.abs(wav).max() + 0.01 * rng.standard_normal(len(t))
        out[name] = wav.astype(np.float32)
    return out


def official_modeling():
    """qwen-asr's model code, without importing the qwen_asr package (whose __init__
    pulls in the forced aligner and its tokenizer dependencies)."""
    import importlib
    import importlib.util
    import types
    from pathlib import Path

    base = Path(importlib.util.find_spec("qwen_asr").submodule_search_locations[0]) / "core" / "transformers_backend"
    pkg = types.ModuleType("qwen_asr_backend")
    pkg.__path__ = [str(base)]
    sys.modules["qwen_asr_backend"] = pkg
    return importlib.import_module("qwen_asr_backend.modeling_qwen3_asr")


def reference(src, path):
    from transformers import WhisperFeatureExtractor

    official = official_modeling()
    Qwen3ASRForConditionalGeneration = official.Qwen3ASRForConditionalGeneration

    model = Qwen3ASRForConditionalGeneration.from_pretrained(src, torch_dtype=torch.float32).cuda().eval()
    tower = model.thinker.audio_tower
    attention = official.Qwen3ASRAudioAttention.forward

    def windowed(self, hidden_states, cu_seqlens=None, attention_mask=None, **kwargs):
        if attention_mask is None and "flash" not in self.config._attn_implementation:
            attention_mask = tower._prepare_attention_mask(hidden_states, cu_seqlens)
        return attention(self, hidden_states, cu_seqlens, attention_mask, **kwargs)

    official.Qwen3ASRAudioAttention.forward = windowed
    fe = WhisperFeatureExtractor.from_pretrained(src)
    captured = {}
    tower.ln_post.register_forward_hook(lambda m, i, o: captured.__setitem__("hidden", o))
    result = {}
    with torch.inference_mode():
        for name, wav in clips().items():
            mel = fe(wav, sampling_rate=SR, return_tensors="pt", return_attention_mask=True, truncation=False, padding="longest")
            feats, n = mel["input_features"][0].cuda(), int(mel["attention_mask"].sum())
            out = tower(feats[:, :n], feature_lens=torch.tensor([n], device="cuda")).last_hidden_state
            result[f"wav:{name}"] = wav
            result[f"hidden:{name}"] = captured["hidden"].float().cpu().numpy()
            result[f"projected:{name}"] = out.float().cpu().numpy()
            print(name, "mel frames", n, "encoder frames", out.shape[0], "dims", captured["hidden"].shape[-1], out.shape[-1])
    np.savez(path, **result)


def compare(encoder_dir, path):
    from laya_omni.encoders import AudioEncoder

    ref = np.load(path)
    names = [k.split(":", 1)[1] for k in ref.files if k.startswith("wav:")]
    worst = 0.0
    sub_second = [n for n in names if len(ref[f"wav:{n}"]) < SR]
    for output in ("hidden", "projected"):
        enc = AudioEncoder(encoder_dir, dtype=torch.float32, output=output)
        batched = enc([ref[f"wav:{n}"] for n in names])  # all clips in one padded batch
        for n, got_b in zip(names, batched):
            want = ref[f"{output}:{n}"]
            got = enc(ref[f"wav:{n}"]).cpu().numpy()
            assert got.shape == want.shape, (n, output, got.shape, want.shape)
            d1 = np.abs(got - want).max() / np.abs(want).max()
            d2 = np.abs(got_b.cpu().numpy() - want).max() / np.abs(want).max()
            note = "  (under 1 s: chunk padding differs, not counted)" if n in sub_second else ""
            if not note:
                worst = max(worst, d1, d2)
            print(f"{output:9s} {n:6s} {want.shape}  max rel diff: single {d1:.2e}  batched {d2:.2e}{note}")
    print("PASS" if worst < 1e-3 else "FAIL", f"worst {worst:.2e}")


if __name__ == "__main__":
    {"reference": reference, "compare": compare}[sys.argv[1]](*sys.argv[2:])
