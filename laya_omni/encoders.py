"""Frozen feature encoders. Their outputs are what the fusion resamplers read.

Encoders are never trained. Training caches their features once; inference
runs them per request. Features are (T, D) float tensors on the encoder device.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder:
    """SigLIP 2 patch features, 2x2 average-pooled: 256 patches -> 64 tokens of 768."""

    name = "image"

    def __init__(self, path, device="cuda", dtype=torch.float16, pool=2):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, torch_dtype=dtype).vision_model.to(device).eval()
        self.device, self.dtype, self.pool = device, dtype, pool
        self.dims = self.model.config.hidden_size

    @torch.inference_mode()
    def __call__(self, images):
        """images: a PIL image or list of them -> (B, T, D) features."""
        single = not isinstance(images, (list, tuple))
        pixels = self.processor(images=[images] if single else list(images), return_tensors="pt")
        h = self.model(pixel_values=pixels["pixel_values"].to(self.device, self.dtype)).last_hidden_state
        if self.pool > 1:
            b, t, d = h.shape
            side = int(t**0.5)
            h = h.transpose(1, 2).reshape(b, d, side, side)
            h = F.avg_pool2d(h, self.pool).flatten(2).transpose(1, 2)
        return h[0] if single else h


def load_audio(path, sampling_rate=16000):
    """Any file librosa can read, as mono float32 at the encoder's sampling rate."""
    import librosa

    wav, _ = librosa.load(path, sr=sampling_rate, mono=True)
    return wav.astype(np.float32)


class AudioEncoder:
    """Qwen3-ASR's audio encoder (AuT): 16 kHz audio -> about 12.5 frames per second.

    Loads the standalone directory written by scripts/extract_audio_encoder.py with
    transformers' own Qwen3ASREncoder. `output="hidden"` returns the encoder's last
    hidden states (d_model, 896 for the 0.6B model); `output="projected"` adds the
    two-layer projection Qwen3-ASR feeds its decoder (output_dim, 1024).
    """

    name = "audio"

    def __init__(self, path, device="cuda", dtype=torch.float16, output="hidden"):
        from safetensors.torch import load_file
        from transformers import Qwen3ASREncoder, Qwen3ASREncoderConfig, Qwen3ASRFeatureExtractor

        if output not in ("hidden", "projected"):
            raise ValueError("output must be 'hidden' or 'projected'")
        cfg = Qwen3ASREncoderConfig.from_pretrained(path)
        self.processor = Qwen3ASRFeatureExtractor.from_pretrained(path)
        self.model = Qwen3ASREncoder(cfg)
        self.projector = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.output_dim)
        )
        weights = load_file(f"{path}/model.safetensors")
        part = lambda prefix: {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
        self.model.load_state_dict(part("encoder."), strict=True)
        proj = part("projector.")
        self.projector.load_state_dict({k.replace("linear_1", "0").replace("linear_2", "2"): v for k, v in proj.items()})
        self.model.to(device, dtype).eval()
        self.projector.to(device, dtype).eval()
        self.device, self.dtype, self.output = device, dtype, output
        self.chunk = 2 * cfg.n_window
        self.sampling_rate = self.processor.sampling_rate
        self.dims = cfg.d_model if output == "hidden" else cfg.output_dim

    def lengths(self, mel_mask):
        """Encoder frames per clip: each 2*n_window-frame chunk goes through three stride-2 convs."""
        b = mel_mask.shape[0]
        per_chunk = mel_mask.view(b, -1, self.chunk).sum(-1).long()
        for _ in range(3):
            per_chunk = torch.where(per_chunk > 0, (per_chunk - 1) // 2 + 1, torch.zeros_like(per_chunk))
        return per_chunk.sum(-1)

    def mel(self, clips):
        """Log-mel per clip, then padded into one batch: (B, 128, frames) features and mask.

        Each clip goes through the feature extractor on its own. Batched extraction
        zero-pads short clips before the STFT, so their last frames would depend on
        what else is in the batch; alone, a clip gets the same features as
        qwen-asr computes for it alone.
        """
        feats = []
        for clip in clips:
            mel = self.processor([clip], sampling_rate=self.sampling_rate, return_tensors="pt", padding="longest", return_attention_mask=True)
            n = int(mel["attention_mask"].sum())
            feats.append(mel["input_features"][0, :, :n])
        width = -(-max(f.shape[1] for f in feats) // self.chunk) * self.chunk
        batch = torch.zeros(len(feats), feats[0].shape[0], width)  # the extractor's own padding value
        mask = torch.zeros(len(feats), width, dtype=torch.long)
        for i, f in enumerate(feats):
            batch[i, :, : f.shape[1]] = f
            mask[i, : f.shape[1]] = 1
        return batch, mask

    @torch.inference_mode()
    def __call__(self, audio):
        """audio: a 1-D float array at 16 kHz (or a list of them) -> (T, D) features (or a list).

        The result for a clip does not depend on the other clips in the batch.
        """
        single = isinstance(audio, np.ndarray) and audio.ndim == 1
        clips = [audio] if single else list(audio)
        feats, mask = self.mel(clips)
        feats, mask = feats.to(self.device, self.dtype), mask.to(self.device)
        packed = self.model(input_features=feats, input_features_mask=mask).last_hidden_state
        if self.output == "projected":
            packed = self.projector(packed)
        out = list(packed.split(self.lengths(mask).tolist()))
        return out[0] if single else out
