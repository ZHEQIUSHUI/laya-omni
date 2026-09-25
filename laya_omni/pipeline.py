"""One object from files to decisions: Laya, the fusion, and both frozen encoders.

    from laya_omni import Omni

    omni = Omni.load("runs/formal-v2", laya="models/laya-multilingual",
                     image_encoder="models/siglip2-base-patch16-256",
                     audio_encoder="models/qwen3-asr-0.6b-audio-encoder")
    omni.predict("Image.", {"animal": {"type": "choice", "instructions": "Which animal is this?",
                                       "criteria": ["cat", "dog", "bird"]}}, image="photo.jpg")
    omni.predict("Audio clip.", {"alarm": {"type": "noul", "instructions": "Is an alarm ringing?"}},
                 audio="clip.wav")
    omni.predict(state, questions)  # no image, no audio: plain Laya

Images default to 64 tokens (fast); detail=True uses 256 (charts, documents, maps).
Several images: pass a list.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .agent import Agent
from .data import pool_tokens


def _image(x):
    from PIL import Image

    if isinstance(x, (str, Path)):
        return Image.open(x).convert("RGB")
    return x.convert("RGB")


def _audio(x, sampling_rate):
    if isinstance(x, (str, Path)):
        from .encoders import load_audio

        return load_audio(x, sampling_rate)
    return np.asarray(x, dtype=np.float32)


class Omni:
    def __init__(self, agent, image_encoder=None, audio_encoder=None):
        self.agent = agent
        self.image_encoder = image_encoder
        self.audio_encoder = audio_encoder

    @classmethod
    def load(cls, fusion, laya, image_encoder=None, audio_encoder=None, device=None, dtype="float32"):
        """fusion: a training run directory (fusion.safetensors + fusion_config.json)."""
        from .encoders import AudioEncoder, ImageEncoder

        agent = Agent(laya, fusion, device=device, dtype=dtype)
        cfg = json.loads((Path(fusion) / "fusion_config.json").read_text())
        dims = cfg["in_dims"]
        img = aud = None
        if "image" in dims:
            img = ImageEncoder(image_encoder or cfg.get("image_encoder"), device=str(agent.device), pool=1)
        if "audio" in dims and audio_encoder:
            aud = AudioEncoder(audio_encoder, device=str(agent.device))
        return cls(agent, img, aud)

    def encode_image(self, image, detail=False):
        """(tokens, D) features for one image, or a list of them."""
        items = image if isinstance(image, (list, tuple)) else [image]
        with torch.inference_mode():
            h = self.image_encoder([_image(x) for x in items])
            h = pool_tokens(h, 256 if detail else 64).float()
        out = list(h)
        return out if isinstance(image, (list, tuple)) else out[0]

    def encode_audio(self, audio):
        items = audio if isinstance(audio, (list, tuple)) else [audio]
        feats = [f.float() for f in self.audio_encoder([_audio(x, self.audio_encoder.sampling_rate) for x in items])]
        return feats if isinstance(audio, (list, tuple)) else feats[0]

    def predict(self, state, questions, *, image=None, audio=None, detail=False):
        """Laya's output schema. image: path, PIL image or list; audio: path, 16 kHz array or list."""
        if image is not None and self.image_encoder is None:
            raise ValueError("This fusion has no image encoder")
        if audio is not None and self.audio_encoder is None:
            raise ValueError("This fusion was loaded without an audio encoder")
        img = self.encode_image(image, detail) if image is not None else None
        aud = self.encode_audio(audio) if audio is not None else None
        return self.agent.predict(state, questions, image=img, audio=aud)
