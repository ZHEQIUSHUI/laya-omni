"""Frozen feature encoders. Their outputs are what the fusion resamplers read.

Encoders are never trained. Training caches their features once; inference
runs them per request. Features are (T, D) float tensors on the encoder device.
"""

import torch
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
