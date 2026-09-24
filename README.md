# laya-omni

**Laya typed decisions with optional image and audio inputs.**

[Laya](https://github.com/NandhaKishorM/laya) answers constrained questions
(`choice`, `score`, `noul`) about a text state in one forward pass, with
calibrated probabilities and no generated tokens. laya-omni lets the state also
include an image, an audio clip, or both.

Status: early work in progress.

## Design

- **Laya is unchanged and frozen.** The upstream checkpoint loads as is.
- **The feature encoders are frozen**:
  - images use [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-256) (base/16, 256 px);
  - audio uses the audio encoder of [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B).
- **Only a small fusion adapter is trained.** Each modality gets a Perceiver
  resampler that turns its features into 32 tokens. After each of Laya's two
  decision-head layers, a Flamingo-style gated cross-attention block lets the
  text tokens read those modality tokens. The gates start at zero, so training
  starts from the original Laya.
- **Each modality is optional.** With no image and no audio the adapter is
  skipped, and the outputs are bit-identical to Laya. A test enforces this
  (`tests/test_identity.py`).
- **Training uses a proper scoring rule**: cross-entropy over the option
  markers. Temperatures are calibrated separately for each modality combination.

```python
from laya_omni import load
from laya_omni.encoders import ImageEncoder

agent = load("models/laya-multilingual", "runs/games-v1")
image = ImageEncoder("models/siglip2-base-patch16-256")(frame)
agent.predict("Game screen.", {"move": {"type": "choice", "instructions": "...", "criteria": ["up", "down"]}}, image=image)
agent.predict(state, questions)  # no image or audio: plain Laya
```

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
