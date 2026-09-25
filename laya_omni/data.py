"""Training samples: a Laya question, its answer, and optional cached modality features.

A sample is a dict:

    {"state": str, "question": {"type", "instructions", "criteria"}, "label": int,
     "features": {"image": (store_name, row), "audio": (store_name, row)}, "source": str}

`label` indexes the options in the order the question lists them (noul: 0 false,
1 true). `features` may omit any modality, or be empty.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .agent import collate_items


class FeatureStore:
    """Named float16 feature arrays written by scripts/cache_features.py.

    Fixed-size features (images) are (N, T, D) and indexed by row; variable-length
    ones (audio) are (total frames, D) with a [start, end) span per clip.
    """

    def __init__(self):
        self.arrays, self.index = {}, {}

    def add(self, name, prefix):
        prefix = Path(prefix)
        self.arrays[name] = np.load(prefix.with_suffix(".npy"), mmap_mode="r")
        meta = json.loads(prefix.with_suffix(".json").read_text())
        self.index[name] = meta["index"] if "index" in meta else meta["spans"]
        return self

    def row(self, name, key):
        return self.index[name][key]

    def dims(self, name):
        return self.arrays[name].shape[-1]

    def get(self, name, row):
        if isinstance(row, (list, tuple)):
            return self.arrays[name][row[0] : row[1]]
        return self.arrays[name][row]


def feature_prefix(roots, stem):
    """The first of the ':'-separated feature directories that holds <stem>.npy."""
    dirs = [r for r in str(roots).split(":") if r]
    for d in dirs:
        if (Path(d) / f"{stem}.npy").exists():
            return Path(d) / stem
    return Path(dirs[0] if dirs else ".") / stem


def shuffled(question, label, rng):
    """Shuffle a choice question's options, returning the question and remapped label."""
    if question["type"] != "choice":
        return question, label
    crit = question["criteria"]
    items = list(crit.items()) if isinstance(crit, dict) else [(c, None) for c in crit]
    order = list(range(len(items)))
    rng.shuffle(order)
    new = [items[i] for i in order]
    q = dict(question)
    q["criteria"] = dict(new) if isinstance(crit, dict) else [k for k, _ in new]
    return q, order.index(label)


class GameSource:
    """Frames from scripts/make_game_data.py, each asked with a random phrasing and label set."""

    def __init__(self, data_dir, game, split, store, store_name, augment=True):
        spec = json.loads((Path(data_dir) / "questions.json").read_text())
        self.name, self.state, self.spec = f"game:{game}", spec["state"], spec["questions"][game]
        self.rows = []
        for line in open(Path(data_dir) / f"{game}.jsonl"):
            r = json.loads(line)
            if r["split"] == split:
                self.rows.append((store.row(store_name, r["image"]), r["label"]))
        self.store_name, self.augment = store_name, augment

    def samples(self, rng):
        out = []
        for row, label in self.rows:
            if not self.augment:  # the first phrasing and label set, canonical order
                q = {"type": "choice", "instructions": self.spec["instructions"][0], "criteria": self.spec["labels"][0]}
                out.append({"state": self.state, "question": q, "label": label, "features": {"image": (self.store_name, row)}, "source": self.name})
                continue
            q = {
                "type": "choice",
                "instructions": rng.choice(self.spec["instructions"]),
                "criteria": rng.choice(self.spec["labels"]),
            }
            q, y = shuffled(q, label, rng)
            out.append({"state": self.state, "question": q, "label": y, "features": {"image": (self.store_name, row)}, "source": self.name})
        return out


class JsonlSource:
    """Converted datasets: jsonl rows of {state, question, label, image?, audio?, split}."""

    def __init__(self, path, split, store, stores, name=None, shuffle_options=True):
        """stores: {"image": store_name, "audio": store_name} for the keys rows may carry."""
        self.name = name or Path(path).stem
        self.shuffle_options = shuffle_options
        self.rows, self.missing = [], 0
        for line in open(path):
            r = json.loads(line)
            if r.get("split", "train") != split:
                continue
            try:
                feats = self._features(r, store, stores)
            except KeyError:  # its image or clip could not be encoded when caching
                self.missing += 1
                continue
            self.rows.append({"state": r.get("state", ""), "question": r["question"], "label": r["label"], "features": feats, "source": self.name})
        if self.missing:
            print(f"{self.name} ({split}): {self.missing} rows skipped, their media are missing from the cache")

    @staticmethod
    def _features(r, store, stores):
        feats = {}
        for m, s in stores.items():
            # A "raw:<dir>" store means the file is read and encoded at training time.
            look = (lambda x, s=s: (s, x)) if s.startswith(RAW) else (lambda x, s=s: (s, store.row(s, x)))
            if r.get(m):
                feats[m] = look(r[m])
            elif r.get(m + "s"):  # several images ("images": [...]) or clips
                feats[m] = [look(x) for x in r[m + "s"]]
        return feats

    def samples(self, rng):
        if not self.shuffle_options:
            return list(self.rows)
        out = []
        for r in self.rows:
            q, y = shuffled(r["question"], r["label"], rng)
            out.append({**r, "question": q, "label": y})
        return out


RAW = "raw:"
IMAGE_SIZE = 256


def load_pixels(path, size=IMAGE_SIZE):
    """SigLIP preprocessing without the processor: bilinear resize, [-1, 1] (3, size, size)."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy((x - 0.5) / 0.5).permute(2, 0, 1)


class Batches(torch.utils.data.Dataset):
    """One epoch's samples cut into batches, collated on the CPU (DataLoader workers):
    tokenised text, cached audio features, and decoded image pixels."""

    def __init__(self, samples, batch_size, tok, max_len, head_max_len, store, modalities=("image", "audio")):
        self.chunks = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]
        self.tok, self.max_len, self.head_max_len = tok, max_len, head_max_len
        self.store, self.modalities = store, modalities

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        from .agent import prepare_items

        samples = self.chunks[i]
        items = [prepare_items(self.tok, self.max_len, self.head_max_len, s["state"], {"q": s["question"]})[0][0] for s in samples]
        batch = collate_items(items, self.tok.pad_token_id)
        cached, pixels, index = {}, [], None
        for m in self.modalities:
            refs = [s["features"].get(m) for s in samples]
            if not any(refs):
                continue
            items_m = [[] if r is None else (r if isinstance(r, list) else [r]) for r in refs]
            if any(x[0].startswith(RAW) for it in items_m for x in it):
                n = max(len(it) for it in items_m)
                index = torch.full((len(samples), n), -1, dtype=torch.long)
                for b, it in enumerate(items_m):
                    for j, (store_name, rel) in enumerate(it):
                        index[b, j] = len(pixels)
                        pixels.append(load_pixels(Path(store_name[len(RAW) :]) / rel))
            else:
                cached[m] = items_m
        feats = {m: stack_cached(self.store, it) for m, it in cached.items()}
        labels = torch.tensor([s["label"] for s in samples])
        return batch, feats, (torch.stack(pixels) if pixels else None), index, labels


def stack_cached(store, items):
    """Cached features for one modality: (feats, mask, present) CPU tensors, (B, N, T, D) or (B, T, D)."""
    arrays = [[store.get(*x) for x in it] for it in items]
    n = max(len(a) for a in arrays)
    t = max(x.shape[0] for a in arrays for x in a)
    d = next(x.shape[-1] for a in arrays for x in a)
    feats = np.zeros((len(items), n, t, d), dtype=np.float32)
    mask = np.zeros((len(items), n, t), dtype=bool)
    for i, a in enumerate(arrays):
        for j, x in enumerate(a):
            feats[i, j, : x.shape[0]] = x
            mask[i, j, : x.shape[0]] = True
    present = torch.tensor([[j < len(a) for j in range(n)] for a in arrays])
    feats, mask = torch.from_numpy(feats), torch.from_numpy(mask)
    return (feats[:, 0], mask[:, 0], present[:, 0]) if n == 1 else (feats, mask, present)


def to_device(batch, feats, pixels, index, labels, device, image_encoder=None, image_tokens=64):
    """Move a CPU batch to the device, encoding raw images (frozen) at the chosen token count."""
    batch = {k: v.to(device) for k, v in batch.items()}
    mods = {m: tuple(x.to(device) for x in v) for m, v in feats.items()}
    if pixels is not None:
        with torch.no_grad():
            h = image_encoder.encode_pixels(pixels.to(device))  # (M, 256, D)
            h = pool_tokens(h, image_tokens).float()
        b, n = index.shape
        index = index.to(device)
        out = torch.zeros(b, n, h.shape[1], h.shape[2], device=device)
        present = index >= 0
        out[present] = h[index[present]]
        mods["image"] = (out[:, 0], None, present[:, 0]) if n == 1 else (out, None, present)
    return batch, mods, labels.to(device)


def pool_tokens(h, tokens):
    """(M, side*side, D) patch features -> (M, tokens, D) by average pooling on the grid."""
    m, t, d = h.shape
    side, target = int(round(t**0.5)), int(round(tokens**0.5))
    if target == side:
        return h
    grid = h.transpose(1, 2).reshape(m, d, side, side)
    return F.avg_pool2d(grid, side // target).flatten(2).transpose(1, 2)


def mix(sources, rng, cap=None, weights=None):
    """One epoch: every source's samples (at most `cap` each), shuffled together.

    weights: {source name: factor}. A factor above 1 draws the source several times
    (each draw reshuffles its options), below 1 draws a fraction; the cap scales
    with the factor.
    """
    out = []
    for src in sources:
        w = (weights or {}).get(src.name, 1.0)
        limit = int(cap * w) if cap else None
        drawn = []
        for _ in range(max(1, int(w + 0.999))):
            drawn += src.samples(rng)
        want = int(len(drawn) / max(1, int(w + 0.999)) * w)
        if limit is not None:
            want = min(want, limit)
        if want < len(drawn):
            drawn = rng.sample(drawn, want)
        out.extend(drawn)
    rng.shuffle(out)
    return out


def seeded(seed):
    return random.Random(seed)
