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
        self.rows = []
        for line in open(path):
            r = json.loads(line)
            if r.get("split", "train") != split:
                continue
            feats = {m: (s, store.row(s, r[m])) for m, s in stores.items() if r.get(m)}
            self.rows.append({"state": r.get("state", ""), "question": r["question"], "label": r["label"], "features": feats, "source": self.name})

    def samples(self, rng):
        if not self.shuffle_options:
            return list(self.rows)
        out = []
        for r in self.rows:
            q, y = shuffled(r["question"], r["label"], rng)
            out.append({**r, "question": q, "label": y})
        return out


def collate(agent, store, samples, modalities, device, dtype=torch.float32):
    """Tokenize and batch samples. Returns (batch, modality inputs, labels)."""
    items = []
    for s in samples:
        prepared, _ = agent.prepare(s["state"], {"q": s["question"]})
        items.append(prepared[0])
    batch = {k: v.to(device) for k, v in collate_items(items, agent.tok.pad_token_id).items()}
    mods = {}
    for m in modalities:
        refs = [s["features"].get(m) for s in samples]
        if not any(refs):
            continue
        arrays = [store.get(*r) if r else None for r in refs]
        shape = next(a.shape for a in arrays if a is not None)
        t = max(a.shape[0] for a in arrays if a is not None)
        feats = np.zeros((len(samples), t, shape[-1]), dtype=np.float32)
        mask = np.zeros((len(samples), t), dtype=bool)
        for i, a in enumerate(arrays):
            if a is not None:
                feats[i, : a.shape[0]] = a
                mask[i, : a.shape[0]] = True
        present = torch.tensor([r is not None for r in refs], device=device)
        mods[m] = (torch.from_numpy(feats).to(device, dtype), torch.from_numpy(mask).to(device), present)
    labels = torch.tensor([s["label"] for s in samples], device=device)
    return batch, mods, labels


def mix(sources, rng, cap=None):
    """One epoch: every source's samples (at most `cap` each), shuffled together."""
    out = []
    for src in sources:
        s = src.samples(rng)
        if cap and len(s) > cap:
            s = rng.sample(s, cap)
        out.extend(s)
    rng.shuffle(out)
    return out


def seeded(seed):
    return random.Random(seed)
