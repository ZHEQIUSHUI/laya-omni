"""Convert public audio datasets into laya-omni rows.

Each dataset becomes <out>/<name>.jsonl plus 16 kHz mono WAV files under
<out>/<name>/. Rows look like {"state", "question", "label", "audio", "split"};
see laya_omni/data.py. Questions vary in phrasing and language, choice options
are a random subset around the answer, and noul questions are balanced.

    python scripts/convert_audio.py esc50 datasets/esc50 --out data/audio
    python scripts/convert_audio.py cremad datasets/CREMA-D --out data/audio
    python scripts/convert_audio.py vocalsound datasets/vocalsound --out data/audio
    python scripts/convert_audio.py gtzan datasets/gtzan --out data/audio
"""

import argparse
import glob
import io
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000
STATE = "Audio clip."


def parquet_rows(src):
    for path in sorted(glob.glob(str(Path(src) / "**/*.parquet"), recursive=True)):
        yield from pq.read_table(path).to_pylist()


def decode(cell):
    import librosa

    data = cell.get("bytes") if isinstance(cell, dict) else cell
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav.astype(np.float32)


def choice(rng, answer, pool, ask, k_range=(2, 4)):
    others = rng.sample([p for p in pool if p != answer], rng.randint(*k_range) - 1)
    options = others + [answer]
    rng.shuffle(options)
    return {"type": "choice", "instructions": rng.choice(ask), "criteria": options}, options.index(answer)


SOUND_ASK = [
    "What is this sound?",
    "Which of these made the sound?",
    "What can be heard in the clip?",
    "Identify the sound source.",
    "这是什么声音？",
    "录音里听到的是哪一种声音？",
]
SOUND_IS = ["Is this the sound of {}?", "Can {} be heard in this clip?", "这段声音是{}吗？"]


def esc50(src, rng):
    rows = list(parquet_rows(src))
    names = sorted({r["category"].replace("_", " ") for r in rows})
    for r in rows:
        name = r["category"].replace("_", " ")
        split = "test" if r["fold"] == 5 else "train"
        q, y = choice(rng, name, names, SOUND_ASK)
        yield split, r["filename"], r["audio"], {"state": STATE, "question": q, "label": y}
        other = name if rng.random() < 0.5 else rng.choice([n for n in names if n != name])
        yield split, r["filename"], r["audio"], {
            "state": STATE,
            "question": {"type": "noul", "instructions": rng.choice(SOUND_IS).format(other)},
            "label": int(other == name),
        }


EMOTIONS = {"angry": "angry", "sad": "sad", "disgust": "disgusted", "fear": "afraid", "neutral": "neutral", "happy": "happy"}
EMOTION_ZH = {"angry": "生气", "sad": "悲伤", "disgust": "厌恶", "fear": "害怕", "neutral": "平静", "happy": "开心"}
EMOTION_ASK = [
    "How does the speaker sound?",
    "What emotion is in the speaker's voice?",
    "Which feeling does this voice express?",
]
EMOTION_ASK_ZH = ["说话人是什么情绪？", "这段语音听起来是什么心情？"]
EMOTION_IS = ["Does the speaker sound {}?", "Is the speaker {}?"]
GENDER_ASK = ["Is the speaker a man or a woman?", "Who is speaking?", "说话的是男声还是女声？"]


def cremad(src, rng):
    for r in parquet_rows(src):
        emotion = (r.get("major_emotion") or "").lower()
        if emotion not in EMOTIONS:
            continue
        actor = int(re.match(r"(\d{4})", Path(r["file"]).name).group(1))
        split = "test" if actor % 10 == 0 else "train"  # unseen voices in test
        zh = rng.random() < 0.3
        names = EMOTION_ZH if zh else EMOTIONS
        q, y = choice(rng, names[emotion], list(names.values()), EMOTION_ASK_ZH if zh else EMOTION_ASK, (2, 6))
        # Human votes per emotion ride along for calibration checks.
        votes = {k: float(r[k]) for k in EMOTIONS if r.get(k) is not None}
        yield split, r["file"], r["audio"], {"state": STATE, "question": q, "label": y, "votes": votes}
        other = emotion if rng.random() < 0.5 else rng.choice([e for e in EMOTIONS if e != emotion])
        q = {"type": "noul", "instructions": rng.choice(EMOTION_IS).format(EMOTIONS[other])}
        yield split, r["file"], r["audio"], {"state": STATE, "question": q, "label": int(other == emotion)}
        gender = (r.get("gender") or "").lower()
        if gender in ("male", "female") and rng.random() < 0.3:
            labels = ["man", "woman"] if rng.random() < 0.5 else ["男声", "女声"]
            q = {"type": "choice", "instructions": rng.choice(GENDER_ASK), "criteria": labels}
            yield split, r["file"], r["audio"], {"state": STATE, "question": q, "label": int(gender == "female")}


VOCAL = {
    "Laughter": ("laughter", "笑声"),
    "Sigh": ("a sigh", "叹气"),
    "Cough": ("a cough", "咳嗽"),
    "Throat clearing": ("throat clearing", "清嗓子"),
    "Sneeze": ("a sneeze", "打喷嚏"),
    "Sniff": ("a sniff", "吸鼻子"),
}
VOCAL_ASK = ["What sound does the person make?", "Which vocal sound is this?", "What is the person doing?"]
VOCAL_ASK_ZH = ["这个人发出的是什么声音？", "这是哪种人声？"]
VOCAL_IS = ["Is this {}?", "Does the clip contain {}?"]


def vocalsound(src, rng):
    """Non-speech vocal sounds; its val and test speakers do not overlap, so val trains."""
    for path in sorted(glob.glob(str(Path(src) / "**/*.parquet"), recursive=True)):
        split = "test" if Path(path).name.startswith("test") else "train"
        for i, r in enumerate(pq.read_table(path).to_pylist()):
            if r["answer"] not in VOCAL:
                continue
            key = f"{Path(path).stem}:{i}"
            zh = rng.random() < 0.3
            names = {k: v[1] if zh else v[0] for k, v in VOCAL.items()}
            q, y = choice(rng, names[r["answer"]], list(names.values()), VOCAL_ASK_ZH if zh else VOCAL_ASK, (2, 6))
            yield split, key, r["audio"], {"state": STATE, "question": q, "label": y}
            other = r["answer"] if rng.random() < 0.5 else rng.choice([k for k in VOCAL if k != r["answer"]])
            q = {"type": "noul", "instructions": rng.choice(VOCAL_IS).format(VOCAL[other][0])}
            yield split, key, r["audio"], {"state": STATE, "question": q, "label": int(other == r["answer"])}
            if r["spk_id"][:1] in "fm" and rng.random() < 0.3:
                labels = ["man", "woman"] if rng.random() < 0.5 else ["男性", "女性"]
                q = {"type": "choice", "instructions": rng.choice(GENDER_ASK), "criteria": labels}
                yield split, key, r["audio"], {"state": STATE, "question": q, "label": int(r["spk_id"][0] == "f")}


GENRES_ZH = {
    "blues": "蓝调", "classical": "古典", "country": "乡村", "disco": "迪斯科", "hiphop": "嘻哈",
    "jazz": "爵士", "metal": "金属", "pop": "流行", "reggae": "雷鬼", "rock": "摇滚",
}
GENRE_ASK = ["What genre is this music?", "Which style of music is playing?", "What kind of music is this?"]
GENRE_ASK_ZH = ["这段音乐是什么风格？", "这是哪种类型的音乐？"]
GENRE_IS = ["Is this {} music?", "Does this sound like {}?"]


def gtzan(src, rng):
    """GTZAN genres from its tarball; the last 20 tracks of each genre are the test split."""
    import tarfile

    # Stream the tarball in its own order: a .tar.gz has no random access, so
    # sorting members first would decompress from the start for every file.
    with tarfile.open(Path(src) / "data" / "genres.tar.gz", mode="r|gz") as tar:
        for member in tar:
            m = re.search(r"(\w+)\.(\d{5})\.(wav|au)$", member.name)
            if not member.isfile() or not m or m.group(1) not in GENRES_ZH:
                continue
            genre, index = m.group(1), int(m.group(2))
            cell = {"bytes": tar.extractfile(member).read()}
            split = "test" if index >= 80 else "train"
            zh = rng.random() < 0.3
            names = GENRES_ZH if zh else {g: "hip hop" if g == "hiphop" else g for g in GENRES_ZH}
            q, y = choice(rng, names[genre], list(names.values()), GENRE_ASK_ZH if zh else GENRE_ASK, (2, 5))
            yield split, member.name, cell, {"state": STATE, "question": q, "label": y}
            other = genre if rng.random() < 0.5 else rng.choice([g for g in GENRES_ZH if g != genre])
            q = {"type": "noul", "instructions": rng.choice(GENRE_IS).format("hip hop" if other == "hiphop" else other)}
            yield split, member.name, cell, {"state": STATE, "question": q, "label": int(other == genre)}


CONVERTERS = {"esc50": esc50, "cremad": cremad, "vocalsound": vocalsound, "gtzan": gtzan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=CONVERTERS)
    ap.add_argument("src")
    ap.add_argument("--out", default="data/audio")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / args.name).mkdir(parents=True, exist_ok=True)
    saved, counts, bad = {}, Counter(), set()
    with open(out / f"{args.name}.jsonl", "w") as f:
        for split, key, cell, row in CONVERTERS[args.name](args.src, rng):
            if key in bad:
                continue
            if key not in saved:  # one file per clip, however many questions ask about it
                try:
                    wav = decode(cell)
                except Exception as e:  # e.g. GTZAN's corrupt jazz.00054.wav
                    print(f"skipping undecodable clip {key}: {e}")
                    bad.add(key)
                    continue
                path = out / args.name / f"{len(saved):06d}.wav"
                sf.write(path, wav, SR)
                saved[key] = str(path.relative_to(out))
            row.update(audio=saved[key], split=split)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[(split, row["question"]["type"])] += 1
    print(args.name, len(saved), "clips;", dict(counts))


if __name__ == "__main__":
    main()
