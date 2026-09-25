"""Convert public audio datasets into laya-omni rows.

Each dataset becomes <out>/<name>.jsonl plus 16 kHz mono FLAC (or WAV) files under
<out>/<name>/. Rows look like {"state", "question", "label", "audio", "split"};
see laya_omni/data.py. Questions vary in phrasing and language, choice options
are a random subset around the answer, and noul questions are balanced.

    python scripts/convert_audio.py esc50 datasets/esc50 --out data/audio
    python scripts/convert_audio.py cremad datasets/CREMA-D --out data/audio
    python scripts/convert_audio.py vocalsound datasets/vocalsound --out data/audio
    python scripts/convert_audio.py gtzan datasets/gtzan --out data/audio
    python scripts/convert_audio.py clotho datasets/Clotho --out data/audio
    python scripts/convert_audio.py audiocaps datasets/audiocaps --out data/audio
    python scripts/convert_audio.py clothoaqa datasets/ClothoAQA --out data/audio
    python scripts/convert_audio.py musicbench datasets/MusicBench --out data/audio
    python scripts/convert_audio.py minds14 datasets/minds14 --out data/audio
"""

import argparse
import glob
import hashlib
import io
import json
import random
import re
import zlib
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


MATCH_ASK = [
    "Does this description match the sound?",
    "Is this an accurate description of the audio?",
    "Does the clip sound like this?",
    "这段描述和声音对得上吗？",
]
PICK_ASK = ["Which description fits the sound best?", "What is happening in this audio?", "哪一条描述最符合这段声音？"]
STOP = set("a an the and or of to in on at by for with from into onto is are was were be being been it its this that "
           "there their then than while as some someone something people person sound sounds noise noises".split())


def content_words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3 and w not in STOP}


class CaptionPool:
    """Captions of all clips in a split, for negatives: half random, half sharing a
    content word with the true caption (water vs. rain), so matching needs listening."""

    def __init__(self, captions_by_clip):
        self.by_clip = captions_by_clip
        self.clips = sorted(captions_by_clip)
        self.by_word = {}
        for clip, caps in captions_by_clip.items():
            for c in caps:
                for w in content_words(c):
                    self.by_word.setdefault(w, []).append((clip, c))

    def negative(self, clip, caption, rng):
        if rng.random() < 0.5:
            words = sorted(content_words(caption))
            rng.shuffle(words)
            for w in words:
                others = [c for k, c in self.by_word.get(w, ()) if k != clip]
                if others:
                    return rng.choice(others)
        other = clip
        while other == clip:
            other = rng.choice(self.clips)
        return rng.choice(self.by_clip[other])


def caption_questions(clip, caption, pool, rng):
    """One match-or-not noul and one pick-the-description choice."""
    text = caption if rng.random() < 0.5 else pool.negative(clip, caption, rng)
    yield {"type": "noul", "instructions": f"{rng.choice(MATCH_ASK)} \"{text}\""}, int(text == caption)
    options = [pool.negative(clip, caption, rng) for _ in range(rng.randint(1, 3))]
    options = list(dict.fromkeys(o for o in options if o != caption)) + [caption]
    if len(options) >= 2:
        rng.shuffle(options)
        yield {"type": "choice", "instructions": rng.choice(PICK_ASK), "criteria": options}, options.index(caption)


def split_captions(text):
    """Clotho stores a clip's five captions run together, sometimes without a full
    stop between them: split at sentence ends and at a lowercase-to-Capital join."""
    parts = re.split(r"(?<=[.!?])\s+|(?<=[a-z])\s+(?=[A-Z])", text.strip())
    return [p.strip().rstrip(".") for p in parts if len(p.split()) >= 4]


def clotho(src, rng):
    """Clotho captions; its development and validation sets train, evaluation tests."""
    splits = {"train": "train", "valid": "train", "test": "test"}
    meta = {}
    for d, split in splits.items():
        for path in sorted(glob.glob(str(Path(src) / "data" / d / "*.parquet"))):
            for r in pq.read_table(path, columns=["index", "text"]).to_pylist():
                caps = split_captions(r["text"])
                if caps:
                    meta[r["index"]] = (split, caps)
    pools = {s: CaptionPool({k: c for k, (sp, c) in meta.items() if sp == s}) for s in ("train", "test")}
    for d, split in splits.items():
        for path in sorted(glob.glob(str(Path(src) / "data" / d / "*.parquet"))):
            for r in pq.read_table(path, columns=["index", "audio"]).to_pylist():
                if r["index"] not in meta:
                    continue
                caps = meta[r["index"]][1]
                for caption in rng.sample(caps, min(2, len(caps))):
                    for q, y in caption_questions(r["index"], caption, pools[split], rng):
                        yield split, r["index"], r["audio"], {"state": STATE, "question": q, "label": y}


def audiocaps(src, rng):
    """AudioCaps; train and validation clips train, test clips test."""
    files = {"train": "train", "validation": "train", "test": "test"}
    meta = {}
    for prefix, split in files.items():
        for path in sorted(glob.glob(str(Path(src) / "data" / f"{prefix}-*.parquet"))):
            for r in pq.read_table(path, columns=["youtube_id", "caption"]).to_pylist():
                meta.setdefault(r["youtube_id"], (split, []))[1].append(r["caption"].strip().rstrip("."))
    pools = {s: CaptionPool({k: c for k, (sp, c) in meta.items() if sp == s}) for s in ("train", "test")}
    seen = set()
    for prefix, split in files.items():
        for path in sorted(glob.glob(str(Path(src) / "data" / f"{prefix}-*.parquet"))):
            for r in pq.read_table(path, columns=["youtube_id", "audio"]).to_pylist():
                clip = r["youtube_id"]
                if clip in seen:  # val/test clips repeat once per caption; ask about each clip once
                    continue
                seen.add(clip)
                caption = rng.choice(meta[clip][1])
                for q, y in caption_questions(clip, caption, pools[split], rng):
                    yield split, clip, r["audio"], {"state": STATE, "question": q, "label": y}


def clothoaqa(src, rng):
    """Clotho-AQA yes/no questions about sounds (answers that start with yes/no count);
    its validation clips train and its test clips test."""
    for path in sorted(glob.glob(str(Path(src) / "clotho_aqa" / "*.parquet"))):
        split = "test" if "_test_" in Path(path).name else "train"
        for i, r in enumerate(pq.read_table(path).to_pylist()):
            answer = (r.get("answer") or "").strip().lower()
            word = re.match(r"^(yes|no)\b", answer)
            if not word or not r.get("question"):
                continue
            # Each clip is asked several questions, one row each: key clips by content.
            key = hashlib.md5(r["audio"]["bytes"]).hexdigest()
            q = {"type": "noul", "instructions": r["question"].strip()}
            yield split, key, r["audio"], {"state": STATE, "question": q, "label": int(word.group(1) == "yes")}


KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
TEMPO = [("slow", "慢"), ("moderate", "中等"), ("fast", "快")]


def music_questions(r, pool, clip, rng):
    """Questions about one MusicBench clip: its description, tempo, mode, key and metre."""
    cap = (r.get("alt_caption") or "").strip()
    if cap:  # alt_caption carries no "the key is ..." control sentences that would give answers away
        for q, y in caption_questions(clip, cap, pool, rng):
            yield q, y
    bpm = r.get("bpm")
    if bpm and min(abs(bpm - 80), abs(bpm - 120)) > 5:  # skip clips near a tempo boundary
        zh = rng.random() < 0.3
        options = [t[1] if zh else t[0] for t in TEMPO]
        answer = options[0 if bpm < 80 else 1 if bpm < 120 else 2]
        ask = "这段音乐的节奏是快还是慢？" if zh else rng.choice(["How fast is this music?", "What is the tempo of this piece?"])
        yield {"type": "choice", "instructions": ask, "criteria": options}, options.index(answer)
    key, prob = r.get("key") or [], (r.get("keyprob") or [0])[0]
    if len(key) == 2 and prob > 0.6:
        root, mode = key
        yield {"type": "choice", "instructions": rng.choice(["Is this in a major or a minor key?", "Does the music sound major or minor?"]),
               "criteria": ["major", "minor"]}, int(mode == "minor")
        if prob > 0.7 and root in KEYS:
            others = rng.sample([k for k in KEYS if k != root], 3)
            options = others + [root]
            rng.shuffle(options)
            yield {"type": "choice", "instructions": "What key is this music in?", "criteria": options}, options.index(root)
    beats = r.get("beats") or []  # [beat times, beat numbers within the bar]
    meter = int(max(beats[1])) if len(beats) > 1 and beats[1] else None
    if meter in (3, 4):  # 2 vs 4 is too ambiguous to hear
        yield {"type": "choice", "instructions": "Does the beat count to three or to four?", "criteria": ["three", "four"]}, int(meter == 4)


def musicbench(src, rng):
    """MusicBench: MusicCaps clips with pitch/tempo-shifted copies and extracted tempo,
    key, chords and beats. Test_B's clips test; every training copy of a test clip is dropped."""
    import tarfile

    src = Path(src)
    train = [json.loads(l) for l in open(src / "MusicBench_train.json")]
    test = [json.loads(l) for l in open(src / "MusicBench_test_B.json")]
    base = lambda loc: re.sub(r"_\d+$", "", Path(loc).stem)  # augmented copies end in _1, _2, ...
    held = {base(r["location"]) for r in test}
    rows = {r["location"]: ("test", r) for r in test}
    rows.update({r["location"]: ("train", r) for r in train if base(r["location"]) not in held})
    pools = {s: CaptionPool({loc: [r["alt_caption"]] for loc, (sp, r) in rows.items() if sp == s and r.get("alt_caption")})
             for s in ("train", "test")}
    archives = sorted(p for p in src.glob("*.tar.gz") if "FMACaps" not in p.name)
    for archive in archives:
        with tarfile.open(archive, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                loc = "/".join(Path(member.name).parts[-2:])  # "data_aug2/<id>_1.wav"
                if loc not in rows:
                    continue
                split, r = rows[loc]
                cell = {"bytes": tar.extractfile(member).read()}
                for q, y in music_questions(r, pools[split], loc, rng):
                    yield split, loc, cell, {"state": STATE, "question": q, "label": y}


MINDS_INTENTS = [  # PolyAI/minds14 intent_class order
    ("abroad", "use the card abroad", "在国外用卡"),
    ("address", "change their address", "修改地址"),
    ("app_error", "report an app error", "反馈应用故障"),
    ("atm_limit", "ask about the ATM withdrawal limit", "询问取款限额"),
    ("balance", "check the balance", "查询余额"),
    ("business_loan", "ask about a business loan", "咨询企业贷款"),
    ("card_issues", "report a card problem", "反馈银行卡问题"),
    ("cash_deposit", "deposit cash", "存现金"),
    ("direct_debit", "set up a direct debit", "设置自动扣款"),
    ("freeze", "freeze the card", "冻结银行卡"),
    ("high_value_payment", "make a large payment", "办理大额付款"),
    ("joint_account", "open a joint account", "开联名账户"),
    ("latest_transactions", "see recent transactions", "查看最近交易"),
    ("pay_bill", "pay a bill", "缴费"),
]
INTENT_ASK = ["What does the caller want to do?", "Why is the customer calling the bank?", "来电者想办什么业务？"]
INTENT_IS = ["Does the caller want to {}?", "Is the customer calling to {}?", "来电者是想{}吗？"]


def minds14(src, rng):
    """MINDS-14 banking intents in 14 languages; a tenth of the recordings, by path, test."""
    for path in sorted(glob.glob(str(Path(src) / "*" / "*.parquet"))):
        if Path(path).parent.name == "all":  # the per-language folders hold the same rows
            continue
        for r in pq.read_table(path).to_pylist():
            intent = r.get("intent_class")
            if intent is None or not 0 <= intent < len(MINDS_INTENTS):
                continue
            key = r.get("path") or hashlib.md5(r["audio"]["bytes"]).hexdigest()
            split = "test" if zlib.crc32(str(key).encode()) % 10 == 0 else "train"
            zh = rng.random() < 0.3
            names = [x[2] if zh else x[1] for x in MINDS_INTENTS]
            q, y = choice(rng, names[intent], names, [INTENT_ASK[2]] if zh else INTENT_ASK[:2], (2, 4))
            yield split, key, r["audio"], {"state": STATE, "question": q, "label": y, "lang": r.get("lang_id")}
            other = intent if rng.random() < 0.5 else rng.choice([i for i in range(len(MINDS_INTENTS)) if i != intent])
            ask = INTENT_IS[2] if zh else rng.choice(INTENT_IS[:2])
            q = {"type": "noul", "instructions": ask.format(names[other])}
            yield split, key, r["audio"], {"state": STATE, "question": q, "label": int(other == intent)}


def similar_labels(label, labels, rng, k):
    """k wrong labels, half sharing a word with the right one ('people marching' vs
    'people running') so the name alone does not give it away."""
    words = set(label.split())
    near = [l for l in labels if l != label and words & set(l.split())]
    far = [l for l in labels if l != label and l not in near]
    out = []
    for _ in range(k):
        pool = near if near and rng.random() < 0.5 else far
        pick = rng.choice(pool)
        pool.remove(pick)
        out.append(pick)
    return out


NO_SPEECH = re.compile(r"\b(no|not contain any|does not contain|doesn't contain|without)\b[^.]*\bspeech", re.I)
NO_MUSIC = re.compile(r"\b(no|not contain any|does not contain|doesn't contain|without)\b[^.]*\bmusic", re.I)
SPEECH_ASK = ["Is anyone speaking in this clip?", "Does the audio contain speech?", "录音里有人说话吗？"]
MUSIC_ASK = ["Is there music in this clip?", "Does the audio contain music?", "录音里有音乐吗？"]


def vggsound(src, rng, audiosetcaps=None):
    """VGGSound's 310 sound classes for every clip; for training clips that
    AudioSetCaps covers, also its short description (caption matching) and whether
    the clip has speech or music. The test clips are VGGSound's own test set."""
    import csv

    qa = {}
    qa_file = Path(audiosetcaps or Path(src).parent / "AudioSetCaps") / "Dataset" / "VGGSound_Qwen-Audio_Q&A.csv"
    if qa_file.exists():
        with open(qa_file) as fh:
            for row in csv.DictReader(fh):
                qa[row["id"]] = row
    labels = sorted({x for p in glob.glob(str(Path(src) / "data" / "*.parquet"))
                     for x in pq.read_table(p, columns=["caption"]).column("caption").to_pylist()})
    pool = CaptionPool({k: [v["answer_1"]] for k, v in qa.items() if v.get("answer_1")})
    for path in sorted(glob.glob(str(Path(src) / "data" / "*.parquet"))):
        split = "test" if Path(path).name.startswith("test") else "train"
        for r in pq.read_table(path).to_pylist():
            clip = re.sub(r"\.\w+$", "", (r["audio"].get("path") or "").split("/")[-1])
            label = r["caption"]
            options = similar_labels(label, list(labels), rng, rng.randint(1, 3)) + [label]
            rng.shuffle(options)
            q = {"type": "choice", "instructions": rng.choice(SOUND_ASK), "criteria": options}
            yield split, clip, r["audio"], {"state": STATE, "question": q, "label": options.index(label)}
            other = label if rng.random() < 0.5 else similar_labels(label, list(labels), rng, 1)[0]
            q = {"type": "noul", "instructions": rng.choice(SOUND_IS).format(other)}
            yield split, clip, r["audio"], {"state": STATE, "question": q, "label": int(other == label)}
            row = qa.get(clip)
            if split != "train" or not row:
                continue
            if row.get("answer_1"):
                for q, y in caption_questions(clip, row["answer_1"].strip().rstrip("."), pool, rng):
                    yield split, clip, r["audio"], {"state": STATE, "question": q, "label": y}
            if row.get("answer_2"):
                q = {"type": "noul", "instructions": rng.choice(SPEECH_ASK)}
                yield split, clip, r["audio"], {"state": STATE, "question": q, "label": int(not NO_SPEECH.search(row["answer_2"]))}
            if row.get("answer_3"):
                q = {"type": "noul", "instructions": rng.choice(MUSIC_ASK)}
                yield split, clip, r["audio"], {"state": STATE, "question": q, "label": int(not NO_MUSIC.search(row["answer_3"]))}


def slurp(src, rng):
    """SLURP spoken commands to a home assistant, 101 intents ('alarm_set'). Wrong
    options are half from the same scenario ('alarm_query'); official splits, with
    the synthetic (TTS) recordings added to training. Clips are keyed by content,
    since each sentence was recorded several times."""
    files = sorted(glob.glob(str(Path(src) / "data" / "*.parquet")))
    meta = json.loads(pq.ParquetFile(files[0]).schema_arrow.metadata[b"huggingface"])
    names = meta["info"]["features"]["intent"]["names"]
    human = [n.replace("_", " ").replace("iot", "smart home").replace("qa", "question") for n in names]
    by_scenario = {}
    for i, n in enumerate(names):
        by_scenario.setdefault(n.split("_")[0], []).append(i)
    for path in files:
        name = Path(path).name
        split = "test" if name.startswith("test") else "train"  # train, train_synthetic and devel train
        for r in pq.read_table(path).to_pylist():
            intent = r.get("intent")
            if intent is None or not r.get("audio") or not r["audio"].get("bytes"):
                continue
            key = hashlib.md5(r["audio"]["bytes"]).hexdigest()
            same = [i for i in by_scenario[names[intent].split("_")[0]] if i != intent]
            wrong = set()
            for _ in range(rng.randint(1, 3)):
                cand = rng.choice(same) if same and rng.random() < 0.5 else rng.randrange(len(names))
                if cand != intent:
                    wrong.add(cand)
            options = [human[i] for i in wrong] + [human[intent]]
            rng.shuffle(options)
            q = {"type": "choice", "instructions": rng.choice(["What does the user want?", "Which command is this?", "用户想让助手做什么？"]),
                 "criteria": options}
            yield split, key, r["audio"], {"state": STATE, "question": q, "label": options.index(human[intent])}


def mmau(src, rng):
    """MMAU test-mini (the split with public answers): sound, speech and music
    multiple choice, as a zero-shot benchmark. Every row is a test row."""
    for path in sorted(glob.glob(str(Path(src) / "**" / "test_mini-*.parquet"), recursive=True)):
        for r in pq.read_table(path).to_pylist():
            try:
                options = json.loads(r["choices"]) if isinstance(r["choices"], str) else list(r["choices"])
            except json.JSONDecodeError:
                continue
            options = [str(o).strip() for o in options]
            if r.get("answer") not in options or len(set(options)) != len(options) or len(options) > 6:
                continue
            q = {"type": "choice", "instructions": r["question"].strip(), "criteria": options}
            row = {"state": STATE, "question": q, "label": options.index(r["answer"]), "task": r.get("task")}
            yield "test", r["id"], r["audio"], row


def songdescriber(src, rng):
    """Song Describer: full tracks with several listeners' captions, as a zero-shot
    caption-matching test (all rows test). Tracks are cut to their middle 30 s below."""
    rows = [r for p in sorted(glob.glob(str(Path(src) / "**" / "*.parquet"), recursive=True)) for r in pq.read_table(p).to_pylist()]
    by_track = {}
    for r in rows:
        by_track.setdefault(r["track_id"], []).append(r["caption"].strip().rstrip("."))
    pool = CaptionPool(by_track)
    seen = set()
    for r in rows:
        if r["track_id"] in seen:  # one audio per track; its captions are asked about once
            continue
        seen.add(r["track_id"])
        caption = rng.choice(by_track[r["track_id"]])
        for q, y in caption_questions(r["track_id"], caption, pool, rng):
            yield "test", r["track_id"], r["path"], {"state": STATE, "question": q, "label": y}


MAX_SECONDS = {"songdescriber": 30}  # long clips are cut to their middle this many seconds


def avqa(src, rng):
    """AVQA's questions about VGGSound clips, answered from the audio (this copy has no
    frames): four lettered choices. Its val split tests."""
    import pyarrow as pa

    for split_dir, split in (("train", "train"), ("val", "test")):
        for path in sorted(glob.glob(str(Path(src) / split_dir / "*.arrow"))):
            with pa.memory_map(path) as f:
                table = pa.ipc.open_stream(f).read_all()
            for r in table.to_pylist():
                m = re.match(r"(?s)(.*?)\nChoices:\n(.*)$", r["question"].strip())
                if not m:
                    continue
                options = [re.sub(r"^[A-Z]\.\s*", "", o).strip() for o in m.group(2).strip().split("\n")]
                answer = re.sub(r"^[A-Z]\.\s*", "", r["answer"].strip())
                if answer not in options or len(set(options)) != len(options):
                    continue
                q = {"type": "choice", "instructions": m.group(1).strip(), "criteria": options}
                key = f"{r['file_name']}"
                yield split, key, r["audio"], {"state": STATE, "question": q, "label": options.index(answer)}


CONVERTERS = {
    "avqa": avqa,
    "songdescriber": songdescriber,
    "mmau": mmau,
    "vggsound": vggsound,
    "slurp": slurp,
    "minds14": minds14,
    "musicbench": musicbench,
    "clothoaqa": clothoaqa,
    "esc50": esc50,
    "cremad": cremad,
    "vocalsound": vocalsound,
    "gtzan": gtzan,
    "clotho": clotho,
    "audiocaps": audiocaps,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=CONVERTERS)
    ap.add_argument("src")
    ap.add_argument("--out", default="data/audio")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=("flac", "wav"), default="flac")
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
                    if len(wav) < SR // 10:  # empty or near-empty source audio
                        raise ValueError(f"only {len(wav)} samples")
                    limit = MAX_SECONDS.get(args.name, 0) * SR
                    if limit and len(wav) > limit:  # keep the middle
                        start = (len(wav) - limit) // 2
                        wav = wav[start : start + limit]
                except Exception as e:  # e.g. GTZAN's corrupt jazz.00054.wav
                    print(f"skipping undecodable clip {key}: {e}")
                    bad.add(key)
                    continue
                path = out / args.name / f"{len(saved):06d}.{args.format}"
                sf.write(path, wav, SR, subtype="PCM_16")  # FLAC is about half the size of WAV
                saved[key] = str(path.relative_to(out))
            row.update(audio=saved[key], split=split)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[(split, row["question"]["type"])] += 1
    print(args.name, len(saved), "clips;", dict(counts))


if __name__ == "__main__":
    main()
