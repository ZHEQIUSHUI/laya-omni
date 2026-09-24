"""Convert HuggingFaceM4/the_cauldron subsets into laya-omni typed questions.

The Cauldron gathers 50 vision datasets as free-form chat turns. Each turn is
turned into a Laya question the image answers:

- lettered choices ("Choices:\\nA. ...", "Answer: B")    -> choice
- listed options ("Options: a, b, c.")                  -> choice
- yes / no answers                                      -> noul
- numbers                                               -> choice among nearby numbers
- other short answers                                   -> choice, distractors from answers
                                                           to similar questions in the subset
- long descriptions (captions, screen summaries)       -> noul: does this description fit?
                                                           (negatives: another image's)

Background text that comes with a question (a lecture, a context passage) goes
in the state, so the answer can need both. One jsonl per subset, so whole
subsets can be held out; within a subset, a twentieth of the images (by
content hash) are the test split.

    python scripts/convert_cauldron.py datasets/the_cauldron --out data/cauldron --subsets ai2d,vsr
"""

import argparse
import glob
import hashlib
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

# Documents, charts and screens keep more pixels: their answers are in small text.
DETAILED = {"docvqa", "infographic_vqa", "ocrvqa", "st_vqa", "textvqa", "chartqa", "screen2words", "figureqa", "mapqa", "textcaps"}
# A line that tells the model how to format its answer, and everything after it
# (some hints span lines: "...then justify: 'Answer: answer\nRationale: rationale.'").
FORMAT_HINT = re.compile(
    r"^\W*(answer|respond|reply|keep it|be (brief|succinct|concise)|short answer|concise|brief|quick|"
    r"write|use|pick|select|choose|make your selection|give|provide|offer|state|answer the question|"
    r"a (short|brief|very short)|one word|single word|very brief|with a)\b",
    re.I,
)
# Past the first line, a line that talks about choices, options or the answer is
# about the format too ("From the following set of four choices, select ...").
FORMAT_WORDS = re.compile(r"\b(choices?|options?|answers?|rationale|respon(se|d)|explain)\b", re.I)
LETTER = re.compile(r"^([A-Z])\.\s*(.+)$")
MAX_OPTIONS = 6
STATE = "Image."
DESCRIBE_ASK = [
    "Does this description match the image?",
    "Is this an accurate description of the picture?",
    "这段描述符合图片吗？",
]


def strip_hints(text):
    """Drop answer-format instructions ("Answer yes or no.", "Keep it brief.") and what follows."""
    lines = text.strip().split("\n")
    for i in range(1, len(lines)):
        if FORMAT_HINT.match(lines[i].strip()) or FORMAT_WORDS.search(lines[i]):
            return "\n".join(lines[:i]).strip()
    return text.strip()


def clean_answer(text):
    text = re.sub(r"^answer:\s*", "", text.strip(), flags=re.I).split("\n")[0].strip()
    return text.rstrip(".").strip()


def split_context(question):
    """'Lecture: ...\\nQuestion: q\\nHint: passage' -> (background text, q).

    Passages go to the state: Laya's question part has a 256-token budget, the
    state gets the rest of the sequence.
    """
    background = ""
    m = re.search(r"(?:^|\n)Question:\s*", question)
    if m:
        background, question = question[: m.start()].strip(), question[m.end() :]
    m = re.search(r"\n(Hint|Context):", question)
    if m:
        background = "\n".join(x for x in (question[m.start() :].strip(), background) if x)
        question = question[: m.start()]
    return background, question.strip()


def lettered(user, answer):
    """Choices listed as 'A. text' lines, answer 'Answer: B'."""
    m = re.search(r"\nChoices:\s*\n", user)
    if not m:
        return None
    options, rest = [], user[m.end() :].split("\n")
    for line in rest:
        hit = LETTER.match(line.strip())
        if not hit:
            break
        options.append(hit.group(2).strip().rstrip("."))
    letter = re.match(r"^(?:answer:\s*)?([A-Z])\b", answer.strip(), re.I)
    if not options or not letter:
        return None
    index = ord(letter.group(1).upper()) - ord("A")
    if not 0 <= index < len(options) or len(set(options)) != len(options):
        return None
    background, question = split_context(user[: m.start()])
    return background, question, options, index


def listed(user, answer):
    """'Options: a, b, c, d.' with the answer given as text."""
    m = re.search(r"\nOptions:\s*(.+?)\.?\s*$", user, re.S)
    if not m:
        return None
    options = [o.strip() for o in m.group(1).split(",") if o.strip()]
    ans = clean_answer(answer).lower()
    lowered = [o.lower() for o in options]
    if ans not in lowered or len(set(lowered)) != len(lowered):
        return None
    question = strip_hints(user[: m.start()])
    return "", question, options, lowered.index(ans)


def number(text):
    try:
        v = float(text.replace(",", ""))
    except ValueError:
        return None
    return v if abs(v) < 1e9 else None


def number_options(v, rng):
    """The answer and 1-3 plausible wrong numbers near it."""
    integer = v == int(v)
    if integer and abs(v) <= 20:
        pool = [int(v) + d for d in range(-4, 5) if d and int(v) + d >= 0]
    elif integer and 1900 <= v <= 2100:  # a year: other years nearby, not scaled ones
        pool = [int(v) + d for d in (-5, -3, -2, -1, 1, 2, 3, 5)]
    else:
        scale = [0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0]
        pool = sorted({round(v * s) if integer else round(v * s, 2) for s in scale} - {v})
    wrong = rng.sample(pool, min(len(pool), rng.randint(1, 3)))
    fmt = lambda x: str(int(x)) if float(x) == int(x) else str(x)
    options = [fmt(x) for x in wrong] + [fmt(v)]
    rng.shuffle(options)
    return options, options.index(fmt(v))


def question_kinds(q):
    """Question types for picking plausible distractors, most specific first: the
    first four, three and two words ("what is the color", "what is the", "what is")."""
    words = re.findall(r"[a-z]+", q.lower())
    return [" ".join(words[:n]) for n in (4, 3, 2)]


def convert(subset, files, out, rng, limit=0):
    img_dir = out / subset
    img_dir.mkdir(parents=True, exist_ok=True)
    detailed = subset in DETAILED
    rows, answers_by_kind, captions = [], defaultdict(set), []
    for path in files:
        for r in pq.read_table(path).to_pylist():
            images = []
            for im in r["images"]:  # some subsets (okvqa) carry only external paths: skipped
                data = im.get("bytes") if isinstance(im, dict) else im
                if not data:
                    continue
                digest = hashlib.md5(data).hexdigest()[:16]
                dest = img_dir / f"{digest}.jpg"
                if not dest.exists():
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    img.thumbnail((1024, 1024) if detailed else (512, 512))
                    img.save(dest, quality=90)
                images.append(f"{subset}/{digest}.jpg")
            if not images:
                continue
            split = "test" if int(hashlib.md5("".join(images).encode()).hexdigest(), 16) % 20 == 0 else "train"
            for turn in r["texts"]:
                rows.append((images, split, turn["user"], turn["assistant"]))
                a = clean_answer(turn["assistant"])
                if 0 < len(a.split()) <= 5:
                    for kind in question_kinds(split_context(strip_hints(turn["user"]))[1]):
                        answers_by_kind[kind].add(a)
                elif len(a.split()) > 5:
                    captions.append(a)
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break
    counts, written = Counter(), 0
    with open(out / f"{subset}.jsonl", "w") as f:
        for images, split, user, answer in rows:
            item = convert_turn(user, answer, rng, answers_by_kind, captions)
            if item is None:
                counts["skipped"] += 1
                continue
            state, question, label = item
            row = {"state": state, "question": question, "label": label, "split": split, "source": subset}
            if len(images) == 1:
                row["image"] = images[0]
            else:
                row["images"] = images
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[f"{split}:{question['type']}"] += 1
            written += 1
    print(subset, written, "questions", dict(counts))


def convert_turn(user, answer, rng, answers_by_kind, captions):
    parsed = lettered(user, answer) or listed(user, answer)
    if parsed:
        background, question, options, index = parsed
        if len(options) > MAX_OPTIONS:
            keep = rng.sample([i for i in range(len(options)) if i != index], MAX_OPTIONS - 1) + [index]
            keep.sort()
            options, index = [options[i] for i in keep], keep.index(index)
        if [o.lower() for o in options] in (["yes", "no"], ["no", "yes"]):
            return background or STATE, {"type": "noul", "instructions": question}, int(options[index].lower() == "yes")
        return background or STATE, {"type": "choice", "instructions": question, "criteria": options}, index
    background, question = split_context(strip_hints(user))
    state = background or STATE
    a = clean_answer(answer)
    if not question or not a:
        return None
    if a.lower() in ("yes", "no"):
        return state, {"type": "noul", "instructions": question}, int(a.lower() == "yes")
    words = len(a.split())
    if words > 5:  # a description: does it fit this image or belong to another one?
        if not captions:
            return None
        text = a if rng.random() < 0.5 else rng.choice(captions)
        q = {"type": "noul", "instructions": f"{rng.choice(DESCRIBE_ASK)} \"{text}\""}
        return state, q, int(text == a)
    v = number(a)
    if v is not None:
        options, index = number_options(v, rng)
        if len(options) < 2:
            return None
        return state, {"type": "choice", "instructions": question, "criteria": options}, index
    pool = []
    for kind in question_kinds(question):
        pool = sorted(answers_by_kind.get(kind, set()) - {a})
        if len(pool) >= 3:
            break
    if len(pool) < 3:
        return None
    options = rng.sample(pool, rng.randint(1, 3)) + [a]
    rng.shuffle(options)
    return state, {"type": "choice", "instructions": question, "criteria": options}, options.index(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="the_cauldron download directory")
    ap.add_argument("--out", default="data/cauldron")
    ap.add_argument("--subsets", default="", help="comma list; default: every downloaded subset")
    ap.add_argument("--limit", type=int, default=0, help="max turns per subset (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    subsets = args.subsets.split(",") if args.subsets else sorted(p.name for p in src.iterdir() if p.is_dir() and not p.name.startswith("."))
    for subset in subsets:
        files = sorted(glob.glob(str(src / subset / "*.parquet")))
        if not files:
            print(subset, "not downloaded")
            continue
        convert(subset, files, out, random.Random(f"{args.seed}:{subset}"), args.limit)


if __name__ == "__main__":
    main()
