"""Location, spatial-relation, object and size questions from COCO boxes.

The standard benchmarks showed laya-omni's largest gaps on where things are
(SEED instance location / spatial relation, MME position) and which objects are
there (instance identity / attributes). COCO's instance boxes answer such questions
exactly, so no teacher or annotator is needed:

    python scripts/make_spatial.py --annotations instances_train2017.json \\
        --images train2017 --out data/coco_spatial

Writes coco_location / coco_relation / coco_identity / coco_size jsonl files whose
image paths are relative to --out (link or extract the images under it). Only
non-crowd boxes of at least --min-area of the image are asked about, and only
categories with a single instance in the image are referred to by name.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

GRID = [["top left", "top", "top right"], ["left", "center", "right"], ["bottom left", "bottom", "bottom right"]]
WHERE = ["Where is the {} in the image?", "Where in the picture is the {}?", "In which part of the image is the {}?"]
LEFT_RIGHT = ["Is the {a} to the left or to the right of the {b}?", "Where is the {a} relative to the {b}?"]
IS_LEFT = ["Is the {a} to the left of the {b}?", "Is the {a} on the left side of the {b}?"]
IS_ABOVE = ["Is the {a} above the {b}?", "Is the {a} higher up in the image than the {b}?"]
WHICH_SEEN = ["Which of these can be seen in the image?", "Which object appears in this picture?"]
WHICH_NOT = ["Which of these is not in the image?", "Which object does not appear in this picture?"]
BIGGER = ["Which looks bigger in the image, the {a} or the {b}?", "Which takes up more of the picture?"]


def cell(box, w, h):
    x, y, bw, bh = box
    cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
    return GRID[min(2, int(cy * 3))][min(2, int(cx * 3))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--images", required=True, help="image folder, relative to --out in the written rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-area", type=float, default=0.01, help="smallest box asked about, as a fraction of the image")
    ap.add_argument("--test-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-answer", type=int, default=10000,
                    help="location questions kept per answer (most objects sit in the center)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    coco = json.load(open(args.annotations))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    supercat = {c["name"]: c["supercategory"] for c in coco["categories"]}
    by_super = defaultdict(list)
    for n, s in supercat.items():
        by_super[s].append(n)
    images = {im["id"]: im for im in coco["images"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    streams = {k: open(out / f"coco_{k}.jsonl", "w") for k in ("location", "relation", "identity", "size")}
    counts, held = defaultdict(int), []
    rel = os.path.relpath(args.images, out)

    def write(kind, img, q, options, label, split):
        if q["type"] == "choice":
            order = list(range(len(options)))
            rng.shuffle(order)
            q = dict(q, criteria=[options[i] for i in order])
            label = order.index(label)
        row = {"state": "Image.", "question": q, "label": label, "image": f"{rel}/{img['file_name']}", "split": split}
        if kind == "location":  # balanced over answers at the end
            held.append(row)
            return
        streams[kind].write(json.dumps(row) + "\n")
        counts[kind] += 1

    for image_id, im in images.items():
        w, h = im["width"], im["height"]
        boxes = [a for a in anns.get(image_id, []) if not a.get("iscrowd")]
        if not boxes:
            continue
        split = "test" if rng.random() < args.test_fraction else "train"
        per_cat = defaultdict(list)
        for a in boxes:
            per_cat[names[a["category_id"]]].append(a)
        present = set(per_cat)
        # categories with one clearly visible instance can be named unambiguously
        single = {n: a[0] for n, a in per_cat.items() if len(a) == 1 and a[0]["area"] >= args.min_area * w * h}

        for name, a in single.items():  # where is it
            right = cell(a["bbox"], w, h)
            wrong = rng.sample([c for row in GRID for c in row if c != right], 3)
            write("location", im, {"type": "choice", "instructions": rng.choice(WHERE).format(name)}, [right] + wrong, 0, split)

        pairs = [(p, q) for p in single for q in single if p < q]
        rng.shuffle(pairs)
        for p, q in pairs[:3]:  # left / right, above / below
            if rng.random() < 0.5:
                p, q = q, p
            (px, py, pw, ph), (qx, qy, qw, qh) = single[p]["bbox"], single[q]["bbox"]
            dx = ((px + pw / 2) - (qx + qw / 2)) / w
            dy = ((py + ph / 2) - (qy + qh / 2)) / h
            if abs(dx) > 0.2 and (px + pw < qx or qx + qw < px):  # clearly apart horizontally
                if rng.random() < 0.5:
                    q_ = {"type": "choice", "instructions": rng.choice(LEFT_RIGHT).format(a=p, b=q)}
                    write("relation", im, q_, ["to the left", "to the right"], 0 if dx < 0 else 1, split)
                else:
                    write("relation", im, {"type": "noul", "instructions": rng.choice(IS_LEFT).format(a=p, b=q)}, None, int(dx < 0), split)
            elif abs(dy) > 0.2 and (py + ph < qy or qy + qh < py):  # clearly apart vertically
                write("relation", im, {"type": "noul", "instructions": rng.choice(IS_ABOVE).format(a=p, b=q)}, None, int(dy < 0), split)
            ratio = single[p]["area"] / max(1.0, single[q]["area"])
            if ratio > 2 or ratio < 0.5:  # clearly different sizes
                write("size", im, {"type": "choice", "instructions": rng.choice(BIGGER).format(a=p, b=q)}, [p, q], 0 if ratio > 1 else 1, split)

        visible = [n for n, a in per_cat.items() if max(x["area"] for x in a) >= args.min_area * w * h]
        absent = [n for n in names.values() if n not in present]
        if visible and len(absent) >= 3:
            seen = rng.choice(visible)
            near = [n for n in by_super[supercat[seen]] if n not in present]  # hard negatives first
            pool = near + [n for n in absent if n not in near]
            wrong = rng.sample(near, min(3, len(near))) if len(near) >= 3 else (near + rng.sample([n for n in absent if n not in near], 3 - len(near)))
            write("identity", im, {"type": "choice", "instructions": rng.choice(WHICH_SEEN)}, [seen] + wrong, 0, split)
            if len(visible) >= 3 and pool:
                shown = rng.sample(visible, 3)
                missing = rng.choice(pool[: max(3, len(near))])
                write("identity", im, {"type": "choice", "instructions": rng.choice(WHICH_NOT)}, [missing] + shown, 0, split)

    by_answer = defaultdict(list)
    for row in held:
        by_answer[row["question"]["criteria"][row["label"]]].append(row)
    for rows in by_answer.values():
        rng.shuffle(rows)
        for row in rows[: args.max_per_answer]:
            streams["location"].write(json.dumps(row) + "\n")
            counts["location"] += 1
    for s in streams.values():
        s.close()
    print(dict(counts))


if __name__ == "__main__":
    main()
