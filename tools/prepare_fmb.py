"""Scan the downloaded FMB folder, build train/val/test indices (no data is copied).

The authors' folder layout is not documented in the repo README, so this script
detects it from folder names:
  RGB   : path contains visible / rgb / vis   (NOT "color": in FMB that folder holds colorised label maps)
  IR    : path contains infrared / thermal / ir
  label : path contains label / mask / gt / semantic (visualisation dirs skipped)
  split : path contains 'train' or 'test'
Files are matched by file stem. If detection fails the script prints what it saw.

split rule (fixed seed => identical for every student):
  official train 1,220 -> shuffle(seed 2026) -> train 1,000 / val rest
  official test  280   -> leaderboard test (use for final prediction only!)

Instructor-only: --export_ref DIR writes test labels (mapped to class ids) for the scorer.
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

EXT = (".png", ".jpg", ".jpeg", ".bmp")
RGB_KEYS, IR_KEYS, LAB_KEYS = ("visible", "rgb", "vis"), ("infrared", "thermal", "ir"), ("label", "mask", "gt", "semantic")
SKIP_LAB = ("vis", "color", "rgb", "viz", "show")


def classify(path_parts):
    low = [p.lower() for p in path_parts]
    dirs = low[:-1]
    split = "train" if any("train" in d for d in dirs) else "test" if any("test" in d for d in dirs) else None

    def has(keys):
        return any(any(k in d for k in keys) for d in dirs)

    if has(LAB_KEYS):
        lab_dirs = [d for d in dirs if any(k in d for k in LAB_KEYS)]
        return split, (None if any(any(s in d for s in SKIP_LAB) for d in lab_dirs) else "label")
    if has(("infrared", "thermal")) or "ir" in dirs:
        return split, "ir"
    if has(RGB_KEYS):
        return split, "rgb"
    return split, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", default="data/FMB_prepared")
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--num_classes", type=int, default=14)
    p.add_argument("--ignore", type=int, default=255)
    p.add_argument("--label_shift", type=int, default=1, help="class id = pixel value - label_shift (FMB: 0=unlabeled, 1..14 -> 0..13)")
    p.add_argument("--export_ref", default=None, help="(instructor) dir for hidden test labels")
    a = p.parse_args()

    found = defaultdict(dict)  # (split, stem) -> {kind: relpath}
    seen_dirs = Counter()
    for root, _, files in os.walk(a.raw):
        for f in files:
            if not f.lower().endswith(EXT):
                continue
            rel = os.path.relpath(os.path.join(root, f), a.raw)
            split, kind = classify(rel.split(os.sep))
            seen_dirs[os.path.dirname(rel)] += 1
            if split and kind:
                found[(split, os.path.splitext(f)[0])][kind] = rel
    triples = {s: sorted((k[1], v) for k, v in found.items() if k[0] == s and {"rgb", "ir", "label"} <= v.keys())
               for s in ("train", "test")}
    print({s: len(v) for s, v in triples.items()})
    if not triples["train"] or not triples["test"]:
        print("!! could not detect the layout. folders seen (path: #images):")
        for d, c in seen_dirs.most_common(30):
            print("  ", d, c)
        raise SystemExit("edit RGB_KEYS / IR_KEYS / LAB_KEYS in this script (or ask the instructor)")

    os.makedirs(a.out, exist_ok=True)
    tr = triples["train"][:]
    random.Random(2026).shuffle(tr)
    parts = {"train": sorted(tr[:a.n_train]), "val": sorted(tr[a.n_train:]), "test": triples["test"]}
    for split, lst in parts.items():
        idx = []
        for stem, v in lst:
            idx.append({"name": stem + ".png", "rgb": os.path.relpath(os.path.join(a.raw, v["rgb"]), a.out),
                        "aux": os.path.relpath(os.path.join(a.raw, v["ir"]), a.out),
                        "label": None if split == "test" else os.path.relpath(os.path.join(a.raw, v["label"]), a.out)})
        json.dump(idx, open(os.path.join(a.out, f"index_{split}.json"), "w"))
    aux0 = np.array(Image.open(os.path.join(a.raw, triples["train"][0][1]["ir"])))
    meta = {"dataset": "FMB", "num_classes": a.num_classes, "ignore": a.ignore, "label_shift": a.label_shift,
            "aux_channels": 1 if aux0.ndim == 2 else min(aux0.shape[2], 3)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"))

    # label histogram -> verify class-id convention before training!
    hist = Counter()
    for _, v in tr[:60]:
        vals, cnt = np.unique(np.array(Image.open(os.path.join(a.raw, v["label"]))), return_counts=True)
        hist.update(dict(zip(vals.tolist(), cnt.tolist())))
    print("label values (60 train images):", sorted(hist.items()))
    print("meta:", meta, "-> FMB: 0=unlabeled, 1..14 = classes (label_shift 1 is the default); if your values differ, change --label_shift")
    if a.export_ref:
        os.makedirs(a.export_ref, exist_ok=True)
        for stem, v in triples["test"]:
            lab = np.array(Image.open(os.path.join(a.raw, v["label"]))).astype(np.int64) - a.label_shift
            lab[(lab < 0) | (lab >= a.num_classes)] = a.ignore
            Image.fromarray(lab.astype(np.uint8)).save(os.path.join(a.export_ref, stem + ".png"))
        json.dump({"num_classes": a.num_classes, "ignore": a.ignore}, open(os.path.join(a.export_ref, "meta.json"), "w"))
        print("hidden-test reference written:", a.export_ref)


if __name__ == "__main__":
    main()
