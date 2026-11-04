#!/usr/bin/env python3
"""Leaderboard scorer (Codabench scoring-program draft) for both tracks.

usage: python scoring.py <input_dir> <output_dir>
  <input_dir>/ref/*.png  ground truth (uint8 class ids, 255 = ignore) + ref/meta.json
  <input_dir>/res/**.png predictions, same file names (sub-folders tolerated)
writes <output_dir>/scores.json  (keys = leaderboard column keys)

ref/meta.json   {"num_classes": 14, "ignore": 255, "group_by_prefix": false}
  group_by_prefix=true (DELIVER track): file name '<condition>__<seq>__<id>.png'
  -> extra columns per condition + adverse_mIoU (mean of cloud, fog, night, rain).

mIoU = mean over classes (present in GT) of IoU computed from a confusion matrix
accumulated over ALL images (not a mean of per-image scores). Missing prediction
files count as completely wrong (their GT pixels become false negatives).
"""
import json
import os
import sys

import numpy as np
from PIL import Image

ADVERSE = ("cloud", "fog", "night", "rain")


def load(path):
    return np.array(Image.open(path)).astype(np.int64)


def collect(res_dir):
    out = {}
    for root, _, files in os.walk(res_dir):
        for f in files:
            if f.lower().endswith(".png") and not f.startswith("."):
                out[f] = os.path.join(root, f)
    return out


class Acc:
    def __init__(self, n):
        self.n = n
        self.conf = np.zeros((n, n), dtype=np.int64)
        self.fn = np.zeros(n, dtype=np.int64)

    def miou(self):
        tp = np.diag(self.conf).astype(float)
        gt_tot = self.conf.sum(1) + self.fn
        union = self.conf.sum(0) + gt_tot - tp
        present = gt_tot > 0
        iou = tp / np.maximum(union, 1)
        return float(iou[present].mean() * 100) if present.any() else 0.0

    def pix_acc(self):
        tot = self.conf.sum() + self.fn.sum()
        return float(np.diag(self.conf).sum() / max(tot, 1) * 100)


def main(inp, out):
    ref_dir, res_dir = os.path.join(inp, "ref"), os.path.join(inp, "res")
    os.makedirs(out, exist_ok=True)
    meta = {"num_classes": 14, "ignore": 255, "group_by_prefix": False}
    mp = os.path.join(ref_dir, "meta.json")
    if os.path.exists(mp):
        meta.update(json.load(open(mp)))
    n, ign = meta["num_classes"], meta["ignore"]
    preds = collect(res_dir)
    names = sorted(f for f in os.listdir(ref_dir) if f.endswith(".png"))
    total, groups, missing = Acc(n), {}, 0
    for nm in names:
        gt = load(os.path.join(ref_dir, nm))
        valid = (gt != ign) & (gt >= 0) & (gt < n)
        accs = [total]
        if meta.get("group_by_prefix"):
            accs.append(groups.setdefault(nm.split("__")[0], Acc(n)))
        if nm not in preds:
            missing += 1
            for acc in accs:
                acc.fn += np.bincount(gt[valid], minlength=n)
            continue
        pr = load(preds[nm])
        if pr.shape != gt.shape:
            raise ValueError(f"{nm}: prediction shape {pr.shape} != ground truth {gt.shape}")
        pr = np.clip(pr[valid], 0, n - 1)
        cm = np.bincount(gt[valid] * n + pr, minlength=n * n).reshape(n, n)
        for acc in accs:
            acc.conf += cm
    scores = {"mIoU": round(total.miou(), 3), "pixAcc": round(total.pix_acc(), 3), "missing": missing}
    if groups:
        for g, acc in sorted(groups.items()):
            scores[f"mIoU_{g}"] = round(acc.miou(), 3)
        adv = [groups[g].miou() for g in ADVERSE if g in groups]
        scores["adverse_mIoU"] = round(float(np.mean(adv)), 3) if adv else 0.0
    json.dump(scores, open(os.path.join(out, "scores.json"), "w"))
    print(scores)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
