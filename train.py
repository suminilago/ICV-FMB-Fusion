"""Single-GPU training for the multimodal segmentation final project.

example (Colab T4):
  python train.py --data_root data/FMB_prepared --fusion sum --epochs 40 --out runs/fmb_sum
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fusion_seg.data import SegDataset, load_meta
from fusion_seg.metrics import ConfMat
from fusion_seg.model import FusionSegformer, upsample_logits


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--fusion", default="sum", help="rgb | early | sum | <your fusion>")
    p.add_argument("--backbone", default="nvidia/mit-b0")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=2, help="warm-up epochs")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_minutes", type=float, default=0, help="stop gracefully after N minutes (0 = off)")
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, meta, dev, amp):
    model.eval()
    cm = ConfMat(meta["num_classes"], meta["ignore"])
    for b in loader:
        with torch.autocast(dev.type, enabled=amp):
            logits = model(b["rgb"].to(dev), b["aux"].to(dev))
        logits = upsample_logits(logits.float(), b["label"].shape[-2:])
        cm.update(logits.argmax(1), b["label"])
    return cm.miou()


def main():
    a = get_args()
    seed_all(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (dev.type == "cuda") and not a.no_amp
    meta = load_meta(a.data_root)
    size = (a.height, a.width)
    tr = SegDataset(a.data_root, "train", size, train=True)
    va = SegDataset(a.data_root, "val", size, train=False)
    tl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    vl = DataLoader(va, 4, shuffle=False, num_workers=a.workers)

    model = FusionSegformer(meta["num_classes"], meta["aux_channels"], a.fusion,
                            a.backbone, pretrained=not a.no_pretrained).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    opt = torch.optim.AdamW(model.param_groups(a.lr), weight_decay=a.wd)
    base = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler(enabled=amp)
    total = a.epochs * len(tl)

    start_ep, best, t0 = 0, -1.0, time.time()
    last = os.path.join(a.out, "last.pth")
    if os.path.exists(last):  # resume after a Colab disconnect
        ck = torch.load(last, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_ep, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {start_ep}")
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

    for ep in range(start_ep, a.epochs):
        model.train()
        run = 0.0
        for it, b in enumerate(tl):
            step = ep * len(tl) + it
            warm = a.warmup * len(tl)
            f = (step + 1) / warm if step < warm else (1 - (step - warm) / max(total - warm, 1)) ** 0.9
            for g, bl in zip(opt.param_groups, base):
                g["lr"] = bl * f
            rgb, aux, lab = b["rgb"].to(dev), b["aux"].to(dev), b["label"].to(dev)
            with torch.autocast(dev.type, enabled=amp):
                logits = model(rgb, aux)
            logits = upsample_logits(logits.float(), lab.shape[-2:])
            loss = F.cross_entropy(logits, lab, ignore_index=meta["ignore"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.item()
        miou = evaluate(model, vl, meta, dev, amp)
        rec = {"epoch": ep, "loss": run / len(tl), "val_mIoU": miou, "min": (time.time() - t0) / 60}
        print(json.dumps(rec))
        with open(os.path.join(a.out, "log.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        if miou > best:
            best = miou
            torch.save({"model": model.state_dict(), "args": vars(a), "meta": meta}, os.path.join(a.out, "best.pth"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best}, last)
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            print("time budget reached; rerun the same command to resume")
            break
    print(f"best val mIoU: {best:.2f}")


if __name__ == "__main__":
    main()
