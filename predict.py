"""Write prediction PNGs (uint8 class ids) + submission.zip for the leaderboard.

  python predict.py --data_root data/FMB_prepared --ckpt runs/fmb_sum/best.pth --split test
"""
import argparse
import os
import zipfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from fusion_seg.data import SegDataset
from fusion_seg.model import FusionSegformer, upsample_logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--out", default="submission")
    p.add_argument("--flip", action="store_true", help="horizontal-flip TTA")
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev)
    args, meta = ck["args"], ck["meta"]
    model = FusionSegformer(meta["num_classes"], meta["aux_channels"], args["fusion"],
                            args["backbone"], pretrained=False).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    ds = SegDataset(a.data_root, a.split, (args["height"], args["width"]), train=False)
    dl = DataLoader(ds, 4, shuffle=False, num_workers=2)
    os.makedirs(a.out, exist_ok=True)
    with torch.no_grad():
        for b in dl:
            rgb, aux = b["rgb"].to(dev), b["aux"].to(dev)
            lg = model(rgb, aux).float().softmax(1)
            if a.flip:
                lg = lg + model(rgb.flip(-1), aux.flip(-1)).float().softmax(1).flip(-1)
            for i, name in enumerate(b["name"]):
                h, w = [int(v) for v in b["orig"][i]]
                pred = upsample_logits(lg[i:i + 1], (h, w)).argmax(1)[0].cpu().numpy().astype(np.uint8)
                Image.fromarray(pred).save(os.path.join(a.out, name))
    zpath = a.out.rstrip("/") + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(a.out)):
            z.write(os.path.join(a.out, f), f)
    print(f"{len(os.listdir(a.out))} files -> {zpath}")


if __name__ == "__main__":
    main()
