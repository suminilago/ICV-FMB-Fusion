"""Generic RGB + auxiliary-modality segmentation dataset.

Reads <root>/meta.json and <root>/index_<split>.json written by
tools/prepare_fmb.py or tools/prepare_deliver.py.
index entry: {"name": "...png", "rgb": rel, "aux": rel, "label": rel|null}
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_meta(root):
    with open(os.path.join(root, "meta.json")) as f:
        return json.load(f)


def _to_float_array(img):
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    scale = 65535.0 if arr.dtype == np.uint16 else 255.0
    return arr.astype(np.float32) / scale


class SegDataset(Dataset):
    def __init__(self, root, split, size=(384, 512), train=False, scale=(0.75, 1.5)):
        self.root, self.size, self.train, self.scale = root, size, train, scale
        self.meta = load_meta(root)
        with open(os.path.join(root, f"index_{split}.json")) as f:
            self.items = json.load(f)

    def __len__(self):
        return len(self.items)

    def _read(self, it):
        rgb = torch.from_numpy(_to_float_array(Image.open(os.path.join(self.root, it["rgb"])).convert("RGB"))).permute(2, 0, 1)
        aux = torch.from_numpy(_to_float_array(Image.open(os.path.join(self.root, it["aux"])))).permute(2, 0, 1)
        if aux.shape[0] == 4:
            aux = aux[:3]
        if it.get("label"):
            lab = np.array(Image.open(os.path.join(self.root, it["label"])))
            if lab.ndim == 3:
                lab = lab[..., 0]
            lab = lab.astype(np.int64)
            lab = lab - self.meta.get("label_shift", 0)
            bad = (lab < 0) | (lab >= self.meta["num_classes"])
            lab[bad] = self.meta["ignore"]
            lab = torch.from_numpy(lab)
        else:
            lab = torch.full(rgb.shape[1:], self.meta["ignore"], dtype=torch.long)
        return rgb, aux, lab

    @staticmethod
    def _resize(rgb, aux, lab, size):
        rgb = F.interpolate(rgb[None], size=size, mode="bilinear", align_corners=False)[0]
        aux = F.interpolate(aux[None], size=size, mode="bilinear", align_corners=False)[0]
        lab = F.interpolate(lab[None, None].float(), size=size, mode="nearest")[0, 0].long()
        return rgb, aux, lab

    def __getitem__(self, i):
        it = self.items[i]
        rgb, aux, lab = self._read(it)
        orig = tuple(lab.shape)
        H, W = self.size
        if self.train:
            s = random.uniform(*self.scale)
            rgb, aux, lab = self._resize(rgb, aux, lab, (int(H * s), int(W * s)))
            ph, pw = max(H - lab.shape[0], 0), max(W - lab.shape[1], 0)
            if ph or pw:
                rgb, aux = F.pad(rgb, (0, pw, 0, ph)), F.pad(aux, (0, pw, 0, ph))
                lab = F.pad(lab, (0, pw, 0, ph), value=self.meta["ignore"])
            y = random.randint(0, lab.shape[0] - H)
            x = random.randint(0, lab.shape[1] - W)
            rgb, aux, lab = rgb[:, y:y + H, x:x + W], aux[:, y:y + H, x:x + W], lab[y:y + H, x:x + W]
            if random.random() < 0.5:  # horizontal flip, all modalities together
                rgb, aux, lab = rgb.flip(-1), aux.flip(-1), lab.flip(-1)
        else:  # eval: network input is resized, label stays at ORIGINAL resolution
            rgb = F.interpolate(rgb[None], size=(H, W), mode="bilinear", align_corners=False)[0]
            aux = F.interpolate(aux[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        rgb = (rgb - IMNET_MEAN) / IMNET_STD
        aux = (aux - 0.5) / 0.5
        return {"rgb": rgb, "aux": aux, "label": lab, "name": it["name"], "orig": torch.tensor(orig)}
