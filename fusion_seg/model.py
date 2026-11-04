"""Single-GPU friendly multimodal SegFormer (MiT-B0 by default).

fusion modes
  rgb    : RGB only (reference for ablation)
  early  : RGB + auxiliary channels concatenated at the input (one encoder)
  <name> : two encoders (RGB / auxiliary), fused after every stage by
           fusion.FUSIONS[name]  (baseline: 'sum')
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead

from .fusion import FUSIONS


def _first_patch_embed(enc):
    # transformers >= 5: enc.stages[i]; transformers 4.x: enc.encoder.patch_embeddings[i]
    return enc.stages[0].patch_embeddings if hasattr(enc, "stages") else enc.encoder.patch_embeddings[0]


def _adapt_first_conv(encoder, in_ch):
    """Replace the 3-channel stem by an in_ch stem, re-using pretrained weights."""
    pe = _first_patch_embed(encoder)
    old = pe.proj
    if old.in_channels == in_ch:
        return
    new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        w = old.weight.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * (3.0 / in_ch)
        new.weight.copy_(w)
        new.bias.copy_(old.bias)
    pe.proj = new


def _load_encoder(backbone, pretrained):
    if pretrained:
        m, info = SegformerModel.from_pretrained(backbone, output_loading_info=True)
        n_miss, n_all = len(info.get("missing_keys", [])), len(m.state_dict())
        if n_miss:
            print(f"[warn] {n_miss}/{n_all} encoder weights were NOT loaded from {backbone}")
        if n_miss > n_all // 2:
            raise RuntimeError("pretrained weights did not load (transformers version mismatch?). "
                               "Try: pip install 'transformers==4.46.3'  or ask the instructor.")
        return m
    return SegformerModel(SegformerConfig())  # default config == MiT-B0 sizes


def _run_stage(enc, i, x):
    """One MiT stage: x (B,C,H,W) -> (B,C',H',W')."""
    if hasattr(enc, "stages"):  # transformers >= 5
        return enc.stages[i](x)
    b = x.shape[0]  # transformers 4.x
    x, h, w = enc.encoder.patch_embeddings[i](x)
    for blk in enc.encoder.block[i]:
        x = blk(x, h, w)[0]
    x = enc.encoder.layer_norm[i](x)
    return x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()


class FusionSegformer(nn.Module):
    def __init__(self, num_classes, aux_channels=1, fusion="sum",
                 backbone="nvidia/mit-b0", pretrained=True):
        super().__init__()
        self.fusion_name = fusion
        self.aux_channels = aux_channels
        self.rgb_enc = _load_encoder(backbone, pretrained)
        cfg = self.rgb_enc.config
        self.aux_enc = None
        self.fuse = None
        if fusion == "early":
            _adapt_first_conv(self.rgb_enc, 3 + aux_channels)
        elif fusion != "rgb":
            if fusion not in FUSIONS:
                raise ValueError(f"unknown fusion '{fusion}', choose from {list(FUSIONS)}")
            self.aux_enc = _load_encoder(backbone, pretrained)
            _adapt_first_conv(self.aux_enc, aux_channels)
            self.fuse = nn.ModuleList([FUSIONS[fusion](c) for c in cfg.hidden_sizes])
        cfg.num_labels = num_classes
        self.head = SegformerDecodeHead(cfg)

    def forward(self, rgb, aux=None):
        if self.fusion_name == "early":
            x = torch.cat([rgb, aux], 1)
        else:
            x = rgb
        a = aux
        feats = []
        for i in range(4):
            x = _run_stage(self.rgb_enc, i, x)
            if self.aux_enc is not None:
                a = _run_stage(self.aux_enc, i, a)
                x = self.fuse[i](x, a)
            feats.append(x)
        return self.head(tuple(feats))  # logits at 1/4 resolution

    def param_groups(self, lr, head_mult=10.0):
        head = list(self.head.parameters())
        head_ids = {id(p) for p in head}
        rest = [p for p in self.parameters() if id(p) not in head_ids]
        return [{"params": rest, "lr": lr}, {"params": head, "lr": lr * head_mult}]


def upsample_logits(logits, size):
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
