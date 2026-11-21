"""Stage-wise fusion modules.

Every module receives the RGB-stream feature `x` and the auxiliary-stream
feature `a` of one encoder stage (both B x C x H x W) and returns the fused
feature that is fed to the next RGB stage and to the decoder.

>>> To add your own idea, write a new class and register it in FUSIONS.
    e.g. SE gate (lecture ICV07 SE block), concat + 1x1 conv bottleneck,
    learnable per-channel weights, ...

Team additions (Track A): concat, adaptive, concat_res. Sources are noted on each class.
"""
import torch
import torch.nn as nn


class SumFusion(nn.Module):
    """Baseline: element-wise sum."""

    def __init__(self, channels):
        super().__init__()

    def forward(self, x, a):
        return x + a


class ConcatConvFusion(nn.Module):
    """Concat RGB and IR features along channels, then a 1x1 conv bottleneck (2C -> C).

    Source: concat + 1x1 conv bottleneck idea from ICV 07 lecture (slides 40-45, 58).
    Note: the 1x1 conv is randomly initialised (PyTorch default), so at the start of
    training the fused feature fed to the next pretrained RGB stage is not x + a.
    """

    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(self, x, a):
        return self.proj(torch.cat([x, a], dim=1))


class AdaptiveFusion(nn.Module):
    """Channel-wise RGB/IR weighting from globally pooled features.

    w = sigmoid(Linear([GAP(x), GAP(a)])), output = 2 * (w * x + (1 - w) * a).
    The gate is zero-initialised, so w = 0.5 and the module starts exactly as x + a.
    Source: channel attention idea from J. Hu, L. Shen, G. Sun,
    "Squeeze-and-Excitation Networks", CVPR 2018; ICV 07 lecture SE block (slides 64-66).
    """

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Linear(2 * channels, channels)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x, a):
        rgb_summary = x.mean(dim=(2, 3))
        ir_summary = a.mean(dim=(2, 3))
        summary = torch.cat([rgb_summary, ir_summary], dim=1)
        weight = torch.sigmoid(self.gate(summary))[:, :, None, None]
        return 2.0 * (weight * x + (1.0 - weight) * a)


class ConcatResFusion(nn.Module):
    """concat -> 1x1 conv added as a *residual* on top of sum, zero-initialised.

    Hypothesis: plain ConcatConvFusion (random 1x1 init) breaks the pretrained
    RGB stream because its output feeds the next RGB stage. Starting exactly at
    `x + a` keeps the pretrained features intact and lets the conv learn a
    correction only where it helps.
    Sources: concat + 1x1 conv from ICV 07 lecture; zero-initialised last layer of a
    residual branch as in P. Goyal et al., "Accurate, Large Minibatch SGD:
    Training ImageNet in 1 Hour", 2017 (zero-init residual).
    """

    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, a):
        return x + a + self.proj(torch.cat([x, a], 1))


FUSIONS = {
    "sum": SumFusion,
    "concat": ConcatConvFusion,
    "adaptive": AdaptiveFusion,
    "concat_res": ConcatResFusion,
}
