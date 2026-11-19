"""Stage-wise fusion modules.

Every module receives the RGB-stream feature `x` and the auxiliary-stream
feature `a` of one encoder stage (both B x C x H x W) and returns the fused
feature that is fed to the next RGB stage and to the decoder.

>>> To add your own idea, write a new class and register it in FUSIONS.
    e.g. SE gate (lecture ICV07 SE block), concat + 1x1 conv bottleneck,
    learnable per-channel weights, ...
"""
import torch
import torch.nn as nn


class SumFusion(nn.Module):
    """Baseline: element-wise sum."""

    def __init__(self, channels):
        super().__init__()

    def forward(self, x, a):
        return x + a


# ---- TODO for students (examples, not implemented in the baseline) ----------
# class ConcatConvFusion(nn.Module):  # cat -> 1x1 conv (channel bottleneck)
# class SEGateFusion(nn.Module):      # squeeze-excitation weights per modality


class ConcatConvFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(self, x, a):
        return self.proj(torch.cat([x, a], dim=1))


class AdaptiveFusion(nn.Module):
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


FUSIONS = {
    "sum": SumFusion,
    "concat": ConcatConvFusion,
    "adaptive": AdaptiveFusion,
}
