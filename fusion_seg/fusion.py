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


FUSIONS = {"sum": SumFusion}
