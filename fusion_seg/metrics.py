"""Confusion-matrix based mIoU (same definition as scoring/scoring.py)."""
import numpy as np
import torch


class ConfMat:
    def __init__(self, num_classes, ignore=255):
        self.n = num_classes
        self.ignore = ignore
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, pred, gt):
        """pred, gt: integer tensors of identical shape (any device)."""
        pred, gt = pred.reshape(-1).cpu(), gt.reshape(-1).cpu()
        valid = (gt != self.ignore) & (gt >= 0) & (gt < self.n)
        idx = gt[valid] * self.n + pred[valid].clamp(0, self.n - 1)
        self.mat += torch.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def per_class_iou(self):
        m = self.mat.double()
        tp = m.diag()
        union = m.sum(0) + m.sum(1) - tp
        present = m.sum(1) > 0
        iou = torch.where(union > 0, tp / union.clamp(min=1), torch.zeros_like(tp))
        return iou.numpy(), present.numpy()

    def miou(self):
        iou, present = self.per_class_iou()
        return float(iou[present].mean() * 100) if present.any() else 0.0
