"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def build_criterion(kind: str = "ce", **kw):
    """Trả về loss function hoặc module phù hợp."""
    kind = kind.lower()
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(gamma=kw.get("gamma", 2.0), alpha=kw.get("alpha"))
    if kind == "ce_weighted":
        weight = kw.get("weight")
        if weight is not None:
            return nn.CrossEntropyLoss(weight=torch.as_tensor(weight, dtype=torch.float32))
        return nn.CrossEntropyLoss()
    raise ValueError(f"Loss '{kind}' không hỗ trợ")


class LabelSmoothingCE(nn.Module):
    """Label smoothing cho CE. Khi smoothing=0 thì tương đương CE."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = float(smoothing)
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("smoothing phải nằm trong [0,1)")

    def forward(self, logits, target):
        if self.smoothing == 0:
            return F.cross_entropy(logits, target)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_target = F.one_hot(target, num_classes=logits.size(-1)).float()
            smooth_target = smooth_target * (1 - self.smoothing) + self.smoothing / logits.size(-1)
        loss = -(smooth_target * log_probs).sum(dim=-1).mean()
        return loss


class FocalLoss(nn.Module):
    """Focal loss đa lớp. gamma=0 tương đương CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        ce = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[target]
        else:
            alpha_t = torch.ones_like(pt)
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * ce
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trả về trọng số lớp theo số lượng dữ liệu, chuẩn hoá trung bình = 1."""
    if isinstance(counts, dict):
        counts = [counts.get(i, 1) for i in range(max(counts.keys()) + 1)]
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.maximum(counts, 1)
    if beta == 0:
        w = 1.0 / counts
    else:
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn theo mixup hoặc cutmix."""
    if alpha <= 0:
        return x, (y, y, 1.0)
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)
    lam = float(np.random.beta(alpha, alpha))
    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mix = lam * x + (1 - lam) * x[perm]
        return x_mix, (y_a, y_b, lam)

    if mode == "cutmix":
        h, w = x.size(-2), x.size(-1)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        cx = int(np.random.randint(w))
        cy = int(np.random.randint(h))
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, w)
        y2 = min(cy + cut_h // 2, h)
        img = x.clone()
        if x2 > x1 and y2 > y1:
            img[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(w * h))
        return img, (y_a, y_b, lam)

    raise ValueError(f"mode={mode} không hỗ trợ; dùng mixup hoặc cutmix")


def mixed_loss(criterion, logits, targets):
    """Áp dụng mất mát mixup/cutmix với trọng số lam."""
    if not isinstance(targets, tuple):
        return criterion(logits, targets)
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
