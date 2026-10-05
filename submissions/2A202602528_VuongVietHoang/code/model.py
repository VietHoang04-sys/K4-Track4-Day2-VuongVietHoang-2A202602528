"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC."""
from __future__ import annotations

import warnings

import torch
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp từ timm."""
    model_name = SUGGESTED_BACKBONES.get(name, name)
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError(f"init phải là scratch|frozen|finetune, nhận {init}")
    model = timm.create_model(
        model_name,
        pretrained=pretrained and init != "scratch",
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    if init == "frozen":
        freeze_backbone(model)
    model.pretrained_tag = getattr(model, "pretrained_cfg", {}).get("tag", model_name)
    return model


def freeze_backbone(model) -> None:
    """Đóng băng các tham số backbone, giữ head train được."""
    for name, param in model.named_parameters():
        keep = any(key in name.lower() for key in ["fc", "head", "classifier"])
        if not keep:
            param.requires_grad = False
    if hasattr(model, "eval"):
        model.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Tạo 3 nhóm: backbone weights, backbone norm+bias, head."""
    backbone_params = []
    backbone_norm_bias = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name.lower() for key in ["fc", "head", "classifier"]):
            head_params.append(param)
        elif param.ndim <= 1:
            backbone_norm_bias.append(param)
        else:
            backbone_params.append(param)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_norm_bias:
        groups.append({"params": backbone_norm_bias, "lr": lr_backbone, "weight_decay": 0.0})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    return groups


def count_params(model) -> float:
    """Số tham số tính theo triệu."""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh; NaN biểu thị công cụ không hỗ trợ kiến trúc này."""
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        warnings.warn("ptflops chưa được cài; GMAC sẽ được ghi là NaN.", RuntimeWarning)
        return float("nan")
    try:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                macs, _ = get_model_complexity_info(
                    model, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False
                )
            return float(macs / 1e9)
        finally:
            model.train(was_training)
    except (RuntimeError, ValueError, TypeError, NotImplementedError) as exc:
        warnings.warn(f"ptflops không hỗ trợ đếm GMAC cho model này: {exc}", RuntimeWarning)
        return float("nan")
