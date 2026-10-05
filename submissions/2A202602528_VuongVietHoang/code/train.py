"""train.py - vòng huấn luyện và lưu dự đoán cho DeepWeeds."""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import eval as ev
from dataset import check_split, load_split, make_loader, build_transforms
from losses import build_criterion, class_weights, mixed_loss, mix_batch
from model import SUGGESTED_BACKBONES, build_model, count_gmacs, count_params, param_groups


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    inference_method: str = "single"  # single | hflip | fivecrop
    temperature_scaling: bool = False
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    optimizer = AdamW(param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay), lr=cfg.lr_backbone)
    return optimizer


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = max(1, int(cfg.warmup_epochs * steps_per_epoch))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model) -> None:
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name not in self.shadow:
                    continue
                if torch.is_floating_point(param):
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[name].copy_(param.detach())

    def apply_to(self, model) -> None:
        model.load_state_dict(self.shadow, strict=False)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    model.train()
    if cfg.init == "frozen":
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    running_loss = 0.0
    seen = 0
    last_lr = optimizer.param_groups[0]["lr"]
    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)

        if cfg.mix in {"mixup", "cutmix"}:
            xb_mixed, mixed_targets = mix_batch(xb, yb, cfg.mix_alpha, cfg.mix)
            if cfg.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xb_mixed)
                    loss = mixed_loss(criterion, logits, mixed_targets)
            else:
                logits = model(xb_mixed)
                loss = mixed_loss(criterion, logits, mixed_targets)
        else:
            if cfg.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xb)
                    loss = criterion(logits, yb)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)

        if cfg.amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        running_loss += loss.item() * xb.size(0)
        seen += xb.size(0)
        last_lr = optimizer.param_groups[0]["lr"]

    return {"train_loss": running_loss / max(1, seen), "lr": last_lr}


def inference_logits(model, xb, method: str = "single"):
    if method == "single":
        return model(xb)
    if method == "hflip":
        p = (torch.softmax(model(xb), dim=1) + torch.softmax(model(torch.flip(xb, dims=[-1])), dim=1)) / 2
        return torch.log(p.clamp_min(1e-12))
    if method == "fivecrop":
        _, _, h, w = xb.shape
        crop = int(min(h, w) * 0.86)
        offsets = [
            (0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop),
            ((h - crop) // 2, (w - crop) // 2),
        ]
        probs = []
        for top, left in offsets:
            view = xb[:, :, top:top + crop, left:left + crop]
            view = F.interpolate(view, size=(h, w), mode="bilinear", align_corners=False)
            probs.append(torch.softmax(model(view), dim=1))
        return torch.log(torch.stack(probs).mean(dim=0).clamp_min(1e-12))
    raise ValueError(f"inference_method không hỗ trợ: {method}")


def evaluate(model, loader, criterion, device, inference_method: str = "single"):
    model.eval()
    filenames = []
    y_true = []
    logits_all = []
    total_loss = 0.0
    total_n = 0
    with torch.inference_mode():
        for xb, yb, names in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = inference_logits(model, xb, inference_method)
            filenames.extend(names)
            y_true.append(yb.cpu().numpy())
            logits_all.append(logits.cpu())
            if criterion is not None:
                total_loss += criterion(logits, yb).item() * xb.size(0)
                total_n += xb.size(0)
    y_true = np.concatenate(y_true, axis=0) if y_true else np.asarray([])
    logits_all = torch.cat(logits_all, dim=0).numpy() if logits_all else np.asarray([])
    return filenames, y_true, logits_all, total_loss / max(1, total_n)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h.get("val_loss") for h in history]
    mac_f1 = [h.get("macro_f1") for h in history]
    lr = [h.get("lr") for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_loss, label="train loss", marker="o")
    if any(v is not None for v in val_loss):
        ax1.plot(epochs, val_loss, label="val loss", marker="s")
    ax1.set_title(f"{title} - loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()

    ax2.plot(epochs, mac_f1, label="macro-F1 val", marker="o", color="C1")
    if any(v is not None for v in lr):
        ax3 = ax2.twinx()
        ax3.plot(epochs, lr, label="lr", linestyle="--", color="C2")
        ax3.set_ylabel("lr")
        ax3.legend(loc="upper right")
    ax2.set_title(f"{title} - macro-F1")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("macro-F1")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    out_root = run_dir(cfg)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    check_split(train_df, val_df, test_df, cfg.images_dir)

    train_loader = make_loader(
        train_df, cfg.images_dir, build_transforms(True, cfg.img_size, cfg.aug),
        batch_size=cfg.batch_size, train=True, sampler=cfg.sampler, num_workers=cfg.num_workers,
    )
    val_loader = make_loader(
        val_df, cfg.images_dir, build_transforms(False, cfg.img_size, cfg.aug),
        batch_size=cfg.batch_size, train=False, sampler=None, num_workers=cfg.num_workers,
    )

    model = build_model(cfg.backbone, pretrained=True, num_classes=9, drop_rate=cfg.drop_rate, init=cfg.init)
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma, weight=None)
    if cfg.class_weight_beta is not None:
        counts = train_df["Label"].value_counts().sort_index().to_dict()
        w = class_weights(counts, beta=cfg.class_weight_beta)
        criterion = build_criterion("ce_weighted", weight=w)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = criterion.to(device)
    model.to(device)
    params_m = count_params(model)
    gmac = count_gmacs(model, cfg.img_size)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=max(1, len(train_loader)))
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None

    best_metric = -1.0
    best_epoch = 0
    best_state = None
    history = []
    epoch_seconds = []

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_info = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        eval_state = None
        if ema is not None:
            eval_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.apply_to(model)
        val_names, y_true, val_logits, val_loss = evaluate(
            model, val_loader, criterion, device, cfg.inference_method
        )
        if eval_state is not None:
            model.load_state_dict(eval_state)
        elapsed = time.perf_counter() - epoch_start
        epoch_seconds.append(elapsed)
        probs = 1.0 / (1.0 + np.exp(-val_logits)) if False else (val_logits - val_logits.max(axis=1, keepdims=True))
        probs = np.exp(probs)
        probs = probs / probs.sum(axis=1, keepdims=True)
        y_pred = probs.argmax(1)
        metrics = ev.compute_metrics(y_true, y_pred, probs)
        metric = metrics["macro_f1"]
        history.append({
            "epoch": epoch,
            "train_loss": train_info["train_loss"],
            "val_loss": val_loss,
            "macro_f1": metric,
            "top1": metrics["top1"],
            "ece": metrics["ece"],
            "lr": train_info["lr"],
            "epoch_seconds": elapsed,
        })

        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in (ema.shadow if ema is not None else model.state_dict()).items()}
            torch.save(
                {
                    "model": best_state,
                    "config": asdict(cfg),
                    "epoch": best_epoch,
                    "macro_f1_val": best_metric,
                    "pretrained_tag": getattr(model, "pretrained_tag", cfg.backbone),
                },
                out_root / "best.pth",
            )

    if best_state is None:
        raise RuntimeError("Không có epoch nào đánh giá được")
    model.load_state_dict(best_state)

    val_names, y_true, val_logits, _ = evaluate(
        model, val_loader, criterion, device, cfg.inference_method
    )
    val_temperature = 1.0
    if cfg.temperature_scaling:
        from inference import fit_temperature
        val_temperature = fit_temperature(val_logits, y_true)
    val_probs_uncal = np.exp(val_logits - val_logits.max(axis=1, keepdims=True))
    val_probs_uncal = val_probs_uncal / val_probs_uncal.sum(axis=1, keepdims=True)
    scaled_val_logits = val_logits / val_temperature
    val_probs = np.exp(scaled_val_logits - scaled_val_logits.max(axis=1, keepdims=True))
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
    ev.save_predictions(pred_path(cfg, "val"), val_names, y_true, val_probs)
    if cfg.temperature_scaling:
        ev.save_predictions(
            Path(cfg.pred_dir) / f"{cfg.exp_id}_uncal_seed{cfg.seed}_val.csv",
            val_names, y_true, val_probs_uncal,
        )
    np.savez_compressed(out_root / "val_logits.npz", filenames=np.asarray(val_names), y_true=y_true, logits=val_logits)
    val_metrics = ev.compute_metrics(y_true, val_probs.argmax(1), val_probs)

    test_metrics = None
    if cfg.save_test_predictions:
        test_loader = make_loader(
            test_df, cfg.images_dir, build_transforms(False, cfg.img_size, cfg.aug),
            batch_size=cfg.batch_size, train=False, sampler=None, num_workers=cfg.num_workers,
        )
        test_names, test_y_true, test_logits, _ = evaluate(
            model, test_loader, criterion, device, cfg.inference_method
        )
        test_probs_uncal = np.exp(test_logits - test_logits.max(axis=1, keepdims=True))
        test_probs_uncal = test_probs_uncal / test_probs_uncal.sum(axis=1, keepdims=True)
        scaled_test_logits = test_logits / val_temperature
        test_probs = np.exp(scaled_test_logits - scaled_test_logits.max(axis=1, keepdims=True))
        test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)
        ev.save_predictions(pred_path(cfg, "test"), test_names, test_y_true, test_probs)
        if cfg.temperature_scaling:
            ev.save_predictions(
                Path(cfg.pred_dir) / f"{cfg.exp_id}_uncal_seed{cfg.seed}_test.csv",
                test_names, test_y_true, test_probs_uncal,
            )
        np.savez_compressed(out_root / "test_logits.npz", filenames=np.asarray(test_names), y_true=test_y_true, logits=test_logits)
        test_metrics = ev.compute_metrics(test_y_true, test_probs.argmax(1), test_probs)

    plot_curves(history, out_root / "curve.png", cfg.exp_id)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_root / "history.csv", index=False)
    (out_root / "summary.json").write_text(
        json.dumps(
            {
                "exp_id": cfg.exp_id,
                "seed": cfg.seed,
                "best_epoch": best_epoch,
                "val": {k: float(val_metrics[k]) for k in ("macro_f1", "top1", "balanced_acc", "ece", "nll")},
                "test": None if test_metrics is None else {
                    k: float(test_metrics[k]) for k in ("macro_f1", "top1", "balanced_acc", "ece", "nll")
                },
                "epoch_seconds_mean": float(np.mean(epoch_seconds)),
                "train_seconds": float(np.sum(epoch_seconds)),
                "temperature": float(val_temperature),
                "inference_method": cfg.inference_method,
                "pretrained_tag": getattr(model, "pretrained_tag", cfg.backbone),
                "params_M": params_m,
                "gmac": gmac,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "best_macro_f1": best_metric,
        "best_top1": float(val_metrics["top1"]),
        "best_ece": float(val_metrics["ece"]),
        "best_balanced_acc": float(val_metrics["balanced_acc"]),
        "temperature": float(val_temperature),
        "inference_method": cfg.inference_method,
        "epoch_seconds_mean": float(np.mean(epoch_seconds)),
        "train_seconds": float(np.sum(epoch_seconds)),
        "pretrained_tag": getattr(model, "pretrained_tag", cfg.backbone),
        "params_M": params_m,
        "gmac": gmac,
        "checkpoint": str(out_root / "best.pth"),
        "out_dir": str(out_root),
    }


def parse_overrides(pairs: list[str]) -> dict:
    cfg = Config()
    fields = {k: type(getattr(cfg, k)) for k in cfg.__dataclass_fields__}
    overrides = {}
    for chunk in pairs:
        if "=" not in chunk:
            raise ValueError(f"Override '{chunk}' không đúng định dạng key=value")
        key, value = chunk.split("=", 1)
        if not hasattr(cfg, key):
            raise ValueError(f"Key '{key}' không hợp lệ cho Config")
        dtype = fields[key]
        if value.lower() in {"none", "null"}:
            val = None
        elif dtype is bool:
            val = value.lower() in {"1", "true", "yes", "on"}
        elif dtype is int:
            val = int(value)
        elif dtype is float:
            val = float(value)
        else:
            val = value
        overrides[key] = val
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    out = run(cfg)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
