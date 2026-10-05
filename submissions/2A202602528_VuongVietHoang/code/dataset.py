"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Đã triển khai đầy đủ để chạy trên Colab / local với dataset DeepWeeds.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc CSV của fold 0 và trả về (train_df, val_df, test_df)."""
    labels_dir = Path(labels_dir)
    files = {
        "train": labels_dir / f"train_subset{fold}.csv",
        "val": labels_dir / f"val_subset{fold}.csv",
        "test": labels_dir / f"test_subset{fold}.csv",
    }
    for k, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Thiếu file {k}: {p}")
    train_df = pd.read_csv(files["train"])
    val_df = pd.read_csv(files["val"])
    test_df = pd.read_csv(files["test"])
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "Filename" not in df.columns or "Label" not in df.columns:
            raise ValueError(f"{name} CSV không có cột Filename/Label")
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra ràng buộc chia dữ liệu DeepWeeds. Trả về dict số liệu để ghi báo cáo."""
    images_dir = Path(images_dir)
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        if split_df["Filename"].duplicated().any():
            raise ValueError(f"Filename trùng trong split {split_name}")
        labels = split_df["Label"].astype(int)
        if not labels.between(0, NUM_CLASSES - 1).all():
            raise ValueError(f"Nhãn ngoài khoảng 0..{NUM_CLASSES - 1} trong split {split_name}")

    def summarise(df):
        out = df["Label"].value_counts().sort_index().to_dict()
        return {int(k): int(v) for k, v in out.items()}

    train_names = set(train_df["Filename"].astype(str))
    val_names = set(val_df["Filename"].astype(str))
    test_names = set(test_df["Filename"].astype(str))

    overlap = {
        "train_val": len(train_names & val_names),
        "train_test": len(train_names & test_names),
        "val_test": len(val_names & test_names),
    }
    if any(v > 0 for v in overlap.values()):
        raise ValueError(f"Giao tập chồng lấn: {overlap}")

    n_total = len(train_names | val_names | test_names)
    if n_total != 17509:
        raise ValueError(f"Tổng ảnh ba tập phải bằng 17509, nhận {n_total}")

    all_names = list(train_names | val_names | test_names)
    missing = [name for name in all_names if not (images_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu ảnh trong images_dir: {missing[:10]}")

    n = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }
    per_class = {
        "train": summarise(train_df),
        "val": summarise(val_df),
        "test": summarise(test_df),
    }

    # Kiểm tra xấp xỉ 60/20/20: cho phép sai lệch nhẹ
    for split, count in n.items():
        if split == "train" and not (count > 9000 and count < 12000):
            raise ValueError(f"Số ảnh train bất thường: {count}")
        if split in {"val", "test"} and not (count > 2500 and count < 4500):
            raise ValueError(f"Số ảnh {split} bất thường: {count}")

    return {"n": n, "per_class": per_class, "overlap": overlap, "total_unique": n_total}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. Dùng cho training và validation/test."""
    if train:
        ops = [transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip()]
        if aug == "color":
            ops += [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        elif aug in {"trivial", "randaug"}:
            ops += [transforms.TrivialAugmentWide()]
        ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        return transforms.Compose(ops)

    # Val/test: resize 256 -> center crop 224
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ đường dẫn ảnh theo DataFrame."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.samples = []
        for _, row in self.df.iterrows():
            filename = str(row["Filename"])
            label = int(row["Label"])
            img_path = self.images_dir / filename
            if not img_path.exists():
                raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
            self.samples.append((filename, label, img_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        filename, label, img_path = self.samples[i]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label), filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader với sampler cân bằng lớp nếu cần."""
    dataset = DeepWeedsDataset(df, images_dir, transform)

    if sampler == "balanced":
        counts = df["Label"].value_counts().sort_index().to_dict()
        weights = np.zeros(len(df), dtype=np.float64)
        for idx, label in enumerate(df["Label"].astype(int).to_numpy()):
            weights[idx] = 1.0 / counts.get(int(label), 1)
        sampler_obj = WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=len(weights), replacement=True)
        shuffle = False
    else:
        sampler_obj = None
        shuffle = train

    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler_obj,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        worker_init_fn=seed_worker,
    )
    return loader
