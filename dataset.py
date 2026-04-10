"""
Dataset utilities:
  - get_transforms()      albumentations pipelines per split
  - FaceDataset           PyTorch Dataset wrapping (path, label) pairs
  - build_file_list()     scan processed/<split>/<class>/ for a single split
  - get_dataloaders()     convenience wrapper returning all three DataLoaders

Splitting is handled by prepare_data_local.py (person-aware), NOT here.
This module simply reads whatever is in processed/{train,val,test}/<class>/.
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

# ImageNet stats — used because all backbones are pretrained on ImageNet
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

IMG_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str, image_size: int = config.EFFICIENTNET_SIZE) -> A.Compose:
    """Return an albumentations transform pipeline for the given split."""
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.75, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """
    Three-class face authenticity dataset.

    Args:
        file_list: list of (Path, int) — image path and class index.
        transform:  albumentations Compose pipeline.
    """

    def __init__(self, file_list: List[Tuple[Path, int]], transform: A.Compose = None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.file_list[idx]
        image = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

    def class_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = defaultdict(int)
        for _, lbl in self.file_list:
            counts[lbl] += 1
        return dict(counts)


# ── File-list helpers ─────────────────────────────────────────────────────────

def build_file_list(
    split: str,
    processed_dir: Path = config.PROCESSED_DIR,
) -> List[Tuple[Path, int]]:
    """
    Scan processed_dir/<split>/<class>/ and collect (path, label) pairs.

    Args:
        split: one of "train", "val", "test"
        processed_dir: root of the processed data tree
    """
    split_dir = processed_dir / split
    file_list: List[Tuple[Path, int]] = []

    for cls, idx in config.CLASS_TO_IDX.items():
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            print(f"[WARNING] Class directory not found: {cls_dir}  — skipping.")
            continue
        found = []
        for ext in IMG_EXTENSIONS:
            found.extend(cls_dir.glob(ext))
        if not found:
            print(f"[WARNING] No images found in {cls_dir}")
        for p in found:
            file_list.append((p, idx))

    random.shuffle(file_list)
    return file_list


# ── Public convenience wrapper ────────────────────────────────────────────────

def get_dataloaders(
    image_size:  int = config.EFFICIENTNET_SIZE,
    batch_size:  int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> Tuple[Dict[str, DataLoader], Dict[str, FaceDataset]]:
    """
    Build and return train / val / test DataLoaders and the underlying Datasets.

    Reads from the pre-split directory structure created by prepare_data_local.py:
        data/processed/{train,val,test}/{real,photoshopped,ai_generated}/

    Example::

        loaders, datasets = get_dataloaders(image_size=299)
        for images, labels in loaders["train"]:
            ...
    """
    split_lists = {}
    for split in ("train", "val", "test"):
        fl = build_file_list(split)
        if not fl:
            raise RuntimeError(
                f"No images found in data/processed/{split}/. "
                "Run  python prepare_data_local.py  first."
            )
        split_lists[split] = fl

    print(
        f"Split sizes — "
        f"train: {len(split_lists['train'])}, "
        f"val: {len(split_lists['val'])}, "
        f"test: {len(split_lists['test'])}"
    )

    datasets = {
        split: FaceDataset(fl, get_transforms(split, image_size))
        for split, fl in split_lists.items()
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )
        for split, ds in datasets.items()
    }
    return loaders, datasets
