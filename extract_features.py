"""
Extract penultimate-layer CNN features for ensemble training.

Loads the best checkpoint produced by train_cnn.py, then passes the
full dataset through the backbone (no classification head) and saves
the resulting feature arrays as NumPy files.

Uses eval transforms (no augmentation) for ALL splits so ensemble
models train on deterministic, consistent features.

Features are saved to:
    data/features/<backbone>/{train,val,test}_features.npy
    data/features/<backbone>/{train,val,test}_labels.npy

Usage
─────
    python extract_features.py --backbone efficientnet
    python extract_features.py --backbone resnet
    python extract_features.py --backbone deit
    python extract_features.py --backbone vit
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

import config
from dataset import build_file_list, FaceDataset, get_transforms
from models.efficientnet import EfficientNetClassifier
from models.resnet import ResNetClassifier
from models.deit import DeiTSmallClassifier
from models.vit import ViTBaseClassifier
from utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract CNN features for ensemble training.")
    p.add_argument("--backbone", choices=["efficientnet", "resnet", "deit", "vit"], default="efficientnet")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


@torch.no_grad()
def _extract(model: torch.nn.Module, loader, device: torch.device):
    model.eval()
    feats, labels = [], []
    for images, lbls in tqdm(loader, desc="  extracting", leave=False):
        images = images.to(device)
        f = model.extract_features(images)
        feats.append(f.cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def main() -> None:
    args   = parse_args()
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(config.CHECKPOINT_DIR) / f"{args.backbone}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. "
            f"Run  python train_cnn.py --model {args.backbone}  first."
        )

    IMG_SIZE_MAP = {
        "efficientnet": config.EFFICIENTNET_SIZE,
        "resnet": config.RESNET_SIZE,
        "deit": config.DEIT_SIZE,
        "vit": config.VIT_SIZE,
    }

    img_size = IMG_SIZE_MAP[args.backbone]

    # Use eval transforms for ALL splits — no augmentation during feature extraction
    eval_transform = get_transforms("val", img_size)
    loaders = {}
    for split in ("train", "val", "test"):
        fl = build_file_list(split)
        ds = FaceDataset(fl, eval_transform)
        loaders[split] = torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
    print(f"Loaded splits — train: {len(loaders['train'].dataset)}, "
          f"val: {len(loaders['val'].dataset)}, test: {len(loaders['test'].dataset)}")

    MODEL_MAP = {
        "efficientnet": EfficientNetClassifier,
        "resnet": ResNetClassifier,
        "deit": DeiTSmallClassifier,
        "vit": ViTBaseClassifier,
    }

    ModelCls = MODEL_MAP[args.backbone]
    model    = ModelCls(num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}")

    out_dir = Path(config.FEAT_DIR) / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        print(f"\nExtracting [{split}] features …")
        feats, lbls = _extract(model, loaders[split], device)
        np.save(out_dir / f"{split}_features.npy", feats)
        np.save(out_dir / f"{split}_labels.npy",   lbls)
        print(f"  shape={feats.shape}  →  {out_dir}")

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
