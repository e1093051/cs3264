"""
Robustness evaluation — measure model performance under realistic degradations.

Dating-app photos go through JPEG recompression, resizing, and screenshotting.
This script applies these transformations to the test set and re-evaluates all
trained models to see how accuracy degrades.

Degradation conditions
──────────────────────
  clean        No degradation (baseline, same as evaluate.py)
  jpeg_q50     JPEG compression at quality 50
  jpeg_q20     JPEG compression at quality 20
  resize_0.5x  Downscale to 50% then upscale back (simulates low-res upload)
  resize_0.25x Downscale to 25% then upscale back
  combined     JPEG q50 + resize 0.5x (typical dating app pipeline)

Usage
─────
    python evaluate_robustness.py --backbone efficientnet
    python evaluate_robustness.py --backbone resnet
"""

import argparse
import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from dataset import build_file_list
from models.efficientnet import EfficientNetClassifier
from models.resnet import ResNetClassifier
from models.ensemble import load_model
from utils import compute_metrics, plot_confusion_matrix, set_seed

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ── Degradation functions ────────────────────────────────────────────────────

def _jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _resize_degrade(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _deg_clean(img: Image.Image) -> Image.Image:
    return img

def _deg_jpeg_q50(img: Image.Image) -> Image.Image:
    return _jpeg_compress(img, 50)

def _deg_jpeg_q20(img: Image.Image) -> Image.Image:
    return _jpeg_compress(img, 20)

def _deg_resize_half(img: Image.Image) -> Image.Image:
    return _resize_degrade(img, 0.5)

def _deg_resize_quarter(img: Image.Image) -> Image.Image:
    return _resize_degrade(img, 0.25)

def _deg_combined(img: Image.Image) -> Image.Image:
    return _jpeg_compress(_resize_degrade(img, 0.5), 50)


DEGRADATIONS: Dict[str, callable] = {
    "clean":        _deg_clean,
    "jpeg_q50":     _deg_jpeg_q50,
    "jpeg_q20":     _deg_jpeg_q20,
    "resize_0.5x":  _deg_resize_half,
    "resize_0.25x": _deg_resize_quarter,
    "combined":     _deg_combined,
}


# ── Dataset with degradation ────────────────────────────────────────────────

class DegradedDataset(Dataset):
    def __init__(self, file_list: List[Tuple[Path, int]], degrade_fn, image_size: int):
        self.file_list = file_list
        self.degrade_fn = degrade_fn
        self.transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        img = Image.open(path).convert("RGB")
        img = self.degrade_fn(img)
        img_np = np.array(img)
        tensor = self.transform(image=img_np)["image"]
        return tensor, label


# ── Evaluation helpers ───────────────────────────────────────────────────────

def eval_cnn(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  eval", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    return all_labels, all_preds, all_probs


def eval_ensemble(clf, features, labels):
    preds = clf.predict(features)
    probs = clf.predict_proba(features)
    return labels.tolist(), preds.tolist(), probs.tolist()


# ── Feature extraction under degradation ─────────────────────────────────────

def extract_features_degraded(model, loader, device):
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  feat", leave=False):
            images = images.to(device)
            feats = model.extract_features(images)
            all_feats.append(feats.cpu().numpy())
            all_labels.extend(labels.tolist())
    return np.concatenate(all_feats), np.array(all_labels)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Robustness evaluation under degradations.")
    p.add_argument("--backbone", choices=["efficientnet", "resnet"], default="efficientnet")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(config.RESULTS_DIR) / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = args.backbone
    img_size = config.EFFICIENTNET_SIZE if backbone == "efficientnet" else config.RESNET_SIZE

    # Load CNN model
    ckpt = Path(config.CHECKPOINT_DIR) / f"{backbone}_best.pth"
    if not ckpt.exists():
        print(f"[ERROR] No checkpoint: {ckpt}")
        return
    ModelCls = EfficientNetClassifier if backbone == "efficientnet" else ResNetClassifier
    cnn_model = ModelCls(num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    cnn_model.load_state_dict(torch.load(ckpt, map_location=device))
    cnn_model.eval()

    # Load ensemble models
    ensembles = {}
    for ens_name in ("random_forest", "xgboost"):
        ens_ckpt = Path(config.CHECKPOINT_DIR) / f"{ens_name}_{backbone}.joblib"
        if ens_ckpt.exists():
            ensembles[ens_name] = load_model(ens_ckpt)

    # Test file list
    test_files = build_file_list("test")

    # Run each degradation
    results: Dict[str, Dict[str, Dict]] = {}  # {degradation: {model: metrics}}

    for deg_name, deg_fn in DEGRADATIONS.items():
        print(f"\n{'='*60}")
        print(f"Degradation: {deg_name}")
        print(f"{'='*60}")

        ds = DegradedDataset(test_files, deg_fn, img_size)
        loader = DataLoader(
            ds, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )

        results[deg_name] = {}

        # CNN evaluation
        labels, preds, probs = eval_cnn(cnn_model, loader, device)
        metrics = compute_metrics(labels, preds, probs, config.CLASSES)
        results[deg_name][backbone] = metrics
        print(f"  {backbone:<30s}  acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}")

        plot_confusion_matrix(
            labels, preds, config.CLASSES,
            title=f"{backbone} — {deg_name}",
            save_path=out_dir / f"cm_{backbone}_{deg_name}.png",
        )

        # Ensemble evaluation — need to re-extract features on degraded images
        if ensembles:
            feats, feat_labels = extract_features_degraded(cnn_model, loader, device)
            for ens_name, clf in ensembles.items():
                full_name = f"{ens_name}+{backbone}"
                e_labels, e_preds, e_probs = eval_ensemble(clf, feats, feat_labels)
                e_metrics = compute_metrics(e_labels, e_preds, e_probs, config.CLASSES)
                results[deg_name][full_name] = e_metrics
                print(f"  {full_name:<30s}  acc={e_metrics['accuracy']:.4f}  F1={e_metrics['macro_f1']:.4f}")

                plot_confusion_matrix(
                    e_labels, e_preds, config.CLASSES,
                    title=f"{full_name} — {deg_name}",
                    save_path=out_dir / f"cm_{ens_name}_{backbone}_{deg_name}.png",
                )

    # ── Summary table ────────────────────────────────────────────────────────
    model_names = [backbone] + [f"{e}+{backbone}" for e in ensembles]
    deg_names = list(DEGRADATIONS.keys())

    print(f"\n{'='*80}")
    print("ROBUSTNESS SUMMARY — Accuracy")
    print(f"{'='*80}")
    header = f"{'Degradation':<16}" + "".join(f"{m:>26}" for m in model_names)
    print(header)
    print("-" * 80)
    for deg in deg_names:
        row = f"{deg:<16}"
        for m in model_names:
            if m in results[deg]:
                row += f"{results[deg][m]['accuracy']:>26.4f}"
            else:
                row += f"{'—':>26}"
        print(row)

    print(f"\n{'='*80}")
    print("ROBUSTNESS SUMMARY — Macro F1")
    print(f"{'='*80}")
    print(header)
    print("-" * 80)
    for deg in deg_names:
        row = f"{deg:<16}"
        for m in model_names:
            if m in results[deg]:
                row += f"{results[deg][m]['macro_f1']:>26.4f}"
            else:
                row += f"{'—':>26}"
        print(row)

    # ── Accuracy drop chart ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(deg_names))
    width = 0.8 / max(len(model_names), 1)

    for i, m in enumerate(model_names):
        accs = [results[d][m]["accuracy"] if m in results[d] else 0 for d in deg_names]
        ax.bar(x + i * width, accs, width, label=m)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(deg_names, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Robustness under degradation — {backbone} backbone")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"robustness_{backbone}.png", dpi=150)
    plt.close()
    print(f"\nSaved robustness chart → {out_dir / f'robustness_{backbone}.png'}")

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
