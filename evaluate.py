"""
Evaluate and compare all trained models; generate visualisations.

Outputs (in results/comparison/):
  - confusion matrix per model
  - model_comparison.png  — bar chart of accuracy + macro-F1
  - gradcam_<model>.png   — Grad-CAM (CNN only)
  - summary table printed to stdout

Usage
─────
    python evaluate.py --backbone efficientnet
    python evaluate.py --backbone resnet --gradcam
    python evaluate.py --backbone deit
    python evaluate.py --backbone vit
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config
from dataset import get_dataloaders
from models.efficientnet import EfficientNetClassifier
from models.resnet import ResNetClassifier
from models.deit import DeiTSmallClassifier
from models.vit import ViTBaseClassifier
from models.ensemble import load_model
from train_cnn import evaluate as backbone_evaluate
from utils import compute_metrics, plot_confusion_matrix, plot_model_comparison, set_seed


# ── MODEL REGISTRY ───────────────────────────────────────────────────────────

MODEL_MAP = {
    "efficientnet": {
        "class": EfficientNetClassifier,
        "img_size": config.EFFICIENTNET_SIZE,
    },
    "resnet": {
        "class": ResNetClassifier,
        "img_size": config.RESNET_SIZE,
    },
    "deit": {
        "class": DeiTSmallClassifier,
        "img_size": config.DEIT_SIZE,
    },
    "vit": {
        "class": ViTBaseClassifier,
        "img_size": config.VIT_SIZE,
    },
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate all models.")
    p.add_argument(
        "--backbone",
        choices=list(MODEL_MAP.keys()),
        default="efficientnet",
    )
    p.add_argument("--gradcam", action="store_true")
    p.add_argument("--n_gradcam", type=int, default=6)
    return p.parse_args()


# ── Grad-CAM (CNN only) ──────────────────────────────────────────────────────

def supports_gradcam(name):
    return name in ["efficientnet", "resnet"]


def get_gradcam_layer(model, name):
    if name == "efficientnet":
        return model.backbone.blocks[-1]
    elif name == "resnet":
        return model.backbone.layer4[-1]
    else:
        return None


def gradcam_visualise(model, loader, device, name, save_dir, n=6):
    if not supports_gradcam(name):
        print(f"[SKIP] Grad-CAM not supported for {name}")
        return

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("[SKIP] pip install grad-cam")
        return

    layer = get_gradcam_layer(model, name)
    cam = GradCAM(model=model, target_layers=[layer])

    images, labels = next(iter(loader))
    images = images[:n].to(device)
    labels = labels[:n]

    targets = [ClassifierOutputTarget(l.item()) for l in labels]
    cams = cam(input_tensor=images, targets=targets)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    imgs = images.cpu().numpy().transpose(0, 2, 3, 1)
    imgs = np.clip(imgs * std + mean, 0, 1)

    fig = plt.figure(figsize=(3 * n, 6))
    gs = gridspec.GridSpec(2, n)

    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(imgs[i])
        ax.axis("off")

        cam_img = show_cam_on_image(imgs[i], cams[i], use_rgb=True)
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(cam_img)
        ax2.axis("off")

    save_path = save_dir / f"gradcam_{name}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


# ── Backbone evaluation ───────────────────────────────────────────────────────

def eval_backbone(name, device, criterion):
    ckpt = Path(config.CHECKPOINT_DIR) / f"{name}_best.pth"
    if not ckpt.exists():
        print(f"[SKIP] {name} not found")
        return None

    cfg = MODEL_MAP[name]
    ModelCls = cfg["class"]
    img_size = cfg["img_size"]

    loaders, _ = get_dataloaders(image_size=img_size)

    model = ModelCls(num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    _, acc, preds, labels, probs = backbone_evaluate(
        model, loaders["test"], criterion, device
    )

    metrics = compute_metrics(labels, preds, probs, config.CLASSES)

    return model, loaders, metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(config.RESULTS_DIR) / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    summary = []

    # ── Backbone ──────────────────────────────────────────────────────────────
    result = eval_backbone(args.backbone, device, criterion)

    if result:
        model, loaders, metrics = result

        print(metrics["report"])

        summary.append({
            "model": args.backbone,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "auc_macro": metrics["auc_macro"],
        })

        plot_confusion_matrix(
            metrics["labels"],
            metrics["preds"],
            config.CLASSES,
            title=args.backbone,
            save_path=out_dir / f"cm_{args.backbone}.png",
        )

        if args.gradcam:
            gradcam_visualise(
                model, loaders["test"], device, args.backbone, out_dir
            )

    # ── Ensemble ──────────────────────────────────────────────────────────────
    feat_dir = Path(config.FEAT_DIR) / args.backbone
    if feat_dir.exists():
        X_test = np.load(feat_dir / "test_features.npy")
        y_test = np.load(feat_dir / "test_labels.npy")

        for name in ["random_forest", "xgboost"]:
            ckpt = Path(config.CHECKPOINT_DIR) / f"{name}_{args.backbone}.joblib"
            if not ckpt.exists():
                continue

            clf = load_model(ckpt)
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)

            metrics = compute_metrics(
                y_test.tolist(), preds.tolist(), probs.tolist(), config.CLASSES
            )

            summary.append({
                "model": f"{name}+{args.backbone}",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "auc_macro": metrics["auc_macro"],
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    summary.sort(key=lambda x: -x["accuracy"])

    print("\nMODEL COMPARISON")
    for r in summary:
        print(r)

    plot_model_comparison(summary, out_dir / "model_comparison.png")


if __name__ == "__main__":
    main()