"""
Evaluate and compare all trained models; generate visualisations.

Outputs (in results/comparison/):
  - confusion matrix per model
  - model_comparison.png  — bar chart of accuracy + macro-F1
  - gradcam_<model>.png   — Grad-CAM heatmaps (requires --gradcam flag)
  - summary table printed to stdout

Usage
─────
    python evaluate.py --backbone efficientnet
    python evaluate.py --backbone resnet --gradcam
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
from models.ensemble import load_model
from train_cnn import evaluate as cnn_evaluate
from utils import compute_metrics, plot_confusion_matrix, plot_model_comparison, set_seed


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate and compare all models.")
    p.add_argument("--backbone", choices=["efficientnet", "resnet"], default="efficientnet",
                   help="CNN backbone used to produce ensemble features.")
    p.add_argument("--gradcam", action="store_true",
                   help="Generate Grad-CAM visualisations (requires grad-cam package).")
    p.add_argument("--n_gradcam", type=int, default=6,
                   help="Number of test images to visualise with Grad-CAM.")
    return p.parse_args()


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def _gradcam_target_layer(model: nn.Module, model_name: str):
    """Return the conv layer to hook for Grad-CAM."""
    if model_name == "efficientnet":
        # Last conv block in EfficientNet-B0 (before global pool)
        return model.backbone.blocks[-1]
    else:
        # Last residual block in ResNet-50 (layer4)
        return model.backbone.layer4[-1]


def gradcam_visualise(
    model:      nn.Module,
    loader,
    device:     torch.device,
    model_name: str,
    save_dir:   Path,
    n: int = 6,
) -> None:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("[SKIP] grad-cam not installed — run  pip install grad-cam")
        return

    target_layer = _gradcam_target_layer(model, model_name)
    cam = GradCAM(model=model, target_layers=[target_layer])

    images, labels = next(iter(loader))
    images = images[:n].to(device)
    labels = labels[:n]

    with torch.enable_grad():
        targets = [ClassifierOutputTarget(l.item()) for l in labels]
        grayscale_cams = cam(input_tensor=images, targets=targets)

    # Denormalise for display
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    imgs_np = images.cpu().numpy().transpose(0, 2, 3, 1)
    imgs_np = np.clip(imgs_np * std + mean, 0, 1)

    fig = plt.figure(figsize=(3 * n, 6))
    gs  = gridspec.GridSpec(2, n, hspace=0.05, wspace=0.05)

    for i in range(n):
        ax_orig = fig.add_subplot(gs[0, i])
        ax_orig.imshow(imgs_np[i])
        ax_orig.set_title(config.CLASSES[labels[i].item()], fontsize=8)
        ax_orig.axis("off")

        cam_img = show_cam_on_image(imgs_np[i], grayscale_cams[i], use_rgb=True)
        ax_cam = fig.add_subplot(gs[1, i])
        ax_cam.imshow(cam_img)
        ax_cam.axis("off")

    fig.text(0.01, 0.75, "Original",  va="center", rotation="vertical", fontsize=10)
    fig.text(0.01, 0.25, "Grad-CAM", va="center", rotation="vertical", fontsize=10)
    plt.suptitle(f"Grad-CAM — {model_name}", fontsize=12)

    save_path = save_dir / f"gradcam_{model_name}.png"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM → {save_path}")


# ── CNN evaluation helper ─────────────────────────────────────────────────────

def _eval_cnn(model_name: str, device: torch.device, criterion: nn.Module):
    ckpt = Path(config.CHECKPOINT_DIR) / f"{model_name}_best.pth"
    if not ckpt.exists():
        print(f"[SKIP] No checkpoint for {model_name}  ({ckpt})")
        return None, None, None, None, None, None

    img_size = (
        config.EFFICIENTNET_SIZE if model_name == "efficientnet"
        else config.RESNET_SIZE
    )
    loaders, _ = get_dataloaders(image_size=img_size)

    ModelCls = EfficientNetClassifier if model_name == "efficientnet" else ResNetClassifier
    model    = ModelCls(num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    _, acc, preds, labels, probs = cnn_evaluate(model, loaders["test"], criterion, device)
    metrics = compute_metrics(labels, preds, probs, config.CLASSES)
    return model, loaders, acc, preds, labels, metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    set_seed(config.SEED)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(config.RESULTS_DIR) / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    summary: list = []

    # ── CNN models ─────────────────────────────────────────────────────────────
    for model_name in ("efficientnet", "resnet"):
        print(f"\n{'='*55}\n{model_name.upper()}\n{'='*55}")
        result = _eval_cnn(model_name, device, criterion)
        model, loaders, acc, preds, labels, metrics = result
        if model is None:
            continue

        print(f"Accuracy={acc:.4f}  macro-F1={metrics['macro_f1']:.4f}  "
              f"AUC={metrics['auc_macro']:.4f}")
        print(metrics["report"])

        summary.append({
            "model":     model_name,
            "accuracy":  metrics["accuracy"],
            "macro_f1":  metrics["macro_f1"],
            "auc_macro": metrics["auc_macro"],
        })

        plot_confusion_matrix(
            labels, preds, config.CLASSES,
            title=model_name,
            save_path=out_dir / f"cm_{model_name}.png",
        )

        if args.gradcam:
            gradcam_visualise(
                model, loaders["test"], device,
                model_name, out_dir, n=args.n_gradcam,
            )

    # ── Ensemble models ────────────────────────────────────────────────────────
    feat_dir = Path(config.FEAT_DIR) / args.backbone
    if feat_dir.exists():
        X_test = np.load(feat_dir / "test_features.npy")
        y_test = np.load(feat_dir / "test_labels.npy")

        for ens_name in ("random_forest", "xgboost"):
            full_name = f"{ens_name}+{args.backbone}"
            print(f"\n{'='*55}\n{full_name.upper()}\n{'='*55}")
            ckpt = Path(config.CHECKPOINT_DIR) / f"{ens_name}_{args.backbone}.joblib"
            if not ckpt.exists():
                print(f"[SKIP] No checkpoint for {ens_name}")
                continue

            clf   = load_model(ckpt)
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)
            metrics = compute_metrics(
                y_test.tolist(), preds.tolist(), probs.tolist(), config.CLASSES
            )

            print(f"Accuracy={metrics['accuracy']:.4f}  "
                  f"macro-F1={metrics['macro_f1']:.4f}  "
                  f"AUC={metrics['auc_macro']:.4f}")
            print(metrics["report"])

            summary.append({
                "model":     full_name,
                "accuracy":  metrics["accuracy"],
                "macro_f1":  metrics["macro_f1"],
                "auc_macro": metrics["auc_macro"],
            })

            plot_confusion_matrix(
                y_test.tolist(), preds.tolist(), config.CLASSES,
                title=full_name,
                save_path=out_dir / f"cm_{ens_name}_{args.backbone}.png",
            )
    else:
        print(f"\n[INFO] No ensemble features found at {feat_dir} — skipping ensemble eval.")

    # ── Summary table ──────────────────────────────────────────────────────────
    if not summary:
        print("\nNo trained models found. Train models first.")
        return

    summary.sort(key=lambda x: -x["accuracy"])
    print(f"\n{'='*65}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*65}")
    print(f"{'Model':<35} {'Accuracy':>10} {'Macro F1':>10} {'AUC':>8}")
    print("-" * 65)
    for r in summary:
        print(f"{r['model']:<35} {r['accuracy']:>10.4f} "
              f"{r['macro_f1']:>10.4f} {r['auc_macro']:>8.4f}")

    plot_model_comparison(summary, out_dir / "model_comparison.png")
    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
