"""
Train DeiT-Small or ViT-Base for 3-class face authenticity classification.

Two-phase strategy
──────────────────
Phase 1 (epochs 1 … UNFREEZE_AFTER):
    Backbone frozen; only the classification head is trained with a high LR.
    This prevents destroying pretrained features in early messy gradient steps.

Phase 2 (epochs UNFREEZE_AFTER+1 … NUM_EPOCHS):
    Full fine-tuning with a lower LR and cosine annealing.

Usage
─────
    python train_transformer.py --model deit
    python train_transformer.py --model vit --epochs 25 --batch_size 16
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from dataset import get_dataloaders
from models.deit import DeiTSmallClassifier
from models.vit import ViTBaseClassifier
from utils import compute_metrics, plot_training_curves, save_results, set_seed


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Transformer backbone.")
    p.add_argument(
        "--model",
        choices=["deit", "vit"],
        default="deit",
        help="Which transformer backbone to train.",
    )
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument(
        "--lr",
        type=float,
        default=getattr(config, "TRANSFORMER_LEARNING_RATE", 5e-5),
        help="Base learning rate for transformer fine-tuning.",
    )
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument(
        "--unfreeze_after",
        type=int,
        default=config.UNFREEZE_AFTER,
        help="Epoch after which backbone weights are unfrozen.",
    )
    p.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Train from random initialisation (ablation).",
    )
    return p.parse_args()


# ── Backbone freeze / unfreeze ────────────────────────────────────────────────

def _set_backbone_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = requires_grad


# ── One epoch helpers ─────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Returns (loss, acc, preds, labels, probs)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1)

        total_loss += loss.item() * images.size(0)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels, all_probs


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(model_name: str, dropout: float, pretrained: bool) -> nn.Module:
    if model_name == "deit":
        return DeiTSmallClassifier(
            num_classes=config.NUM_CLASSES,
            pretrained=pretrained,
            dropout=dropout,
        )
    elif model_name == "vit":
        return ViTBaseClassifier(
            num_classes=config.NUM_CLASSES,
            pretrained=pretrained,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown transformer model: {model_name}")


def get_image_size(model_name: str) -> int:
    if model_name == "deit":
        return getattr(config, "DEIT_SIZE", 224)
    elif model_name == "vit":
        return getattr(config, "VIT_SIZE", 224)
    else:
        raise ValueError(f"Unknown transformer model: {model_name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model}")

    img_size = get_image_size(args.model)
    loaders, _ = get_dataloaders(image_size=img_size, batch_size=args.batch_size)

    # ── Build model ────────────────────────────────────────────────────────────
    model = build_model(
        model_name=args.model,
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Phase 1: head-only
    _set_backbone_grad(model, False)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 10,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.unfreeze_after)

    ckpt_dir = Path(config.CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.model}_best.pth"

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):

        # Switch to phase 2 fine-tuning
        if epoch == args.unfreeze_after + 1:
            print(f"\nEpoch {epoch}: unfreezing backbone — full fine-tuning begins.")
            _set_backbone_grad(model, True)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=config.WEIGHT_DECAY,
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - args.unfreeze_after,
            )

        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        vl_loss, vl_acc, _, _, _ = evaluate(
            model, loaders["val"], criterion, device
        )

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train  loss={tr_loss:.4f}  acc={tr_acc:.4f} | "
            f"Val    loss={vl_loss:.4f}  acc={vl_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  > New best — checkpoint saved  (val_acc={vl_acc:.4f})")

    # ── Test evaluation ────────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation …")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    ts_loss, ts_acc, preds, labels, probs = evaluate(
        model, loaders["test"], criterion, device
    )
    metrics = compute_metrics(labels, preds, probs, config.CLASSES)

    print(
        f"\nTest  loss={ts_loss:.4f}  acc={ts_acc:.4f}  "
        f"macro-F1={metrics['macro_f1']:.4f}  AUC={metrics['auc_macro']:.4f}"
    )
    print(metrics["report"])

    results_dir = Path(config.RESULTS_DIR) / args.model
    plot_training_curves(history, results_dir / "training_curves.png")
    save_results(metrics, preds, labels, config.CLASSES, results_dir, args.model)


if __name__ == "__main__":
    main()