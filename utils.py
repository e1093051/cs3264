"""
Shared utilities: seeding, metrics, plotting, result persistence.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import ConfusionMatrixDisplay


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    labels:      Sequence,
    preds:       Sequence,
    probs:       Sequence,          # shape (N, num_classes)
    class_names: List[str],
) -> Dict:
    labels = list(labels)
    preds  = list(preds)
    probs  = np.asarray(probs)

    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    report   = classification_report(labels, preds, target_names=class_names, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    return {
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "auc_macro": auc,
        "report":    report,
        "labels":    labels,
        "preds":     preds,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curves(history: Dict[str, list], save_path: Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, ylabel in zip(
        axes,
        ["loss", "acc"],
        ["Cross-Entropy Loss", "Accuracy"],
    ):
        ax.plot(history[f"train_{metric}"], label="train", marker="o", markersize=3)
        ax.plot(history[f"val_{metric}"],   label="val",   marker="s", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Training {ylabel}")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    labels:      Sequence,
    preds:       Sequence,
    class_names: List[str],
    title:       str,
    save_path:   Path,
) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {save_path}")


def plot_model_comparison(summary: List[Dict], save_path: Path) -> None:
    """Bar chart comparing accuracy and macro-F1 across all models."""
    names = [r["model"] for r in summary]
    palette = sns.color_palette("Set2", len(names))

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 2 * len(names)), 5))
    for ax, metric, label in zip(
        axes,
        ["accuracy", "macro_f1"],
        ["Accuracy", "Macro F1"],
    ):
        vals = [r[metric] for r in summary]
        bars = ax.bar(names, vals, color=palette)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(label)
        ax.set_title(f"Model Comparison — {label}")
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison chart → {save_path}")


# ── Result persistence ────────────────────────────────────────────────────────

def save_results(
    metrics:     Dict,
    preds:       Sequence,
    labels:      Sequence,
    class_names: List[str],
    results_dir: Path,
    model_name:  str,
) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model":     model_name,
        "accuracy":  metrics["accuracy"],
        "macro_f1":  metrics["macro_f1"],
        "auc_macro": metrics["auc_macro"],
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(metrics["report"])

    plot_confusion_matrix(
        labels, preds, class_names,
        title=model_name,
        save_path=results_dir / "confusion_matrix.png",
    )
    print(f"Results saved → {results_dir}")
