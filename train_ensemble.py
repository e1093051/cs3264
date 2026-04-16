"""
Train Random Forest and XGBoost classifiers on top of CNN-extracted features.

Why compare CNN-only vs. CNN + ensemble?
  - A single deep model is end-to-end but can overfit on smaller datasets.
  - Random Forest on frozen CNN features is fast, interpretable (feature
    importances), and often competitive when labelled data is limited.
  - XGBoost adds boosting to capture residual errors the RF misses.

The train+val split is merged for ensemble training because the backbone
is already fixed (no risk of data leakage from the validation split).

Usage
─────
    python train_ensemble.py --backbone efficientnet
    python train_ensemble.py --backbone resnet
    python train_ensemble.py --backbone deit
    python train_ensemble.py --backbone vit
"""

import argparse
from pathlib import Path

import numpy as np

import config
from models.ensemble import build_rf_pipeline, build_xgb_pipeline, save_model
from utils import compute_metrics, save_results, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ensemble models on CNN features.")
    p.add_argument("--backbone", choices=["efficientnet", "resnet", "deit", "vit"], default="efficientnet",
                   help="Which CNN backbone's features to use.")
    return p.parse_args()


def _load_split(feat_dir: Path, split: str):
    X = np.load(feat_dir / f"{split}_features.npy")
    y = np.load(feat_dir / f"{split}_labels.npy")
    return X, y


def main() -> None:
    args = parse_args()
    set_seed(config.SEED)

    feat_dir = Path(config.FEAT_DIR) / args.backbone
    if not feat_dir.exists():
        raise FileNotFoundError(
            f"Feature directory {feat_dir} not found. "
            f"Run  python extract_features.py --backbone {args.backbone}  first."
        )

    X_train, y_train = _load_split(feat_dir, "train")
    X_val,   y_val   = _load_split(feat_dir, "val")
    X_test,  y_test  = _load_split(feat_dir, "test")

    # Merge train + val — backbone is frozen so no leakage
    X_tv = np.concatenate([X_train, X_val], axis=0)
    y_tv = np.concatenate([y_train, y_val], axis=0)
    print(f"Feature dim: {X_train.shape[1]}  |  "
          f"train+val: {len(y_tv)}  |  test: {len(y_test)}")

    ensemble_specs = {
        "random_forest": build_rf_pipeline(),
        "xgboost":       build_xgb_pipeline(),
    }

    ckpt_dir = Path(config.CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for name, clf in ensemble_specs.items():
        print(f"\n{'='*55}")
        print(f"Training {name} on {args.backbone} features …")
        print(f"{'='*55}")
        clf.fit(X_tv, y_tv)

        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)
        metrics = compute_metrics(
            y_test.tolist(), preds.tolist(), probs.tolist(), config.CLASSES
        )

        print(f"Test accuracy : {metrics['accuracy']:.4f}")
        print(f"Macro F1      : {metrics['macro_f1']:.4f}")
        print(f"Macro AUC     : {metrics['auc_macro']:.4f}")
        print(metrics["report"])

        # Persist model
        save_model(clf, ckpt_dir / f"{name}_{args.backbone}.joblib")

        # Persist results
        results_dir = Path(config.RESULTS_DIR) / f"{name}_{args.backbone}"
        save_results(
            metrics, preds.tolist(), y_test.tolist(),
            config.CLASSES, results_dir,
            model_name=f"{name}+{args.backbone}",
        )

        # Feature importance (Random Forest only)
        if name == "random_forest":
            rf = clf.named_steps["clf"]
            importances = rf.feature_importances_
            top_k = min(20, len(importances))
            top_idx = np.argsort(importances)[::-1][:top_k]
            print(f"\nTop-{top_k} feature indices by importance:")
            for rank, idx in enumerate(top_idx, 1):
                print(f"  {rank:2d}. feature[{idx:4d}]  importance={importances[idx]:.4f}")

    print("\nEnsemble training complete.")


if __name__ == "__main__":
    main()
