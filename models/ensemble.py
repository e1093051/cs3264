"""
Ensemble classifiers that sit on top of CNN-extracted features.

Two pipelines:
  - Random Forest   (sklearn) — non-parametric, no feature scaling needed
                                but we include StandardScaler for XGBoost parity.
  - XGBoost         (or sklearn GradientBoosting as fallback if xgboost missing)

Both implement sklearn's fit / predict / predict_proba interface, so
evaluation code is identical for both.
"""

import joblib
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


# ── Pipeline builders ─────────────────────────────────────────────────────────

def build_rf_pipeline() -> Pipeline:
    """
    Random Forest on CNN features.
    Scaler is included for consistency; RF is scale-invariant but it
    doesn't hurt and makes the pipeline interface uniform.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            class_weight="balanced",    # handles class imbalance
            n_jobs=-1,
            random_state=config.SEED,
        )),
    ])


def build_xgb_pipeline() -> Pipeline:
    """
    XGBoost classifier on CNN features.
    Falls back to sklearn GradientBoostingClassifier if xgboost is not installed.
    """
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=config.XGB_N_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LR,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=config.SEED,
            verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        print("[INFO] xgboost not found — using sklearn GradientBoostingClassifier instead.")
        clf = GradientBoostingClassifier(
            n_estimators=config.XGB_N_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LR,
            subsample=0.8,
            random_state=config.SEED,
        )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Saved ensemble model → {path}")


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)
