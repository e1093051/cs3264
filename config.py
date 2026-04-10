"""
Central configuration — paths, hyperparameters, class definitions.
All other modules import from here; edit this file to tune the pipeline.
"""
from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_ROOT    = ROOT / "data"
RAW_DIR      = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"   # class subdirs live here
FEAT_DIR     = DATA_ROOT / "features"     # extracted CNN features (.npy)

# ── Output paths ─────────────────────────────────────────────────────────────
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR    = ROOT / "results"

# ── Classes ───────────────────────────────────────────────────────────────────
# Three-way classification for the dating-app fake-image detector
CLASSES      = ["real", "photoshopped", "ai_generated"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES  = len(CLASSES)

# ── Image sizes ───────────────────────────────────────────────────────────────
EFFICIENTNET_SIZE = 224   # EfficientNet-B0 default
RESNET_SIZE       = 224   # ResNet-50

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-4      # used after backbone unfreeze
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 4
SEED          = 42

# Warm-up phase: train only head for this many epochs, then unfreeze backbone
UNFREEZE_AFTER = 5

# ── Train / val / test split ─────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Ensemble hyperparameters ──────────────────────────────────────────────────
RF_N_ESTIMATORS  = 300
RF_MAX_DEPTH     = None      # grow full trees; rely on ensemble for regularisation

XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH    = 6
XGB_LR           = 0.05
