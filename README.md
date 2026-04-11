# CS3264 Term Project: Fake Dating Profile Image Detector

Three-class image classifier to detect **real**, **photoshopped**, and **AI-generated** face images, targeting dating app trust and safety in Singapore.

## Project Structure

```
cs3264/
├── environment.yml          # Conda environment specification
├── config.py                # Central configuration (paths, hyperparameters, classes)
├── prepare_data_local.py    # Organise raw data into person-aware train/val/test splits
├── dataset.py               # PyTorch Dataset, transforms, DataLoader factory
├── models/
│   ├── efficientnet.py      # EfficientNet-B0 backbone (5.3M params, 1280-d features)
│   ├── resnet.py            # ResNet-50 backbone (25M params, 2048-d features)
│   └── ensemble.py          # Random Forest + XGBoost pipelines on CNN features
├── train_cnn.py             # Two-phase CNN training (frozen head → full fine-tune)
├── extract_features.py      # Extract penultimate-layer features for ensemble models
├── train_ensemble.py        # Train RF + XGBoost on extracted CNN features
├── evaluate.py              # Cross-model comparison, confusion matrices, Grad-CAM
├── evaluate_robustness.py   # Robustness evaluation under realistic image degradations
└── utils.py                 # Seeding, metrics, plotting, result persistence
```

## Setup

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate cs3264
```

If using a CUDA GPU, verify PyTorch detects it:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Download raw datasets

Three datasets are needed. Download them and place under `data/raw/` as shown below.

#### Dataset A: 140k Real and Fake Faces (real + AI-generated classes)

- **Source:** Kaggle — `xhlulu/140k-real-and-fake-faces`
- **Download:** Go to https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces and download the dataset.
- **Place at:** `data/raw/ai_generated/`
- The extracted structure should look like:
  ```
  data/raw/ai_generated/
  └── real_vs_fake/real-vs-fake/
      ├── train/{real,fake}/
      ├── valid/{real,fake}/
      └── test/{real,fake}/
  ```
- `real/` folders provide the **real** class; `fake/` folders provide the **AI-generated** class (StyleGAN-generated faces).

#### Dataset B: Face Manipulation Dataset (photoshopped class, primary)

- **Source:** GitHub — `jasonkks/face-manipulation-dataset`
- **Download:** Go to https://github.com/stresearch/face-manipulation-datasets, download `train.zip` and `test.zip` from CelebHQ-FM. Extract both into `data/raw/photoshopped/`.
- **Place at:** `data/raw/photoshopped/`
- The extracted structure should look like:
  ```
  data/raw/photoshopped/
  ├── train/<person_id>/     # e.g. train/0001/, train/0002/, ...
  │   ├── <id>_orig.jpg      # original (unedited) face
  │   ├── <id>_ref.jpg       # reference crop (ignored)
  │   ├── <id>_none.jpg      # no-edit baseline (ignored)
  │   └── <id>_<edit>_<level>.jpg  # photoshopped edits
  └── test/<person_id>/      # same structure as train
  ```
- Each person folder contains the original photo plus multiple photoshopped edits (smile, age, expression changes at different levels). The `_orig.jpg` images are used as supplemental **real** images; the edit images form the **photoshopped** class.

#### Dataset C: Photoshopped Faces (photoshopped class, supplemental)

- **Source:** Kaggle — `tbourton/photoshopped-faces`
- **Download:** Go to https://www.kaggle.com/datasets/tbourton/photoshopped-faces and download the dataset.
- **Place at:** `data/raw/photoshopped/`
- The extracted structure adds these folders (only `modified/` is used):
  ```
  data/raw/photoshopped/
  └── modified/    # 1000 photoshopped portraits → photoshopped class (train only)
  ```
- The `original/` and `reference/` folder from this dataset is not used by the pipeline.

#### Final raw data layout

After downloading and extracting all three datasets:

```
data/raw/
├── ai_generated/
│   └── real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/
│
└── photoshopped/
    ├── train/<person_id>/
    ├── test/<person_id>/
    ├── modified/
```

### 3. Process data into train/val/test splits

```bash
python prepare_data_local.py
```

This performs **person-aware splitting** to prevent identity leakage:
- Persons are split 70/15/15 into train/val/test groups
- Train persons: **all** photoshopped edits included (more training data)
- Val/test persons: **one** random edit each (honest evaluation)
- All three classes are balanced to the photoshopped class size (bottleneck)

Output: `data/processed/{train,val,test}/{real,photoshopped,ai_generated}/`

## Training Pipeline

### Step 1: Train CNN backbones

```bash
python train_cnn.py --model efficientnet
python train_cnn.py --model resnet
```

Two-phase training strategy:
1. **Phase 1** (epochs 1–5): Backbone frozen, only classification head trains with high LR
2. **Phase 2** (epochs 6–20): Full fine-tuning with lower LR and cosine annealing

Outputs:
- Best checkpoint: `checkpoints/{efficientnet,resnet}_best.pth`
- Training curves: `results/{efficientnet,resnet}/training_curves.png`
- Test metrics: `results/{efficientnet,resnet}/summary.json`

### Step 2: Extract features for ensemble models

```bash
python extract_features.py --backbone efficientnet
python extract_features.py --backbone resnet
```

Passes all splits through the frozen CNN backbone using eval transforms (no augmentation) and saves the penultimate-layer feature vectors as `.npy` files.

Output: `data/features/{efficientnet,resnet}/{train,val,test}_{features,labels}.npy`

### Step 3: Train ensemble classifiers

```bash
python train_ensemble.py --backbone efficientnet
python train_ensemble.py --backbone resnet
```

Trains Random Forest and XGBoost on the extracted features. Train and val splits are merged (backbone is frozen, no leakage risk).

Output:
- Models: `checkpoints/{random_forest,xgboost}_{efficientnet,resnet}.joblib`
- Test metrics: `results/{random_forest,xgboost}_{efficientnet,resnet}/`

### Step 4: Evaluate and compare all models

```bash
python evaluate.py --backbone efficientnet --gradcam
```

Generates:
- Confusion matrices for all trained models
- Side-by-side accuracy and macro-F1 bar chart
- Grad-CAM heatmaps showing which image regions drive predictions

Output: `results/comparison/`

### Step 5: Evaluate robustness under image degradations

```bash
python evaluate_robustness.py --backbone efficientnet
python evaluate_robustness.py --backbone resnet
```

Dating-app photos typically go through JPEG recompression, resizing, and screenshotting. This script applies realistic degradations to the test set and re-evaluates all trained models (CNN + ensembles) to measure how accuracy degrades.

Degradation conditions tested:

| Condition | Description |
|---|---|
| `clean` | No degradation (baseline) |
| `jpeg_q50` | JPEG compression at quality 50 |
| `jpeg_q20` | JPEG compression at quality 20 |
| `resize_0.5x` | Downscale to 50% then upscale back |
| `resize_0.25x` | Downscale to 25% then upscale back |
| `combined` | JPEG q50 + resize 0.5x (typical dating app pipeline) |

Generates:
- Confusion matrices for each model under each degradation
- Accuracy and macro-F1 summary tables
- Grouped bar chart comparing accuracy across degradations

Output: `results/robustness/`

## Configuration

All hyperparameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_EPOCHS` | 20 | Total training epochs |
| `LEARNING_RATE` | 1e-4 | LR after backbone unfreeze |
| `UNFREEZE_AFTER` | 5 | Epochs before unfreezing backbone |
| `TRAIN_RATIO` | 0.70 | Person-level train split |
| `VAL_RATIO` | 0.15 | Person-level val split |
| `RF_N_ESTIMATORS` | 300 | Random Forest trees |
| `XGB_N_ESTIMATORS` | 300 | XGBoost rounds |

## Models Compared

| Model | Type | Parameters | Feature Dim |
|---|---|---|---|
| EfficientNet-B0 | End-to-end CNN | 5.3M | 1280 |
| ResNet-50 | End-to-end CNN | 25M | 2048 |
| RF + EfficientNet | Ensemble on CNN features | 5.3M + RF | 1280 |
| RF + ResNet | Ensemble on CNN features | 25M + RF | 2048 |
| XGBoost + EfficientNet | Ensemble on CNN features | 5.3M + XGB | 1280 |
| XGBoost + ResNet | Ensemble on CNN features | 25M + XGB | 2048 |

## Dataset Summary

| Class | Source | Description |
|---|---|---|
| Real | `real_vs_fake/real/` + `_orig.jpg` | Unmanipulated FFHQ face photos |
| Photoshopped | `train/test/<id>/<edit>.jpg` + `modified/` | Facial edits: smile, age, expression changes |
| AI-generated | `real_vs_fake/fake/` | StyleGAN-generated synthetic faces |

## Team

CS3264 Foundations of Machine Learning, Semester II 2025/26 Group 12
National University of Singapore
