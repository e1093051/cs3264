"""
Organise raw data into pre-split, balanced processed class folders.

Output structure
────────────────
data/processed/
├── train/
│   ├── real/
│   ├── photoshopped/    (ALL edits for train-split persons)
│   └── ai_generated/
├── val/
│   ├── real/
│   ├── photoshopped/    (ONE random edit per val-split person)
│   └── ai_generated/
└── test/
    ├── real/
    ├── photoshopped/    (ONE random edit per test-split person)
    └── ai_generated/

Identity-aware splitting
────────────────────────
The photoshopped dataset has multiple edits of the same person.  If the same
person appears in both train and val/test the model can memorise identity
features instead of learning manipulation artefacts.

Strategy:
  1. Pool all person IDs from photoshopped/{train,test}/.
  2. Split person IDs 70/15/15 into train/val/test groups.
  3. Train persons → include ALL their edits (more data for training).
     Val/test persons → include ONE random edit each (honest evaluation).
  4. Real (_orig.jpg per person) follows the same person split, topped up
     with real_vs_fake/real/ images (randomly assigned to splits).
  5. AI-generated images are randomly split (no person identity concern).
  6. photoshopped/modified/ (Kaggle dataset, separate persons) added to train.

Raw data layout
───────────────
  ai_generated/real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/
  photoshopped/{train,test}/<person_id>/
      <id>_orig.jpg, <id>_ref.jpg, <id>_none.jpg, <id>_<edit>_<level>.jpg
  photoshopped/modified/  (1000 extra photoshopped images)

Usage
─────
    python prepare_data_local.py
    python prepare_data_local.py --max_per_class 0   # auto-balance (default)
    python prepare_data_local.py --max_per_class 5000
"""

import argparse
import random
import shutil
from pathlib import Path

import config

RAW       = Path(config.RAW_DIR)
PROCESSED = Path(config.PROCESSED_DIR)
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS    = ("train", "val", "test")


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _clear_and_create(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def _copy_list(paths: list[Path], dst: Path, prefix: str = "") -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in paths:
        fname = f"{prefix}{p.name}" if prefix else p.name
        dest = dst / fname
        if dest.exists():
            dest = dst / f"{prefix}{p.stem}_{count}{p.suffix}"
        shutil.copy2(p, dest)
        count += 1
    return count


# ── Scan photoshopped person directories ──────────────────────────────────────

def _scan_persons(split_dir: Path) -> dict[str, dict]:
    """
    Scan photoshopped/{train or test}/<person_id>/ and return a dict:
        person_id -> {"orig": Path, "edits": [Path, ...]}
    """
    persons: dict[str, dict] = {}
    if not split_dir.exists():
        print(f"  [WARNING] Not found: {split_dir}")
        return persons

    for person_dir in sorted(split_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        pid = person_dir.name
        orig = None
        edits: list[Path] = []

        for img in person_dir.iterdir():
            if not _is_image(img):
                continue
            stem = img.stem
            if stem.endswith("_orig"):
                orig = img
            elif stem.endswith("_ref") or stem.endswith("_none"):
                continue
            else:
                edits.append(img)

        if orig is not None or edits:
            persons[pid] = {"orig": orig, "edits": edits}

    return persons


# ── Collect real_vs_fake images ───────────────────────────────────────────────

def _collect_rvf(label: str) -> list[Path]:
    rvf_root = RAW / "ai_generated" / "real_vs_fake" / "real-vs-fake"
    paths: list[Path] = []
    for split in ("train", "valid", "test"):
        d = rvf_root / split / label
        if d.exists():
            paths.extend(p for p in d.iterdir() if _is_image(p))
    return paths


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Organise raw data into pre-split balanced class folders."
    )
    p.add_argument(
        "--max_per_class", type=int, default=0,
        help="Max images per class PER SPLIT. 0 = auto-balance to smallest class.",
    )
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # ── 1. Scan all photoshopped persons ──────────────────────────────────────
    print("Scanning photoshopped person directories …")
    persons_train = _scan_persons(RAW / "photoshopped" / "train")
    persons_test  = _scan_persons(RAW / "photoshopped" / "test")

    # Merge into one pool with globally unique keys
    all_persons: dict[str, dict] = {}
    for pid, data in persons_train.items():
        all_persons[f"tr_{pid}"] = data
    for pid, data in persons_test.items():
        all_persons[f"te_{pid}"] = data

    person_ids = list(all_persons.keys())
    rng.shuffle(person_ids)
    n = len(person_ids)
    n_train = int(n * config.TRAIN_RATIO)
    n_val   = int(n * config.VAL_RATIO)

    split_pids = {
        "train": person_ids[:n_train],
        "val":   person_ids[n_train : n_train + n_val],
        "test":  person_ids[n_train + n_val :],
    }
    print(f"  Total persons: {n}  →  "
          f"train={len(split_pids['train'])}, "
          f"val={len(split_pids['val'])}, "
          f"test={len(split_pids['test'])}")

    # ── 2. Build per-split file lists ─────────────────────────────────────────
    #  photoshopped: train gets ALL edits, val/test get ONE random edit
    #  real (_orig): one per person, same split as their edits
    ps_by_split:   dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    orig_by_split: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

    for split in SPLITS:
        for pid in split_pids[split]:
            data = all_persons[pid]
            # Orig → real class
            if data["orig"] is not None:
                orig_by_split[split].append(data["orig"])
            # Edits → photoshopped class
            edits = data["edits"]
            if not edits:
                continue
            if split == "train":
                ps_by_split[split].extend(edits)       # ALL edits
            else:
                ps_by_split[split].append(rng.choice(edits))  # ONE random

    # Add photoshopped/modified/ to train (separate persons, no leakage risk)
    modified_dir = RAW / "photoshopped" / "modified"
    modified = [p for p in modified_dir.iterdir() if _is_image(p)] if modified_dir.exists() else []
    ps_by_split["train"].extend(modified)

    for split in SPLITS:
        print(f"  {split:5s}  photoshopped={len(ps_by_split[split]):>5}  "
              f"orig(real)={len(orig_by_split[split]):>4}")

    # ── 3. Supplement real + collect AI-generated from real_vs_fake ────────────
    print("\nScanning real_vs_fake …")
    rvf_real = _collect_rvf("real")
    rvf_fake = _collect_rvf("fake")
    rng.shuffle(rvf_real)
    rng.shuffle(rvf_fake)
    print(f"  real_vs_fake/real: {len(rvf_real)}")
    print(f"  real_vs_fake/fake: {len(rvf_fake)}")

    # Split rvf images proportionally to match person split ratios
    def _split_list(lst: list, n_tr: int, n_vl: int) -> dict[str, list]:
        return {
            "train": lst[:n_tr],
            "val":   lst[n_tr : n_tr + n_vl],
            "test":  lst[n_tr + n_vl :],
        }

    # Proportional split sizes for rvf data
    rvf_n_train = int(len(rvf_real) * config.TRAIN_RATIO)
    rvf_n_val   = int(len(rvf_real) * config.VAL_RATIO)
    rvf_real_splits = _split_list(rvf_real, rvf_n_train, rvf_n_val)

    rvf_n_train_f = int(len(rvf_fake) * config.TRAIN_RATIO)
    rvf_n_val_f   = int(len(rvf_fake) * config.VAL_RATIO)
    ai_splits = _split_list(rvf_fake, rvf_n_train_f, rvf_n_val_f)

    # ── 4. Balance classes per split ──────────────────────────────────────────
    print("\nBalancing classes per split …")
    for split in SPLITS:
        n_ps = len(ps_by_split[split])

        # Real: orig + supplement from rvf_real
        n_orig = len(orig_by_split[split])
        supplement_needed = max(0, n_ps - n_orig)
        rvf_supplement = rvf_real_splits[split][:supplement_needed]
        real_paths = orig_by_split[split] + rvf_supplement

        # AI-generated: cap to match photoshopped count
        ai_paths = ai_splits[split][:n_ps]

        # If max_per_class is set, further cap all three
        if args.max_per_class > 0:
            cap = args.max_per_class
        else:
            cap = n_ps  # bottleneck is photoshopped

        real_paths = real_paths[:cap]
        ps_paths   = ps_by_split[split][:cap]
        ai_paths   = ai_paths[:cap]

        # Shuffle within each class
        rng.shuffle(real_paths)
        rng.shuffle(ps_paths)
        rng.shuffle(ai_paths)

        # ── 5. Copy to processed/{split}/{class}/ ────────────────────────────
        for cls in config.CLASSES:
            _clear_and_create(PROCESSED / split / cls)

        n_r  = _copy_list(real_paths, PROCESSED / split / "real")
        n_p  = _copy_list(ps_paths,   PROCESSED / split / "photoshopped")
        n_ai = _copy_list(ai_paths,   PROCESSED / split / "ai_generated")

        print(f"  {split:5s}  real={n_r:>5}  photoshopped={n_p:>5}  ai_generated={n_ai:>5}")

    # ── 6. Clean up old flat structure if it exists ───────────────────────────
    for cls in config.CLASSES:
        old_flat = PROCESSED / cls
        if old_flat.exists() and old_flat.is_dir():
            # Only remove if it's a flat class dir (not a split dir)
            children = [c.name for c in old_flat.iterdir()]
            if not any(c in ("train", "val", "test") for c in children):
                shutil.rmtree(old_flat)
                print(f"  Removed old flat dir: {old_flat}")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print(f"\nFinal counts (data/processed/):")
    grand_total = 0
    for split in SPLITS:
        split_total = 0
        for cls in config.CLASSES:
            d = PROCESSED / split / cls
            n = sum(1 for p in d.iterdir() if _is_image(p)) if d.exists() else 0
            split_total += n
        print(f"  {split:5s}: {split_total:>6} images")
        grand_total += split_total
    print(f"  {'TOTAL':5s}: {grand_total:>6} images")

    print(
        "\nDone. Person-level split ensures no identity leakage.\n"
        "Train persons get ALL edits; val/test persons get ONE random edit.\n"
        "\nNext:  python train_cnn.py --model efficientnet\n"
        "  or:  python train_cnn.py --model resnet"
    )


if __name__ == "__main__":
    main()
