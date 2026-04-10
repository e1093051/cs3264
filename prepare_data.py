"""
Organise already-downloaded raw data into the three processed class folders.

Raw layout (data/raw/)
──────────────────────
  ai_generated/real_vs_fake/real-vs-fake/{train,valid,test}/
      real/   ← used for the 'real' class
      fake/   ← used for the 'ai_generated' class

  photoshopped/
      modified/   ← used for the 'photoshopped' class
      original/   ← used as supplemental 'real' images
      reference/  ← IGNORED (cropped face used as editing reference;
                              same filenames as modified, not independent)

Class boundary justification
─────────────────────────────
  real          Unedited face photographs (Flickr-sourced, from the
                real_vs_fake dataset) + photoshopped/original/ portraits.
                Both are genuine, unmanipulated photos of real people.

  photoshopped  Retouched portraits from tbourton/photoshopped-faces
                (skin smoothing, facial reshaping, colour grading).
                Boundary case: minor colour grading alone could be
                borderline; only the 'modified' set (explicitly labelled
                as photoshopped) is used, not any ambiguous reference crops.

  ai_generated  StyleGAN1-generated faces from xhlulu/140k-real-and-fake-faces.
                Boundary case: high-quality GAN images can look indistinguishable
                from real ones at a glance; artefacts (background blur, ear/hair
                asymmetry) are detectable by CNNs but not always by humans.

Imbalance note
──────────────
  photoshopped has only 1 000 images; real and ai_generated have up to 70 000.
  By default --max_per_class=1000 to keep classes balanced.  Raise it only if
  you plan to use class-weighted loss (already the case in train_cnn.py and
  the ensemble) and accept lower recall on the photoshopped class.

Usage
─────
    python prepare_data.py                      # balanced, 1 000 per class
    python prepare_data.py --max_per_class 5000 # real+ai capped at 5 000;
                                                # photoshopped still maxes at ~1 000
"""

import argparse
import random
import shutil
from pathlib import Path

import config

RAW       = Path(config.RAW_DIR)
PROCESSED = Path(config.PROCESSED_DIR)
IMG_EXTS  = (".jpg", ".jpeg", ".png", ".webp")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _makedirs() -> None:
    for cls in config.CLASSES:
        (PROCESSED / cls).mkdir(parents=True, exist_ok=True)


def _collect(src: Path) -> list[Path]:
    """Recursively collect all image paths under src."""
    return [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS]


def _copy(paths: list[Path], dst: Path, max_n: int, prefix: str = "") -> int:
    """
    Copy up to max_n images from paths into dst (flat directory).
    prefix is prepended to filenames to avoid name collisions between sources.
    """
    dst.mkdir(parents=True, exist_ok=True)
    random.shuffle(paths)          # random sample when max_n < len(paths)
    count = 0
    for p in paths:
        if count >= max_n:
            break
        fname = f"{prefix}{p.name}" if prefix else p.name
        dest  = dst / fname
        # avoid silent overwrites if two source files share a name
        if dest.exists():
            dest = dst / f"{prefix}{p.stem}_{count}{p.suffix}"
        shutil.copy2(p, dest)
        count += 1
    return count


# ── Per-class preparers ───────────────────────────────────────────────────────

def prepare_real(max_n: int) -> None:
    """
    Sources
    -------
    Primary   : data/raw/ai_generated/real_vs_fake/real-vs-fake/{train,valid,test}/real/
    Supplement: data/raw/photoshopped/original/

    The two sources are labelled with a prefix ('rvf_' / 'ps_') so filename
    collisions are impossible even if both sets share naming conventions.
    """
    dst = PROCESSED / "real"

    # Primary source — real faces from the real-vs-fake dataset
    rvf_root = RAW / "ai_generated" / "real_vs_fake" / "real-vs-fake"
    rvf_paths: list[Path] = []
    for split in ("train", "valid", "test"):
        split_dir = rvf_root / split / "real"
        if split_dir.exists():
            rvf_paths.extend(_collect(split_dir))
        else:
            print(f"  [WARNING] Not found: {split_dir}")

    # Supplement — original (pre-edit) portraits from photoshopped dataset
    ps_orig_dir = RAW / "photoshopped" / "original"
    ps_paths: list[Path] = _collect(ps_orig_dir) if ps_orig_dir.exists() else []
    if not ps_paths:
        print(f"  [WARNING] Not found: {ps_orig_dir}")

    # Copy: fill up to max_n from the primary source first
    n_rvf  = _copy(rvf_paths, dst, max_n,           prefix="rvf_")
    remain = max(0, max_n - n_rvf)
    n_ps   = _copy(ps_paths,  dst, remain,           prefix="ps_")
    total  = n_rvf + n_ps
    print(f"  real          : {total:>6} images  "
          f"(real-vs-fake={n_rvf}, photoshopped/original={n_ps})")


def prepare_photoshopped(max_n: int) -> None:
    """
    Source: data/raw/photoshopped/modified/

    1 000 images available — the only explicitly labelled photoshopped set.
    'reference/' is excluded: it is a normalised crop used during editing,
    paired 1-to-1 with 'modified/', so including it would contaminate the
    boundary between photoshopped and real.
    """
    src = RAW / "photoshopped" / "modified"
    if not src.exists():
        print(f"  [ERROR] photoshopped/modified not found at {src}")
        return
    paths = _collect(src)
    n     = _copy(paths, PROCESSED / "photoshopped", max_n)
    print(f"  photoshopped  : {n:>6} images  (source: photoshopped/modified)")
    if n < max_n:
        print(f"  [NOTE] Only {n} photoshopped images available "
              f"(< requested {max_n}).  Consider using --max_per_class {n} "
              "for a balanced dataset.")


def prepare_ai_generated(max_n: int) -> None:
    """
    Source: data/raw/ai_generated/real_vs_fake/real-vs-fake/{train,valid,test}/fake/

    StyleGAN1-generated faces: no real person, fully synthetic.
    All three splits are pooled so the script has up to 70 000 to choose from.
    """
    rvf_root = RAW / "ai_generated" / "real_vs_fake" / "real-vs-fake"
    paths: list[Path] = []
    for split in ("train", "valid", "test"):
        split_dir = rvf_root / split / "fake"
        if split_dir.exists():
            paths.extend(_collect(split_dir))
        else:
            print(f"  [WARNING] Not found: {split_dir}")

    n = _copy(paths, PROCESSED / "ai_generated", max_n)
    print(f"  ai_generated  : {n:>6} images  (source: real-vs-fake/{'{train,valid,test}'}/fake)")


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary() -> None:
    print("\nFinal class counts (data/processed/):")
    total = 0
    for cls in config.CLASSES:
        cls_dir = PROCESSED / cls
        n = sum(1 for p in cls_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
        print(f"  {cls:<18}: {n:>6}")
        total += n
    print(f"  {'TOTAL':<18}: {total:>6}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Organise raw face images into processed class folders."
    )
    p.add_argument(
        "--max_per_class", type=int, default=1000,
        help=(
            "Max images copied per class. Default 1000 matches the size of "
            "the photoshopped class (the smallest). Increase only if you "
            "accept class imbalance."
        ),
    )
    p.add_argument("--seed", type=int, default=config.SEED,
                   help="Random seed for sampling when max_per_class < available images.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    _makedirs()

    print(f"Preparing data  (max_per_class={args.max_per_class}) …\n")
    prepare_real(args.max_per_class)
    prepare_photoshopped(args.max_per_class)
    prepare_ai_generated(args.max_per_class)

    _print_summary()
    print(
        "\nData preparation complete.\n"
        "Next:  python train_cnn.py --model efficientnet\n"
        "  or:  python train_cnn.py --model resnet"
    )


if __name__ == "__main__":
    main()
