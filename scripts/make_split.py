"""Carve a stratified train/val split out of D-Fire's train/ folder and lay out
data/processed/{train,val,test}/{images,labels} using symlinks (no copying, no
extra disk usage). test/ is symlinked wholesale from the untouched raw test set.

Stratified by content: background (no boxes), smoke-only, fire-only, both.

Usage:
    python scripts/make_split.py [--val-frac 0.1] [--seed 0]
"""
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "D-Fire"
OUT = ROOT / "data" / "processed"


def categorize(label_path: Path) -> str:
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return "bg"
    classes = {line.split()[0] for line in lines}
    if classes == {"0"}:
        return "smoke"
    if classes == {"1"}:
        return "fire"
    return "both"


def symlink_force(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def main(val_frac: float, seed: int):
    train_img_dir = RAW / "train" / "images"
    train_lbl_dir = RAW / "train" / "labels_clean"
    test_img_dir = RAW / "test" / "images"
    test_lbl_dir = RAW / "test" / "labels_clean"

    label_files = sorted(train_lbl_dir.glob("*.txt"))
    buckets = {"bg": [], "smoke": [], "fire": [], "both": []}
    for lp in label_files:
        buckets[categorize(lp)].append(lp)

    rng = random.Random(seed)
    train_set, val_set = [], []
    for name, files in buckets.items():
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_frac))
        val_set.extend(files[:n_val])
        train_set.extend(files[n_val:])
        print(f"  {name}: total={len(files)} -> train={len(files) - n_val} val={n_val}")

    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)

    for split_name, files in [("train", train_set), ("val", val_set)]:
        for lbl_path in files:
            stem = lbl_path.stem
            img_src = train_img_dir / f"{stem}.jpg"
            symlink_force(img_src, OUT / split_name / "images" / f"{stem}.jpg")
            symlink_force(lbl_path, OUT / split_name / "labels" / f"{stem}.txt")

    symlink_force(test_img_dir, OUT / "test" / "images")
    symlink_force(test_lbl_dir, OUT / "test" / "labels")

    yaml_path = OUT / "data.yaml"
    yaml_path.write_text(
        f"path: {OUT}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: smoke\n"
        "  1: fire\n"
    )

    print(f"\nTotal train={len(train_set)} val={len(val_set)} "
          f"test={len(list(test_lbl_dir.glob('*.txt')))} (symlinked, untouched)")
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.val_frac, args.seed)
