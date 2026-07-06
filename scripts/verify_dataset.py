"""Sanity-check the D-Fire dataset: image/label pairing, class balance,
and render sample boxes per class so we can confirm which id is fire vs smoke.

Usage:
    python scripts/verify_dataset.py
Outputs:
    docs/verify_samples/class0_*.jpg
    docs/verify_samples/class1_*.jpg
    prints class balance + pairing stats to stdout
"""
import random
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DFIRE = ROOT / "data" / "raw" / "D-Fire"
OUT_DIR = ROOT / "docs" / "verify_samples"
SAMPLES_PER_CLASS = 6

COLORS = {0: (255, 0, 0), 1: (0, 0, 255)}  # BGR: class0 blue box, class1 red box


def check_pairing(split: str):
    img_dir = DFIRE / split / "images"
    lbl_dir = DFIRE / split / "labels_clean"
    images = {p.stem for p in img_dir.glob("*.jpg")}
    labels = {p.stem for p in lbl_dir.glob("*.txt")}
    missing_labels = images - labels
    missing_images = labels - images
    print(f"[{split}] images={len(images)} labels={len(labels)} "
          f"missing_labels={len(missing_labels)} missing_images={len(missing_images)}")
    return img_dir, lbl_dir


def class_balance(lbl_dir: Path):
    counts = Counter()
    empty = 0
    total = 0
    for f in lbl_dir.glob("*.txt"):
        total += 1
        lines = f.read_text().strip().splitlines()
        if not lines:
            empty += 1
            continue
        for line in lines:
            cls = line.split()[0]
            counts[cls] += 1
    print(f"  class balance (box count): {dict(counts)}  | background images: {empty}/{total}")
    return counts


def draw_samples(img_dir: Path, lbl_dir: Path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_class = {0: [], 1: []}
    label_files = list(lbl_dir.glob("*.txt"))
    random.seed(0)
    random.shuffle(label_files)

    for lbl_path in label_files:
        if len(by_class[0]) >= SAMPLES_PER_CLASS and len(by_class[1]) >= SAMPLES_PER_CLASS:
            break
        lines = lbl_path.read_text().strip().splitlines()
        if not lines:
            continue
        classes_here = {int(line.split()[0]) for line in lines}
        for cls in classes_here:
            if len(by_class[cls]) < SAMPLES_PER_CLASS:
                by_class[cls].append(lbl_path)

    for cls, paths in by_class.items():
        for i, lbl_path in enumerate(paths):
            img_path = img_dir / (lbl_path.stem + ".jpg")
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            for line in lbl_path.read_text().strip().splitlines():
                c, xc, yc, bw, bh = map(float, line.split())
                c = int(c)
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                color = COLORS.get(c, (0, 255, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, str(c), (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            out_path = OUT_DIR / f"class{cls}_{i}_{lbl_path.stem}.jpg"
            cv2.imwrite(str(out_path), img)
    print(f"Saved sample renders to {OUT_DIR} "
          f"(class0=blue boxes, class1=red boxes, label id printed on box)")


if __name__ == "__main__":
    for split in ["train", "test"]:
        img_dir, lbl_dir = check_pairing(split)
        class_balance(lbl_dir)

    train_img_dir = DFIRE / "train" / "images"
    train_lbl_dir = DFIRE / "train" / "labels_clean"
    draw_samples(train_img_dir, train_lbl_dir)
