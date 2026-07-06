"""Run val on the held-out D-Fire test split for one or more trained models and
print a comparison table (mAP@0.5, mAP@0.5:0.95, speed) to paste into the README.

Usage:
    python scripts/evaluate.py --weights runs/detect/yolo11n_fire_smoke/weights/best.pt runs/detect/yolo11s_fire_smoke/weights/best.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "processed" / "data.yaml"


def main(weights_paths):
    rows = []
    for wp in weights_paths:
        wp = Path(wp)
        model = YOLO(str(wp))
        metrics = model.val(data=str(DATA_YAML), split="test", imgsz=416, device="mps")
        speed = metrics.speed  # dict: preprocess/inference/postprocess ms
        row = {
            "model": wp.parent.parent.name,
            "mAP50": round(metrics.box.map50, 4),
            "mAP50-95": round(metrics.box.map, 4),
            "precision": round(metrics.box.mp, 4),
            "recall": round(metrics.box.mr, 4),
            "inference_ms": round(speed.get("inference", 0.0), 2),
        }
        rows.append(row)

    print("\n| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Inference (ms/img) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['mAP50']} | {r['mAP50-95']} | {r['precision']} | "
              f"{r['recall']} | {r['inference_ms']} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    args = ap.parse_args()
    main(args.weights)
