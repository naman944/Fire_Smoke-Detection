from pathlib import Path
from ultralytics import YOLO
import argparse
import torch


ROOT= Path(__file__).resolve().parent.parent
LAST_MODEL_DIR = ROOT / "runs" / "fire_detection" / "yolo11n_baseline" / "weights" / "last.pt"
BEST_MODEL_DIR = ROOT / "runs" / "fire_detection" / "yolo11n_baseline" / "weights" / "best.pt"
YAML_DIR = ROOT / "configs" / "dfire_small.yaml"
REPORTS_DIR = ROOT / "scripts" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on test dataset")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(BEST_MODEL_DIR),
        help="Path to the trained YOLO model weights (default: last.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(YAML_DIR),
        help="Path to the dataset YAML file (default: dfire_small.yaml)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo11n_evaluation",
        help="Name of evalution test",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    DEVICE = get_device()
    print(f"Loading YOLO model from {args.weights}...")
    weights_path = Path(args.weights)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    print(f"Evaluating model on dataset defined in {args.data}...")
    metrics = model.val(data=args.data,
                        split = "test",
                        imgsz=640,
                        batch=8,
                        device=DEVICE,
                        project = str(ROOT / "runs" / "evaluate"),
                        name = args.name,
                        verbose=True,
                        plots= True)

if __name__ == "__main__":
    main()

