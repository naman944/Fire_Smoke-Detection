from pathlib import Path
import argparse
import time

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "runs" / "fire_detection" / "yolo11n_baseline" / "weights" / "best.pt"

def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to the trained YOLO model weights (default: best.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for the exported model (default: 640)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export model with dynamic axes (default: False)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export model with half precision (default: False)",
    )
    return parser.parse_args()

def load_model(model_path):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    return YOLO(model_path)

def export_model(model,imgsz,dynamic,half):
    print(f"\n Exporting to ONNX...")
    start = time.time()
    onnx_path = model.export(format="onnx", imgsz=imgsz, dynamic=dynamic, half=half)
    end = time.time()
    export_time = end - start

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX export failed, file not found: {onnx_path}")


    onnx_size = onnx_path.stat().st_size / (1024 * 1024)
    print(f"ONNX model exported successfully to {onnx_path}")
    print(f"ONNX Model size: {onnx_size} MB")
    print(f"Export time: {export_time:.2f} seconds")

def main():
    args = parse_args()
    print("="*45)
    print(f"YOLO TO ONNX EXPORTER")
    print("="*45)

    print(f"Loading YOLO model from {args.model}...")
    model = load_model(args.model)
    pt_size = Path(args.model).stat().st_size / (1024 * 1024)
    print(f"Model loaded successfully from {args.model}")
    print(f"Pytorch Model size: {pt_size} MB")


    export_model(model, args.imgsz, args.dynamic, args.half)

if __name__ == "__main__":
    main()
