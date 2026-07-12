from pathlib import Path
from ultralytics import YOLO
import time
import argparse

ROOT = Path(__file__).resolve().parent.parent
MODEL_PT = ROOT / "runs" / "fire_detection" / "yolo11n_baseline" / "weights" / "best.pt"
MODEL_ONNX = MODEL_PT.with_suffix(".onnx")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Pytorch and ONNX models for inference speed")
    parser.add_argument(
        "--pt",
        type =str,
        default= str(MODEL_PT),
        help="model path of pt",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=str(MODEL_ONNX),
        help="model path of ONNX",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size ",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="image path",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="warmup iterations",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="benchmark iterations",
    )
    return parser.parse_args()

def check_file_exists(file_path):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

def benchmark(model_path:Path, args):

    print(f"\n Loading {model_path.name} ...")
    model = YOLO(model_path,task="detect")

    print(f"Warm-up ({args.warmup} iteration) ...")
    for _ in range(args.warmup):
        model.predict(source=args.image, imgsz=args.imgsz,device="cpu", conf=0.325, save=False, verbose=False)

    print(f"Benchmark ({args.runs} iteration) ...")
    start = time.perf_counter()

    for _ in range(args.runs):
        model.predict(source=args.image, imgsz=args.imgsz,device="cpu", conf=0.325, save=False, verbose=False)

    end = time.perf_counter()
    total_time = end - start
    avg_ms = (total_time/args.runs)*1000
    fps = 1000/ avg_ms
    size_mb = model_path.stat().st_size / (1024*1024)

    return {
        "latency": avg_ms,
        "fps": fps,
        "size": size_mb,
        "total_time": total_time
    }

def main():
    print("="*45)
    print("YOLO Benchmark")
    print("="*45)
    args = parse_args()
    for path in [args.pt,args.onnx, args.image]:
        check_file_exists(path)

    pt_result = benchmark(Path(args.pt) , args)
    onnx_result = benchmark(Path(args.onnx), args)

    speedup = pt_result["latency"]/onnx_result["latency"]

    print("\n" + "="*45)
    print("Benchmark_result")
    print("="*45)
    print(f"Image:{args.image}")
    print(f"Image_size:{args.imgsz}")
    print(f"Runs:{args.runs}")

    print("\nPytorch Model:")
    print("-"*45)
    print(f"latency: {pt_result['latency']:.2f} ms")
    print(f"FPS: {pt_result['fps']:.2f}")
    print(f"Size: {pt_result['size']:.2f} MB")

    print("\n ONNX Model")
    print("-"*45)
    print(f"latency: {onnx_result['latency']:.2f} ms")
    print(f"FPS: {onnx_result['fps']:.2f}")
    print(f"Size: {onnx_result['size']:.2f} MB")

    print("Comparision")
    print("-"*45)
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()
