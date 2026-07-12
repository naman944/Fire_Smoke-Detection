from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from utils.Temporal_smoothing import TemporalSmoothing
from utils.inference import load_model , extract_detections , draw_detections

import argparse
import time
import cv2

DEFAULT_MODEL = ROOT / "runs" / "fire_detection" / "yolo11n_baseline" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIRE_COLOR = (0, 0, 255)
SMOKE_COLOR = (255, 255, 0)
DEFAULT_COLOR = (0, 255, 0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a video or webcam")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to the trained YOLO model weights (default: best.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.325,
        help="Confidence threshold for detection (default: 0.325)",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Path to the input video")
    group.add_argument("--webcam", action="store_true", help="Use webcam")

    return parser.parse_args()




def open_source(args):
    """Open the video/webcam capture and return (cap, save_output, output_path)."""
    if args.webcam:
        cap = cv2.VideoCapture(0)
        save_output = False
        output_path = None
    else:
        cap = cv2.VideoCapture(str(args.video))
        save_output = True
        output_path = OUTPUT_DIR / f"annotated_{args.video.name}"

    if not cap.isOpened():
        if args.webcam:
            raise RuntimeError("Could not open webcam (device 0).")
        raise FileNotFoundError(f"Video file not found: {args.video}")

    return cap, save_output, output_path


def create_writer(cap, output_path):
    """Build a VideoWriter matching the source resolution/fps and print video info."""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\nVideo Information")
    print("----------------------------")
    print(f"Frames : {total_frames}")
    print(f"FPS    : {fps:.2f}")
    print(f"Size   : {width} x {height}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    return writer, total_frames


def draw_hud(annotated, fire_count, smoke_count, current_fps):
    """Overlay the per-class counts and FPS readout."""
    cv2.putText(annotated, f"Fire : {fire_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, FIRE_COLOR, 2)
    cv2.putText(annotated, f"Smoke: {smoke_count}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, SMOKE_COLOR, 2)
    cv2.putText(annotated, f"FPS : {current_fps:.1f}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, DEFAULT_COLOR, 2)


def process_stream(cap, model, smoother, conf, writer, total_frames, save_output):
    """Read frames, run detection + smoothing, display/save, and return timing stats."""
    names = model.names
    frame_count = 0
    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf, verbose=False, save=False)
        detections = extract_detections(results[0])
        smoothed = smoother.smooth(detections)

        annotated, fire_count, smoke_count = draw_detections(frame, smoothed, names)

        frame_count += 1
        elapsed = time.perf_counter() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0

        draw_hud(annotated, fire_count, smoke_count, current_fps)

        cv2.imshow("YOLO Fire & Smoke Detection", annotated)

        if save_output:
            writer.write(annotated)
            if frame_count % 50 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    total_time = time.perf_counter() - start_time
    return frame_count, total_time


def print_summary(frame_count, total_time, save_output, output_path):
    avg_ms = (total_time / frame_count) * 1000 if frame_count else 0
    effective_fps = frame_count / total_time if total_time else 0

    print("\n" + "=" * 50)
    print("Inference Complete")
    print("=" * 50)
    print(f"Processed Frames : {frame_count}")
    print(f"Total Time       : {total_time:.2f} s")
    print(f"Average/frame    : {avg_ms:.2f} ms")
    print(f"Effective FPS    : {effective_fps:.2f}")
    if save_output:
        print(f"Saved Video      : {output_path}")
    print("=" * 50)


def main():
    args = parse_args()

    model = load_model(args.model)
    smoother = TemporalSmoothing()

    cap, save_output, output_path = open_source(args)

    writer = None
    total_frames = -1
    if save_output:
        writer, total_frames = create_writer(cap, output_path)
    else:
        print("\nWebcam started.")
        print("Press 'q' to quit.\n")

    try:
        frame_count, total_time = process_stream(
            cap, model, smoother, args.conf, writer, total_frames, save_output
        )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print_summary(frame_count, total_time, save_output, output_path)


if __name__ == "__main__":
    main()
