from pathlib import Path
import tempfile

import cv2
import gradio as gr

from utils.inference import load_model, predict_image, predict_frame
from utils.Temporal_smoothing import TemporalSmoothing

ROOT = Path(__file__).resolve().parent
MODEL_PT = ROOT / "weights" / "best.pt"
MODEL_ONNX = MODEL_PT.with_suffix(".onnx")

# Prefer the ONNX export when available, otherwise fall back to the .pt weights.
MODEL_PATH = MODEL_ONNX if MODEL_ONNX.exists() else MODEL_PT

model = load_model(MODEL_PATH)


def detect_image(image, conf=0.325):
    """Run detection on a single RGB image coming from the Gradio Image component."""
    if image is None:
        return None, "No image provided.", 0, 0, 0.0

    # Gradio delivers RGB; the model/drawing pipeline works in BGR.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated, detections, fire_count, smoke_count, inference_ms = predict_image(
        bgr, model, conf
    )
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    summary = format_summary(detections, fire_count, smoke_count, inference_ms)
    return annotated_rgb, summary, fire_count, smoke_count, round(inference_ms, 1)


def detect_video(video_path, conf=0.325, progress=gr.Progress()):
    """Run detection frame-by-frame on a video, with temporal smoothing across frames."""
    if not video_path:
        return None, "No video provided."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Could not open video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = Path(tempfile.mkdtemp()) / "annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    smoother = TemporalSmoothing()
    frame_idx = 0
    total_fire = 0
    total_smoke = 0
    total_ms = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # cv2 frames are already BGR, matching the drawing pipeline.
        annotated, _, fire_count, smoke_count, inference_ms = predict_frame(
            frame, model, conf, smoother=smoother
        )
        writer.write(annotated)

        total_fire = max(total_fire, fire_count)
        total_smoke = max(total_smoke, smoke_count)
        total_ms += inference_ms
        frame_idx += 1

        if total_frames:
            progress(frame_idx / total_frames, desc=f"Frame {frame_idx}/{total_frames}")

    cap.release()
    writer.release()

    avg_ms = total_ms / frame_idx if frame_idx else 0.0
    summary = (
        f"Processed {frame_idx} frames\n"
        f"Peak fire detections: {total_fire}\n"
        f"Peak smoke detections: {total_smoke}\n"
        f"Avg inference: {avg_ms:.1f} ms/frame"
    )
    return str(out_path), summary


def detect_webcam(frame, conf=0.325, smoother=None):
    """Run detection on a single live webcam frame, smoothing across frames per session."""
    if frame is None:
        return None, "Waiting for webcam...", smoother

    # A fresh smoother is created on the first frame of each session and reused after.
    if smoother is None:
        smoother = TemporalSmoothing()

    # Gradio delivers RGB; the model/drawing pipeline works in BGR.
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated, detections, fire_count, smoke_count, inference_ms = predict_frame(
        bgr, model, conf, smoother=smoother
    )
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    summary = format_summary(detections, fire_count, smoke_count, inference_ms)
    return annotated_rgb, summary, smoother


def format_summary(detections, fire_count, smoke_count, inference_ms):
    lines = [
        f"Fire: {fire_count}   Smoke: {smoke_count}",
        f"Inference: {inference_ms:.1f} ms",
        f"Total detections: {len(detections)}",
    ]
    for det in detections:
        cls = model.names[det["class_id"]]
        lines.append(f"  - {cls} ({det['confidence']:.2f})")
    return "\n".join(lines)


with gr.Blocks(title="Fire & Smoke Detection") as demo:
    gr.Markdown("# 🔥 Fire & Smoke Detection")
    gr.Markdown("YOLO-based fire and smoke detector. Upload an image or a video.")

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="numpy", label="Input image")
                image_conf = gr.Slider(0.05, 0.95, value=0.325, step=0.005, label="Confidence")
                image_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                image_out = gr.Image(label="Detections")
                image_summary = gr.Textbox(label="Summary", lines=8)
                with gr.Row():
                    fire_out = gr.Number(label="Fire count")
                    smoke_out = gr.Number(label="Smoke count")
                    ms_out = gr.Number(label="Inference (ms)")

        image_btn.click(
            detect_image,
            inputs=[image_in, image_conf],
            outputs=[image_out, image_summary, fire_out, smoke_out, ms_out],
        )

    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="Input video")
                video_conf = gr.Slider(0.05, 0.95, value=0.325, step=0.005, label="Confidence")
                video_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                video_out = gr.Video(label="Detections")
                video_summary = gr.Textbox(label="Summary", lines=6)

        video_btn.click(
            detect_video,
            inputs=[video_in, video_conf],
            outputs=[video_out, video_summary],
        )

    with gr.Tab("Webcam"):
        with gr.Row():
            with gr.Column():
                webcam_in = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Webcam",
                )
                webcam_conf = gr.Slider(0.05, 0.95, value=0.325, step=0.005, label="Confidence")
            with gr.Column():
                webcam_out = gr.Image(label="Detections")
                webcam_summary = gr.Textbox(label="Summary", lines=8)

        # Per-session temporal smoother so tracks persist across streamed frames.
        webcam_smoother = gr.State()

        webcam_in.stream(
            detect_webcam,
            inputs=[webcam_in, webcam_conf, webcam_smoother],
            outputs=[webcam_out, webcam_summary, webcam_smoother],
            stream_every=0.1,
            concurrency_limit=1,
        )


if __name__ == "__main__":
    demo.launch()
