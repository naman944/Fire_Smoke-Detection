"""Gradio demo for the fire/smoke detector. Works locally and on Hugging Face
Spaces (Spaces auto-detects gradio apps and runs this file as the entrypoint).

Local run:
    python app/gradio_app.py
"""
import os
from pathlib import Path

import cv2
import gradio as gr
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MODEL_CHOICES = {
    "YOLOv11n (faster)": HERE / "weights" / "yolo11n_best.pt",
    "YOLOv11s (more accurate)": HERE / "weights" / "yolo11s_best.pt",
}
DEFAULT_MODEL = "YOLOv11n (faster)"

_loaded = {}


def get_model(name: str) -> YOLO:
    if name not in _loaded:
        path = MODEL_CHOICES[name]
        if not path.exists():
            raise gr.Error(f"Model weights not found at {path}. Copy best.pt there first.")
        _loaded[name] = YOLO(str(path))
    return _loaded[name]


def detect_image(image, model_name, conf):
    model = get_model(model_name)
    results = model.predict(image, conf=conf, verbose=False)
    return results[0].plot()[:, :, ::-1]  # BGR -> RGB


def detect_video(video_path, model_name, conf):
    model = get_model(model_name)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = str(HERE / "_last_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, conf=conf, verbose=False)
        writer.write(results[0].plot())

    cap.release()
    writer.release()
    return out_path


with gr.Blocks(title="Fire & Smoke Detector (YOLOv11)") as demo:
    gr.Markdown("# 🔥 Fire & Smoke Detector\nYOLOv11n / YOLOv11s trained on the D-Fire dataset.")

    with gr.Row():
        model_dd = gr.Dropdown(choices=list(MODEL_CHOICES.keys()), value=DEFAULT_MODEL, label="Model")
        conf_slider = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")

    with gr.Tab("Image"):
        img_in = gr.Image(type="numpy", label="Input image")
        img_out = gr.Image(type="numpy", label="Detections")
        img_btn = gr.Button("Detect")
        img_btn.click(detect_image, inputs=[img_in, model_dd, conf_slider], outputs=img_out)

    with gr.Tab("Video"):
        vid_in = gr.Video(label="Input video")
        vid_out = gr.Video(label="Detections")
        vid_btn = gr.Button("Detect")
        vid_btn.click(detect_video, inputs=[vid_in, model_dd, conf_slider], outputs=vid_out)

if __name__ == "__main__":
    demo.launch()
