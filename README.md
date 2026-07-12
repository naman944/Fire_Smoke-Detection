# 🔥 Fire & Smoke Detection

Real-time fire and smoke detection with a fine-tuned **YOLO11n** model, trained on a class-balanced subset of the [D-Fire dataset](https://github.com/gaiasd/DFireDataset). Ships with a Gradio demo for image, video, and webcam inference.

## Results

Fine-tuned YOLO11n (30 epochs, 640px) on **D-Fire-small** (2,100 train / 450 val / 450 test images, 2 classes: `smoke`, `fire`).

| Metric | Value |
|--------|-------|
| Precision | 0.71 |
| Recall | 0.59 |
| mAP@50 | 0.64 |
| mAP@50-95 | 0.34 |

An ONNX export is provided for faster CPU inference (`scripts/benchmark.py` compares PyTorch vs ONNX latency).

## Setup

```bash
pip install ultralytics opencv-python gradio onnx onnxruntime
```

## Demo

```bash
python app.py
```
Opens a Gradio UI with **Image**, **Video**, and **Webcam** tabs. Video/webcam use temporal smoothing to stabilize detections across frames.

Single-file inference:
```bash
python scripts/predict_image.py --image path/to/image.jpg
python scripts/predict_video.py --video path/to/video.mp4
```

## Pipeline

| Step | Script |
|------|--------|
| Analyze full dataset by category | `scripts/analyze_categories.py` |
| Build balanced 3k-image subset | `scripts/create_subset.py` |
| Verify subset integrity | `scripts/verify_subset.py` |
| Train | `scripts/train.py` |
| Evaluate on test split | `scripts/evalute.py` |
| Export to ONNX | `scripts/export_onnx.py` |
| Benchmark PyTorch vs ONNX | `scripts/benchmark.py` |
| Error analysis (FP/FN samples) | `scripts/error_analysis.py` |

## Dataset

D-Fire (21,527 labeled images) is subsampled with a fixed seed into a balanced set across four categories — `fire_only`, `smoke_only`, `fire_and_smoke`, and `negative/background` — to keep training fast and the class distribution even. See `scripts/reports/` for the generated manifests and analysis.

## Structure

```
app.py                  # Gradio demo (image / video / webcam)
configs/                # dataset YAML
scripts/                # data prep, training, eval, export, benchmarking
utils/                  # inference + temporal smoothing
runs/                   # trained weights and results
```
