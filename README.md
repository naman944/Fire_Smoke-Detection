# 🔥 Fire & Smoke Detection (YOLOv11)

Real-time fire and smoke detection with YOLOv11n and YOLOv11s, trained on the
[D-Fire dataset](https://github.com/gaiasd/DFireDataset) plus a small set of
personally captured and annotated images.

**Live demo:** TODO — Hugging Face Spaces link
**Demo video:** TODO — 1-minute clip link

## Results

Evaluated on the held-out D-Fire `test/` split (4,306 images, never used in training):

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Params | Inference (ms/img, M3 MPS) |
|---|---|---|---|---|---|---|
| YOLOv11n | TODO | TODO | TODO | TODO | ~2.6M | TODO |
| YOLOv11s | TODO | TODO | TODO | TODO | ~9.4M | TODO |

## Dataset

- **D-Fire**: 17,221 train / 4,306 test images, YOLO-format boxes, 2 classes
  (`0: smoke`, `1: fire`). ~45% of train images are background (no fire/smoke),
  kept to reduce false positives.
- Training used a stratified subsample (~8,000 images, proportional across
  background/smoke/fire/both categories) of the train split to keep training
  time tractable on local hardware; a 90/10 stratified split of D-Fire's
  original train set produced the validation set. The test set was never
  touched during training or model selection.
- `data/custom/`: ~100-200 images captured and annotated by hand (CVAT/Roboflow)
  for real-world qualitative testing and the demo video.

## Method

- Models: [Ultralytics YOLOv11](https://docs.ultralytics.com) nano and small,
  fine-tuned from COCO-pretrained weights.
- Training hardware: Apple M3, 8 GB unified memory (PyTorch MPS backend).
- Config: imgsz=416, batch=8, up to 40 epochs with early stopping (patience=15).

## Repo structure

```
data/
  raw/D-Fire/        # original dataset (not committed, see Setup)
  processed/         # generated train/val/test split + data.yaml (not committed)
  custom/            # self-collected + annotated images (committed)
scripts/
  verify_dataset.py  # label sanity checks + class-mapping visualization
  make_split.py      # stratified train/val split + data.yaml generator
  evaluate.py         # mAP evaluation on the test split
app/
  gradio_app.py      # demo app (image + video), deployable to HF Spaces
runs/                # ultralytics training outputs (not committed)
docs/                # project plan PDF, sample renders
```

## Setup

```bash
conda create -n fireai python=3.11 -y
conda activate fireai
pip install -r requirements.txt
```

Download D-Fire and place it at `data/raw/D-Fire/{train,test}/{images,labels_clean}`,
then:

```bash
python scripts/verify_dataset.py     # sanity check + class mapping renders
python scripts/make_split.py --train-cap 8000
```

## Training

```bash
yolo detect train model=yolo11n.pt data=data/processed/data.yaml \
  epochs=40 imgsz=416 batch=8 device=mps patience=15 \
  project=runs/detect name=yolo11n_fire_smoke

yolo detect train model=yolo11s.pt data=data/processed/data.yaml \
  epochs=40 imgsz=416 batch=8 device=mps patience=15 \
  project=runs/detect name=yolo11s_fire_smoke
```

## Evaluation

```bash
python scripts/evaluate.py --weights \
  runs/detect/yolo11n_fire_smoke/weights/best.pt \
  runs/detect/yolo11s_fire_smoke/weights/best.pt
```

## Run the demo locally

```bash
cp runs/detect/yolo11n_fire_smoke/weights/best.pt app/weights/yolo11n_best.pt
cp runs/detect/yolo11s_fire_smoke/weights/best.pt app/weights/yolo11s_best.pt
python app/gradio_app.py
```

## Limitations & future work

- TODO after evaluation: note failure modes (e.g. small/distant fire, night
  scenes, orange/red objects triggering false positives, thin smoke).
- Custom dataset is small (~100-200 images); mainly useful for qualitative
  checks, not a rigorous benchmark.

## Credits

- Dataset: [D-Fire](https://github.com/gaiasd/DFireDataset) (Pedro Vinícius
  Almeida Borges de Venâncio et al.)
- Models: [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
