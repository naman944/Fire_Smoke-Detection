from ultralytics import YOLO
import cv2
import time


FIRE_COLOR = (0, 0, 255)
SMOKE_COLOR = (255, 255, 0)
DEFAULT_COLOR = (0, 255, 0)


def load_model(model_path):
    print(f"\nLoading YOLO model from {model_path}...")
    return YOLO(model_path)

def extract_detections(result):
    """Convert YOLO boxes into the dict format expected by TemporalSmoothing."""
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf.item()),
                "class_id": int(box.cls.item()),
            }
        )
    return detections



def draw_detections(frame, smoothed, names):
    """Draw boxes/labels for smoothed detections. Returns (annotated, fire_count, smoke_count)."""
    annotated = frame.copy()
    fire_count = 0
    smoke_count = 0

    for det in smoothed:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls = det["class_id"]
        conf_score = det["confidence"]
        name = names[cls]
        label = f"{name} {conf_score:.2f}"

        if name.lower() == "fire":
            fire_count += 1
            color = FIRE_COLOR
        elif name.lower() == "smoke":
            smoke_count += 1
            color = SMOKE_COLOR
        else:
            color = DEFAULT_COLOR

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return annotated, fire_count, smoke_count


def predict_frame(frame,model:YOLO,conf:float=0.325,smoother=None):
    """
    Run inference on one image/frame.

    Returns
    -------
    annotated_frame

    detections

    fire_count

    smoke_count

    inference_ms
    """

    start = time.perf_counter()
    results = model.predict(source=frame, conf=conf, verbose=False, save=False)
    end = time.perf_counter()
    inference_ms = (end - start) * 1000
    result = results[0]
    detections = extract_detections(result)
    if smoother is not None:
        detections = smoother.smooth(detections)
    annotated, fire_count, smoke_count = draw_detections(
    frame,
    detections,
    model.names,
    )
    return (
    annotated,
    detections,
    fire_count,
    smoke_count,
    inference_ms
    )


def predict_image(image,model:YOLO,conf:float=0.325):
    annotated,detections,fire_count,smoke_count , inference_ms = predict_frame(
    image,
    model,
    conf,
    )


    return annotated, detections, fire_count, smoke_count, inference_ms
