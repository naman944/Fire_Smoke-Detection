from pathlib import Path
from ultralytics import YOLO
import argparse

import cv2 as cv
import numpy as np

ROOT= Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve().parent
LAST_MODEL_DIR = ROOT / "runs" / "fire_detection" / "yolo11n_baseline" / "weights" / "last.pt"
BEST_MODEL_DIR = ROOT / "runs" / "fire_detection" / "yolo11n_baseline" / "weights" / "best.pt"
YAML_DIR = ROOT / "configs" / "dfire_small.yaml"

TEST_DIR = ROOT / "data" / "processed" / "D-Fire-small" / "test"
IMAGE_DIR = TEST_DIR / "images"
LABELS_DIR = TEST_DIR / "labels"

OUTPUT = SCRIPT / "assets" / "error_analysis"

MAPPING= {
    0 : "smoke",
    1 : "fire"
}

EXTENTION = {
    ".jpeg",
    ".jpg",
    ".png"
}

def parse_args():
    parse = argparse.ArgumentParser(description="Error analysis for YOLO model")
    parse.add_argument(
        "--weights",
        type= Path,
        default=str(BEST_MODEL_DIR)
        , help = "Path to the trained YOLO model weights (default: best.pt)"
    )
    parse.add_argument(
        "--images",
        type = Path,
        default=str(IMAGE_DIR),
        help = "Path to the test images directory "
    )
    parse.add_argument(
        "--labels",
        type = Path,
        default=str(LABELS_DIR),
        help = "Path to the test labels directory"
    )
    parse.add_argument(
        "--conf",
        type=float,
        default=0.325,
        help="Prediction confidence threshold."
    )
    parse.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="IoU threshold for GT/prediction matching."
    )
    parse.add_argument(
        "--max_per_category",
        type=int,
        default=5,
        help="Maximum number of images to save per error category."
    )
    return parse.parse_args()

def yolo_to_xyxy(xc,yc,w,h,img_width,img_height):
    xc *= img_width
    yc *= img_height
    w *= img_width
    h *= img_height
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def load_ground_truth(label_path:Path, img_width:int, img_height:int):
    gt_boxes=[]

    if not label_path.exists():
        return gt_boxes

    with label_path.open("r", encoding="utf-8") as f:
        for line_no , line in enumerate(f,start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"Invalid label format in {label_path} at line {line_no}: {line}")
                continue

            try:
                class_id = int(float(parts[0]))

                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:
                print(
                    f"Warning: non-numeric label line "
                    f"{label_path}:{line_no}"
                )
                continue
            box = yolo_to_xyxy(xc,yc,w,h,img_width,img_height)
            gt_boxes.append({"class_id": class_id, "box": box})

    return gt_boxes

def extract_prediction(result):
    prediction =[]

    if result.boxes is None:
        return prediction
    if len(result.boxes)==0:
        return prediction
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box , class_id,conf in zip(boxes, classes,confidences):
        prediction.append(
            {
                "class_id":class_id,
                "confidence":conf,
                "box":box.tolist()
            }
        )
    return prediction

def compute_iou(box_a, box_b):
    ax1,ay1,ax2,ay2= box_a
    bx1,by1,bx2,by2= box_b

    #  intersection x1,y1,x2,y2, width, height
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    width = max(0.0,x2-x1)
    height = max(0.0,y2-y1)

    area = width * height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - area

    if union_area <= 0:
        return 0.0

    return area / union_area

def match_pred_to_gt(gt_truth,prediction,iou_threshold):
    matches=[]

    for gt_index,gt in enumerate(gt_truth):
        for pred_index, pred in enumerate(prediction):
            iou = compute_iou(gt["box"],pred["box"])

            if iou>=iou_threshold:
                matches.append(
                    (
                        iou,
                        gt_index,
                        pred_index
                    )
                )
    matches.sort(key= lambda x:x[0] , reverse = True)

    matched_gt = set()
    matched_predictions = set()
    matches = []
    for iou, gt_index, pred_index in matches:

        if gt_index in matched_gt:
            continue

        if pred_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)

        matches.append(
            {
                "gt_index": gt_index,
                "pred_index": pred_index,
                "iou": iou,
            }
        )

    return (
        matches,
        matched_gt,
        matched_predictions,
    )

def classify_errors(ground_truth, predictions,matches,matched_gt,matched_predictions):
    errors = {
        "missed_fire": [],
        "missed_smoke": [],
        "false_fire": [],
        "false_smoke": [],
        "wrong_class": [],
    }
    for match in matches:

        gt_index = match["gt_index"]
        pred_index = match["pred_index"]

        gt = ground_truth[gt_index]
        pred = predictions[pred_index]

        if gt["class_id"] != pred["class_id"]:

            errors["wrong_class"].append(
                {
                    "gt": gt,
                    "prediction": pred,
                    "iou": match["iou"],
                }
            )

    for gt_index, gt in enumerate(ground_truth):

        if gt_index in matched_gt:
            continue

        class_id = gt["class_id"]

        if class_id == 0:
            errors["missed_smoke"].append(gt)

        elif class_id == 1:
            errors["missed_fire"].append(gt)

    for pred_index, pred in enumerate(predictions):

        if pred_index in matched_predictions:
            continue

        class_id = pred["class_id"]

        if class_id == 0:
            errors["false_smoke"].append(pred)

        elif class_id == 1:
            errors["false_fire"].append(pred)

    return errors


def draw_box(image,box,text,color,thickness = 2):
    x1,y1,x2,y2 = map(int,box)
    cv.rectangle(image,(x1,y1),(x2,y2),color,thickness)
    text_y = max(20,y1-10)
    cv.putText(image,text,(x1,text_y),cv.FONT_HERSHEY_SIMPLEX,0.5,color,2,cv.LINE_AA)

def draw_failure_case(image,ground_truth,prediction,error_category):
    output_image = image.copy()

    for gt in ground_truth:
        class_id = gt["class_id"]
        box = gt["box"]
        label = f"GT: {MAPPING[class_id]}"
        draw_box(output_image,box,label,(0,255,0),2)

    for pred in prediction:
        class_id = pred["class_id"]
        box = pred["box"]
        confidence = pred.get("confidence",0.0)
        label = f"Pred: {MAPPING[class_id]} ({confidence:.2f})"
        draw_box(output_image,box,label,(0,0,255),2)

    error_text = " | ".join(error_category)

    cv.putText(output_image,error_text,(10,30),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv.LINE_AA)
    return output_image

def create_output_directories(output_dir: Path):

    categories = [
        "missed_fire",
        "missed_smoke",
        "false_fire",
        "false_smoke",
        "wrong_class",
    ]

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for category in categories:
        (
            output_dir / category
        ).mkdir(
            parents=True,
            exist_ok=True
        )

def find_test_images(images_dir: Path):
    image_paths = [
        path
        for path in images_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in EXTENTION
        )
    ]

    return sorted(image_paths)

def main():
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(
            f"Model not found: {args.weights}"
        )

    if not args.images.exists():
        raise FileNotFoundError(
            f"Test images directory not found: {args.images}"
        )

    if not args.labels.exists():
        raise FileNotFoundError(
            f"Test labels directory not found: {args.labels}"
        )

    create_output_directories(OUTPUT)

    print(f"Loading model: {args.weights}")

    model = YOLO(str(args.weights))

    image_paths = find_test_images(args.images)
    if not image_paths:
        raise RuntimeError(
            f"No test images found in: {args.images}"
        )

    print(f"Test images found: {len(image_paths)}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU matching threshold: {args.iou}")
    print()

    error_counts = {
        "missed_fire": 0,
        "missed_smoke": 0,
        "false_fire": 0,
        "false_smoke": 0,
        "wrong_class": 0,
    }

    saved_counts = {
        "missed_fire": 0,
        "missed_smoke": 0,
        "false_fire": 0,
        "false_smoke": 0,
        "wrong_class": 0,
    }

    images_with_errors = 0

    for image_index, image_path in enumerate(
        image_paths,
        start=1
    ):

        print(
            f"[{image_index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        # Read image
        image = cv.imread(str(image_path))

        if image is None:
            print(
                f"Warning: could not read image: "
                f"{image_path}"
            )
            continue

        image_height, image_width = image.shape[:2]

        relative_image_path = image_path.relative_to(
            args.images
        )

        label_path = (
            args.labels
            / relative_image_path
        ).with_suffix(".txt")
        ground_truth = load_ground_truth(
            label_path=label_path,
            img_width=image_width,
            img_height=image_height,
        )
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            verbose=False,
        )

        predictions = extract_prediction(
            results[0]
        )


        ( matches,
            matched_gt,
            matched_predictions,
        ) = match_pred_to_gt(
            gt_truth=ground_truth,
            prediction=predictions,
            iou_threshold=args.iou,
        )
        errors = classify_errors(
            ground_truth=ground_truth,
            predictions=predictions,
            matches=matches,
            matched_gt=matched_gt,
            matched_predictions=matched_predictions,
        )

        # Categories present in this image
        image_error_categories = [
            category
            for category, items in errors.items()
            if items
        ]

        # No errors
        if not image_error_categories:
            continue

        images_with_errors += 1

        for category in error_counts:
            error_counts[category] += len(
                errors[category]
            )
        annotated_image = draw_failure_case(
            image=image,
            ground_truth=ground_truth,
            prediction=predictions,
            error_category=image_error_categories,
        )

        for category in image_error_categories:

            if (
                saved_counts[category]
                >= args.max_per_category
            ):
                continue

            output_path = (
                OUTPUT
                / category
                / image_path.name
            )

            success = cv.imwrite(
                str(output_path),
                annotated_image
            )

            if success:
                saved_counts[category] += 1

    print()
    print("=" * 60)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 60)

    print(
        f"Total test images:      {len(image_paths)}"
    )

    print(
        f"Images with errors:     {images_with_errors}"
    )

    print()

    print(
        f"Missed fire:            "
        f"{error_counts['missed_fire']}"
    )

    print(
        f"Missed smoke:           "
        f"{error_counts['missed_smoke']}"
    )

    print(
        f"False fire:             "
        f"{error_counts['false_fire']}"
    )

    print(
        f"False smoke:            "
        f"{error_counts['false_smoke']}"
    )

    print(
        f"Wrong-class confusion:  "
        f"{error_counts['wrong_class']}"
    )

    print()
    print("Saved representative images:")

    for category, count in saved_counts.items():
        print(
            f"  {category:<20} {count}"
        )

    print()
    print(
        f"Output directory: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
