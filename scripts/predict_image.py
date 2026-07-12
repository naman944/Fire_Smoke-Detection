from pathlib import Path
from ultralytics import YOLO
import argparse
import time
import cv2

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "runs" / "fire_detection" / "yolo11n_baseline" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents= True, exist_ok=True)

parser = argparse.ArgumentParser(description="Run inference on single image")
parser.add_argument(
    "--model",
    type = str,
    default= str(DEFAULT_MODEL),
    help = "Path to the trained YOLO model weights (default: best.pt)",
)
parser.add_argument(
    "--image",
    type=Path,
    required=True,
    help="Path to the input image for inference",
)
args=parser.parse_args()

model= YOLO(args.model)

start= time.time()

results= model.predict(
    source= str(args.image),
    conf=0.325,
    save=False,
    verbose=False
)
end=time.time()

infernce_time= (end-start)

result=results[0]
annotated= result.plot()
output_path = OUTPUT_DIR / f"annotated_{args.image.name}"

cv2.imwrite(str(output_path),annotated)


print("="*55)
print(f"Input Image: {args.image}")
print("="*55)

boxes= result.boxes
if len(boxes)==0:
    print("No Detection Found")

else:
    names= result.names
    for idx,box in enumerate(boxes,start=1):
        cls_id= int(box.cls.item())
        conf= float(box.conf.item())
        label= names[cls_id]
        x1,y1,x2,y2= box.xyxy[0].tolist()

        print(f"Detection {idx}")
        print(f"Class: {label}")
        print(f"Confidence: {conf:.4f}")
        print(f"Bounding Box: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")

print("\n" + "="*55)
print(f"Inference Time: {infernce_time:.4f} seconds")
print(f"Annotated Image saved at: {output_path}")
print("="*55)



