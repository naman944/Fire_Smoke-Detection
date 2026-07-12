from pathlib import Path
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parent.parent

YAML_DIR = ROOT / "configs" / "dfire_small.yaml"
OUTPUT_DIR = ROOT / "runs" / "fire_detection"


MODEL_NAME = "yolo11n.pt"

EPOCHS= 30
IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = 2

RANDOM_SEED = 42
EXPERIMENT_NAME = "yolo11n_Finetune"


def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'


def validate_path():
    if not YAML_DIR.exists():
        raise FileNotFoundError(f"YAML file not found: {YAML_DIR}")

    print("dataset YAML Found")
    print (f"YAML Path: {YAML_DIR}")

def load_model():
    print("\n Loading YOLO model ...")
    model = YOLO(MODEL_NAME) # for pretrained model

    return model

def train(model:YOLO):
    print("\n Starting training ...")
    DEVICE = get_device()

    result = model.train(
        data= str(YAML_DIR),
        epochs=EPOCHS,
        imgsz = IMAGE_SIZE,
        batch= BATCH_SIZE,
        device = DEVICE,
        workers = WORKERS,
        seed = RANDOM_SEED,
        project = str(OUTPUT_DIR),
        name = EXPERIMENT_NAME,
        plots= True,
        save= True,
        verbose= True
    )
    return result

def main():
    print("="*45)
    print("FIRE AND SMOKE DETECTION TRAINING ")
    print("="*45)

    validate_path()
    model = load_model()

    print("Model load successfully")
    result = train(model)
    print("\n TRAINING FINISH SUCCESSFULLY")
    print("-"*45)

    print(f"\n Experiment output"
          f"{OUTPUT_DIR / EXPERIMENT_NAME}")
    return result

if __name__ == "__main__":
    main()



