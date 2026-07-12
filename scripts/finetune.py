from pathlib import Path
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
YAML_DIR = ROOT / "configs" / "dfire_small.yaml"
OUTPUT_DIR = ROOT / "runs" / "fire_detection"

BASELINE_RUN = OUTPUT_DIR / "yolo11n_baseline"

LAST_WEIGHTS = (
    BASELINE_RUN
    / "weights"
    / "last.pt"
)


# The baseline run has already finished, so we continue training from its
# weights for 20 *additional* epochs. This is a fresh fine-tune run seeded
# from last.pt, NOT resume=True (which only continues an interrupted run to
# its original epoch target and ignores a new epochs value).
EPOCHS = 20

IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = 2

# Lower learning rate for the fine-tuning pass.
LEARNING_RATE = 0.001

DEVICE = "mps"
RANDOM_SEED = 42

EXPERIMENT_NAME = "yolo11n_finetune_low_lr"




def validate_paths():

    if not YAML_DIR.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {YAML_DIR}"
        )

    if not LAST_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Last weights not found: {LAST_WEIGHTS}"
        )

    print("Dataset YAML found")
    print(f"YAML path: {YAML_DIR}")

    print("\nBaseline last.pt found")
    print(f"Weights path: {LAST_WEIGHTS}")


def load_model():

    print("\nLoading baseline last.pt...")

    model = YOLO(str(LAST_WEIGHTS))

    return model


def continue_training(model: YOLO):
    print(f"\nContinuing training from baseline last.pt for {EPOCHS} more epochs ...")
    result = model.train(
        data=str(YAML_DIR),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        seed=RANDOM_SEED,
        lr0=LEARNING_RATE,
        project=str(OUTPUT_DIR),
        name=EXPERIMENT_NAME,
        plots=True,
        save=True,
        verbose=True,
    )
    return result


def main():

    print("=" * 55)
    print("FIRE AND SMOKE DETECTION - LOW LR FINE-TUNING")
    print("=" * 55)

    validate_paths()

    model = load_model()

    print("\nBaseline last.pt model loaded successfully")

    results = continue_training(model)

    print("\nTRAINING FINISHED SUCCESSFULLY")
    print("-" * 55)

    print(
        f"\nExperiment output: "
        f"{OUTPUT_DIR / EXPERIMENT_NAME}"
    )

    return results


if __name__ == "__main__":
    main()
