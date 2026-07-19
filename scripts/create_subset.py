from pathlib import Path
import json
import random
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_ROOT =PROJECT_ROOT / "data" / "raw" / "D-Fire"
SUBSET_ROOT =PROJECT_ROOT / "data" / "processed" / "D-Fire-small"


MANIFEST_JSON =SCRIPT_DIR / "reports" / "dataset_manifest.json"

RANDOM_SEED = 42

SPLIT_TARGETS = {
    "train": {
        "fire_only": 844,
        "smoke_only": 4500,
        "fire_and_smoke": 3600,
        "negative_or_background": 7733
    },

    "val": {
        "fire_only": 100,
        "smoke_only": 181,
        "fire_and_smoke": 163,
        "negative_or_background": 100
    },

    "test": {
        "fire_only": 220,
        "smoke_only": 1186,
        "fire_and_smoke": 895,
        "negative_or_background": 2005
    }
}

def load_manifest(manifest_path: Path):
    """
    Load the category manifest JSON file.
    """

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest file does not exist: "
            f"{manifest_path}"
        )

    with manifest_path.open("r",encoding="utf-8") as f:
        manifest = json.load(f)

    return manifest

def validate_manifest_structure(manifest: dict):
    """
    Check that required splits and categories
    exist in the manifest.
    """

    required_source_splits = {"train","test"}
    missing_splits = (required_source_splits- manifest.keys())

    if missing_splits:
        raise ValueError(f"Missing source splits in manifest:{sorted(missing_splits)}")

    required_categories = set(SPLIT_TARGETS["train"].keys())

    for split_name in required_source_splits:

        available_categories = set(
            manifest[split_name].keys()
        )

        missing_categories = (
            required_categories
            - available_categories
        )

        if missing_categories:
            raise ValueError(
                f"Missing categories in manifest "
                f"split '{split_name}': "
                f"{sorted(missing_categories)}"
            )

def validate_targets(manifest: dict):
    """
    Validate whether enough images are available.

    """

    categories = SPLIT_TARGETS["train"].keys()

    for category in categories:

        available_train = len(
            manifest["train"][category]
        )

        available_test = len(
            manifest["test"][category]
        )

        required_from_train = (
            SPLIT_TARGETS["train"][category]
            + SPLIT_TARGETS["val"][category]
        )

        required_from_test = (
            SPLIT_TARGETS["test"][category]
        )

        print(f"\nCategory: {category}")

        print(
            f"  Original train: "
            f"available={available_train}, "
            f"required={required_from_train}"
        )

        print(
            f"  Original test : "
            f"available={available_test}, "
            f"required={required_from_test}"
        )

        if available_train < required_from_train:
            raise ValueError(
                f"Not enough original TRAIN images "
                f"for category '{category}'. "
                f"Available={available_train}, "
                f"Required={required_from_train}"
            )

        if available_test < required_from_test:
            raise ValueError(
                f"Not enough original TEST images "
                f"for category '{category}'. "
                f"Available={available_test}, "
                f"Required={required_from_test}"
            )


def sample_subset(manifest: dict):
    """
    Reproducibly

    """

    rng = random.Random(RANDOM_SEED)

    selected = {
        "train": {},
        "val": {},
        "test": {}
    }

    categories = SPLIT_TARGETS["train"].keys()

    for category in categories:


        train_pool = list(
            manifest["train"][category]
        )

        rng.shuffle(train_pool)

        train_count = (
            SPLIT_TARGETS["train"][category]
        )

        val_count = (
            SPLIT_TARGETS["val"][category]
        )

        # First section -> small train
        selected["train"][category] = (
            train_pool[:train_count]
        )

        # Next section -> small val
        selected["val"][category] = (
            train_pool[
                train_count:
                train_count + val_count
            ]
        )

        test_pool = list(
            manifest["test"][category]
        )

        rng.shuffle(test_pool)

        test_count = (
            SPLIT_TARGETS["test"][category]
        )

        selected["test"][category] = (
            test_pool[:test_count]
        )

    return selected

def verify_no_overlap(selected: dict):

    split_files = {}

    for split_name in ("train", "val", "test"):

        filenames = set()

        for category_files in (
            selected[split_name].values()
        ):
            filenames.update(category_files)

        split_files[split_name] = filenames

    train_val_overlap = (
        split_files["train"]
        & split_files["val"]
    )

    train_test_overlap = (
        split_files["train"]
        & split_files["test"]
    )

    val_test_overlap = (
        split_files["val"]
        & split_files["test"]
    )

    if train_val_overlap:
        raise ValueError(
            f"Train/val overlap detected: "
            f"{len(train_val_overlap)} files"
        )

    if train_test_overlap:
        raise ValueError(
            f"Train/test overlap detected: "
            f"{len(train_test_overlap)} files"
        )

    if val_test_overlap:
        raise ValueError(
            f"Val/test overlap detected: "
            f"{len(val_test_overlap)} files"
        )

    print("\nNo split overlap detected.")

def validate_source_files(selected: dict):

    missing_images = []
    missing_labels = []

    for output_split in ("train", "val", "test"):

        if output_split in ("train", "val"):
            source_split = "train"
        else:
            source_split = "test"

        source_image_dir = (
            RAW_ROOT
            / source_split
            / "images"
        )

        source_label_dir = (
            RAW_ROOT
            / source_split
            / "labels"
        )

        for category, filenames in (
            selected[output_split].items()
        ):

            for filename in filenames:

                image_path = (
                    source_image_dir
                    / filename
                )

                label_path = (
                    source_label_dir
                    / f"{Path(filename).stem}.txt"
                )

                if not image_path.exists():
                    missing_images.append(
                        str(image_path)
                    )

                if not label_path.exists():
                    missing_labels.append(
                        str(label_path)
                    )

    if missing_images:
        preview = "\n".join(
            missing_images[:10]
        )

        raise FileNotFoundError(
            f"\nMissing source images: "
            f"{len(missing_images)}\n"
            f"First missing files:\n{preview}"
        )

    if missing_labels:
        preview = "\n".join(
            missing_labels[:10]
        )

        raise FileNotFoundError(
            f"\nMissing source labels: "
            f"{len(missing_labels)}\n"
            f"First missing files:\n{preview}"
        )

    print(
        "All selected source image-label "
        "pairs exist."
    )

def prepare_output_directories():

    if SUBSET_ROOT.exists():
        raise FileExistsError(
            f"Subset directory already exists:\n"
            f"{SUBSET_ROOT}\n\n"
            f"Delete it manually if you really "
            f"want to rebuild the subset."
        )

    for split_name in ("train", "val", "test"):

        image_dir = (
            SUBSET_ROOT
            / split_name
            / "images"
        )

        label_dir = (
            SUBSET_ROOT
            / split_name
            / "labels"
        )

        image_dir.mkdir(
            parents=True,
            exist_ok=False
        )

        label_dir.mkdir(
            parents=True,
            exist_ok=False
        )

def copy_selected_files(selected: dict):

    copied_images = 0
    copied_labels = 0

    for output_split in ("train", "val", "test"):

        if output_split in ("train", "val"):
            source_split = "train"
        else:
            source_split = "test"

        source_image_dir = (
            RAW_ROOT
            / source_split
            / "images"
        )

        source_label_dir = (
            RAW_ROOT
            / source_split
            / "labels"
        )

        destination_image_dir = (
            SUBSET_ROOT
            / output_split
            / "images"
        )

        destination_label_dir = (
            SUBSET_ROOT
            / output_split
            / "labels"
        )

        for category, filenames in (
            selected[output_split].items()
        ):

            for filename in filenames:

                source_image = (
                    source_image_dir
                    / filename
                )

                source_label = (
                    source_label_dir
                    / f"{Path(filename).stem}.txt"
                )

                destination_image = (
                    destination_image_dir
                    / filename
                )

                destination_label = (
                    destination_label_dir
                    / source_label.name
                )

                shutil.copy2(
                    source_image,
                    destination_image
                )

                shutil.copy2(
                    source_label,
                    destination_label
                )

                copied_images += 1
                copied_labels += 1

    return copied_images, copied_labels

def verify_final_counts(selected: dict):

    print("\n" + "=" * 60)
    print("FINAL SUBSET VERIFICATION")
    print("=" * 60)

    grand_total = 0

    for split_name in ("train", "val", "test"):

        image_dir = (
            SUBSET_ROOT
            / split_name
            / "images"
        )

        label_dir = (
            SUBSET_ROOT
            / split_name
            / "labels"
        )

        actual_images = [
            path
            for path in image_dir.iterdir()
            if path.is_file()
        ]

        actual_labels = [
            path
            for path in label_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".txt"
        ]

        expected_count = sum(
            SPLIT_TARGETS[split_name].values()
        )

        actual_image_count = len(actual_images)
        actual_label_count = len(actual_labels)

        print(f"\n{split_name.upper()}")

        print(
            f"  Expected images : "
            f"{expected_count}"
        )

        print(
            f"  Actual images   : "
            f"{actual_image_count}"
        )

        print(
            f"  Actual labels   : "
            f"{actual_label_count}"
        )

        if actual_image_count != expected_count:
            raise ValueError(
                f"{split_name}: image count mismatch. "
                f"Expected={expected_count}, "
                f"Actual={actual_image_count}"
            )

        if actual_label_count != expected_count:
            raise ValueError(
                f"{split_name}: label count mismatch. "
                f"Expected={expected_count}, "
                f"Actual={actual_label_count}"
            )

        grand_total += actual_image_count

    expected_grand_total = sum(
        sum(split_counts.values())
        for split_counts in SPLIT_TARGETS.values()
    )

    if grand_total != expected_grand_total:
        raise ValueError(
            f"Grand total mismatch. "
            f"Expected={expected_grand_total}, "
            f"Actual={grand_total}"
        )

    print(
        f"\nTotal subset images: "
        f"{grand_total}"
    )


def save_subset_manifest(selected: dict):

    output_manifest = (
        SCRIPT_DIR
        / "reports"
        / "subset_manifest.json"
    )

    data = {
        "dataset": "D-Fire-small",
        "random_seed": RANDOM_SEED,
        "source_dataset": "D-Fire",
        "split_strategy": {
            "train": "sampled from original train",
            "val": "sampled from original train",
            "test": "sampled from original test"
        },
        "targets": SPLIT_TARGETS,
        "selected": selected
    }

    with output_manifest.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            indent=2
        )

    print(
        f"\nSubset manifest saved:\n"
        f"{output_manifest}"
    )


def print_selection_summary(selected: dict):

    print("\n" + "=" * 60)
    print("SELECTION SUMMARY")
    print("=" * 60)

    total_selected = 0

    for split_name in ("train", "val", "test"):

        print(f"\n{split_name.upper()}")

        split_total = 0

        for category, filenames in (
            selected[split_name].items()
        ):

            count = len(filenames)

            print(
                f"  {category:<25} "
                f"{count}"
            )

            split_total += count

        print(
            f"  {'TOTAL':<25} "
            f"{split_total}"
        )

        total_selected += split_total

    print(
        f"\nGrand total selected: "
        f"{total_selected}"
    )


def main():

    print("=" * 60)
    print("D-FIRE SMALL SUBSET CREATION")
    print("=" * 60)

    print(f"\nRaw dataset:")
    print(RAW_ROOT)

    print(f"\nManifest:")
    print(MANIFEST_JSON)

    print(f"\nOutput subset:")
    print(SUBSET_ROOT)


    manifest = load_manifest(
        MANIFEST_JSON
    )

    print("\nManifest loaded successfully.")

    print(
        "\nValidating manifest structure..."
    )

    validate_manifest_structure(
        manifest
    )

    print(
        "Manifest structure is valid."
    )

    print(
        "\nValidating target counts..."
    )

    validate_targets(
        manifest
    )

    print(
        "\nTarget validation passed."
    )

    print(
        "\nSampling subset..."
    )

    selected = sample_subset(
        manifest
    )

    print_selection_summary(
        selected
    )


    print(
        "\nChecking split overlap..."
    )

    verify_no_overlap(
        selected
    )


    print(
        "\nValidating source files..."
    )

    validate_source_files(
        selected
    )

    print(
        "\nCreating output directories..."
    )

    prepare_output_directories()

    print(
        "\nCopying selected files..."
    )

    copied_images, copied_labels = (
        copy_selected_files(selected)
    )

    print(
        f"Copied images: {copied_images}"
    )

    print(
        f"Copied labels: {copied_labels}"
    )


    verify_final_counts(
        selected
    )


    save_subset_manifest(
        selected
    )

    print("\n" + "=" * 60)
    print("SUBSET CREATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"\nCreated dataset:\n"
        f"{SUBSET_ROOT}"
    )


if __name__ == "__main__":
    main()
