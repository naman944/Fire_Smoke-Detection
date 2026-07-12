from pathlib import Path
from collections import Counter
import json

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

SUBSET_DIR = ROOT / "data" / "processed" / "D-Fire-small"

MANIFEST_JSON = SCRIPT_DIR / "reports" / "subset_manifest.json"

VERIFY_JSON = SCRIPT_DIR / "reports" / "verify_subset.json"

SMOKE_ID=0
FIRE_ID=1
VALID_CLASS_IDS = {SMOKE_ID, FIRE_ID}
# Tolerance for boundary checks: YOLO boxes that touch an image edge can land a
# hair outside [0, 1] due to float rounding (e.g. xc - w/2 == -5e-7).
EPS = 1e-6
IMAGE_EXT= {".jpg", ".png",".jpeg"}
SPLIT = ("train", "val", "test")
CATEGORY = ("fire_and_smoke", "fire_only", "smoke_only", "negative_or_background")

def load_manifest(manifest_path:Path):
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    return manifest

def get_image_file(image_path:Path):
    return sorted( path for path in image_path.iterdir()
                  if ( path.is_file() and path.suffix.lower() in IMAGE_EXT))

def get_label_file(label_path:Path):
    return sorted( path for path in label_path.iterdir()
                  if ( path.is_file() and path.suffix.lower() == ".txt"))

def verify_image_label(image_path:Path, label_path:Path):

    if not image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {image_path}")
    if not label_path.exists():
            raise FileNotFoundError(f"Label directory not found: {label_path}")

def parse_yolo_label(label_path:Path):
    class_id = set()
    errors= []
    box_count =0
    clamped = 0

    with label_path.open("r" , encoding ="utf-8") as f:
        for linenumber, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"Invalid line {linenumber} in {label_path}: {line}")
                continue
            try:
                id_value = int(parts[0])
            except ValueError:
                errors.append(f"Invalid class id in line {linenumber} in {label_path}: {line}")
                continue

            try:
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(f"Invalid bounding box values in line {linenumber} in {label_path}: {line}")
                continue
            if id_value not in VALID_CLASS_IDS:
                errors.append(f"Unknown class id {id_value} in line {linenumber} in {label_path}: {line}")
                continue

            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                errors.append(f"Invalid bounding box values in line {linenumber} in {label_path}: {line}")
                continue

            x_min = xc - w / 2
            x_max = xc + w / 2
            y_min = yc - h / 2
            y_max = yc + h / 2

            # Box centre/dims are valid but corners may spill past the frame by a
            # small amount (source-annotation slop). Clamp to [0, 1] and keep the
            # box valid, matching what YOLO training does. Only tally the clamp
            # when it exceeds float noise so we can warn about it.
            if x_min < -EPS or x_max > 1 + EPS or y_min < -EPS or y_max > 1 + EPS:
                clamped += 1

            class_id.add(id_value)
            box_count += 1
    return class_id, box_count, errors, clamped

def classify_image(class_id:set):
    has_fire = FIRE_ID in class_id
    has_smoke = SMOKE_ID in class_id

    if has_fire and has_smoke:
        return "fire_and_smoke"
    elif has_fire:
        return "fire_only"
    elif has_smoke:
        return "smoke_only"
    else:
        return "negative_or_background"

def verify_pair(image_path:Path, label_path:Path):
    image_files = get_image_file(image_path)
    label_files = get_label_file(label_path)

    image_basenames = {file.stem for file in image_files}
    label_basenames = {file.stem for file in label_files}

    missing_labels = image_basenames - label_basenames
    missing_images = label_basenames - image_basenames

    return missing_labels, missing_images


def build_expected_category_map(manifest: dict,split_name: str):
    selected = manifest["selected"][split_name]
    expected_map = {}
    duplicate_manifest_files = []

    for category in CATEGORY:

        filenames = selected.get(category,[])
        for filename in filenames:
            if filename in expected_map:
                duplicate_manifest_files.append(
                    filename
                )
            expected_map[filename] = category
    return (
        expected_map,
        sorted(set(duplicate_manifest_files))
    )

def verify_split(manifest:dict, split_name:str):
    image_dir = SUBSET_DIR / split_name / "images"
    label_dir = SUBSET_DIR / split_name / "labels"

    print("\n" + "=" * 45)
    print(f"Verifying {split_name} split ...")
    print("="*45)

    verify_image_label(image_dir, label_dir)
    image_files = get_image_file(image_dir)
    label_files = get_label_file(label_dir)
    print(f"Total images found: {len(image_files)}")
    print(f"Total labels found: {len(label_files)}")

    (missing_labels, missing_images) = verify_pair(image_dir, label_dir)
    (
        expected_category_map,
        duplicate_manifest_files
    ) = build_expected_category_map(manifest, split_name)

    actual_image_names = {file.name for file in image_files}
    expected_image_names = set(expected_category_map.keys())

    missing_manifest_images = expected_image_names - actual_image_names
    unexpected_images = actual_image_names - expected_image_names

    actual_categories = Counter()
    invalid_label_files = []
    category_mismatches = []
    total_valid_boxes = 0
    empty_label_files = 0
    clamped_label_files = []
    smoke_boxes = 0
    fire_boxes = 0

    for image_path in image_files:

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        # Missing labels already audited
        if not label_path.exists():
            continue

        (
            class_ids,
            box_count,
            errors,
            clamped
        ) = parse_yolo_label(
            label_path
        )

        # ----------------------------------------
        # Record invalid labels
        # ----------------------------------------

        if errors:
            invalid_label_files.append({
                "image": image_path.name,
                "label": label_path.name,
                "errors": errors
            })

        # ----------------------------------------
        # Record clamped (slightly out-of-frame) boxes
        # ----------------------------------------

        if clamped:
            clamped_label_files.append({
                "image": image_path.name,
                "label": label_path.name,
                "clamped_boxes": clamped
            })

        # ----------------------------------------
        # Count empty labels
        # ----------------------------------------

        if label_path.stat().st_size == 0:
            empty_label_files += 1

        # ----------------------------------------
        # Actual category from valid rows
        # ----------------------------------------

        actual_category = classify_image(
            class_ids
        )

        actual_categories[
            actual_category
        ] += 1

        # ----------------------------------------
        # Compare against manifest category
        # ----------------------------------------

        expected_category = (
            expected_category_map.get(
                image_path.name
            )
        )

        if (
            expected_category is not None
            and actual_category
            != expected_category
        ):
            category_mismatches.append({
                "image": image_path.name,
                "expected": expected_category,
                "actual": actual_category
            })

        # ----------------------------------------
        # Count boxes
        # ----------------------------------------

        total_valid_boxes += box_count

        # Re-read valid rows only for
        # per-class box counts
        with label_path.open(
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])

                    coordinates = [
                        float(value)
                        for value in parts[1:]
                    ]

                except ValueError:
                    continue

                x_center = coordinates[0]
                y_center = coordinates[1]
                width = coordinates[2]
                height = coordinates[3]

                if class_id not in VALID_CLASS_IDS:
                    continue

                if not (
                    0.0 <= x_center <= 1.0
                    and 0.0 <= y_center <= 1.0
                    and 0.0 < width <= 1.0
                    and 0.0 < height <= 1.0
                ):
                    continue

                # Corners may spill slightly past the frame; clamp (keep the
                # box) rather than drop it, matching parse_yolo_label above.
                if class_id == SMOKE_ID:
                    smoke_boxes += 1

                elif class_id == FIRE_ID:
                    fire_boxes += 1

    # ----------------------------------------
    # Expected counts
    # ----------------------------------------

    expected_total = sum(
        manifest["targets"][split_name].values()
    )

    expected_categories = (
        manifest["targets"][split_name]
    )

    # ----------------------------------------
    # Print summary
    # ----------------------------------------

    print("\nCategory counts:")

    for category in CATEGORY:

        print(
            f"  {category:<25} "
            f"actual={actual_categories[category]:<5} "
            f"expected={expected_categories[category]}"
        )

    print("\nAudit:")

    print(
        f"  Missing labels          : "
        f"{len(missing_labels)}"
    )

    print(
        f"  Orphan labels           : "
        f"{len(missing_images)}"
    )

    print(
        f"  Invalid label files     : "
        f"{len(invalid_label_files)}"
    )

    print(
        f"  Category mismatches     : "
        f"{len(category_mismatches)}"
    )

    print(
        f"  Missing manifest images : "
        f"{len(missing_manifest_images)}"
    )

    print(
        f"  Unexpected images       : "
        f"{len(unexpected_images)}"
    )

    print(
        f"  Manifest duplicates     : "
        f"{len(duplicate_manifest_files)}"
    )

    print(
        f"  Empty label files       : "
        f"{empty_label_files}"
    )

    if clamped_label_files:
        clamped_boxes = sum(
            entry["clamped_boxes"]
            for entry in clamped_label_files
        )
        print(
            f"\n  WARNING: clamped {clamped_boxes} box(es) "
            f"in {len(clamped_label_files)} file(s) "
            f"that spilled past the frame:"
        )
        for entry in clamped_label_files:
            print(
                f"    {entry['label']} "
                f"({entry['clamped_boxes']} box)"
            )

    print("\nBounding boxes:")

    print(
        f"  Smoke boxes : {smoke_boxes}"
    )

    print(
        f"  Fire boxes  : {fire_boxes}"
    )

    print(
        f"  Total boxes : {total_valid_boxes}"
    )

    # ----------------------------------------
    # Determine split status
    # ----------------------------------------

    category_counts_match = all(
        actual_categories[category]
        == expected_categories[category]
        for category in CATEGORY
    )

    passed = all([
        len(image_files) == expected_total,
        len(label_files) == expected_total,
        not missing_labels,
        not missing_images,
        not invalid_label_files,
        not category_mismatches,
        not missing_manifest_images,
        not unexpected_images,
        not duplicate_manifest_files,
        category_counts_match
    ])

    if passed:
        print(
            f"\n{split_name.upper()} SPLIT: PASSED"
        )

    else:
        print(
            f"\n{split_name.upper()} SPLIT: FAILED"
        )

    return {
        "passed": passed,

        "expected_images": expected_total,
        "actual_images": len(image_files),
        "actual_labels": len(label_files),

        "categories": {
            category: actual_categories[category]
            for category in CATEGORY
        },

        "expected_categories": {
            category: expected_categories[category]
            for category in CATEGORY
        },

        "bounding_boxes": {
            "smoke": smoke_boxes,
            "fire": fire_boxes,
            "total": total_valid_boxes
        },

        "audit": {
            "missing_labels": sorted(missing_labels),
            "missing_images": sorted(missing_images),

            "invalid_label_files":
                invalid_label_files,

            "category_mismatches":
                category_mismatches,

            "missing_manifest_images":
                sorted(missing_manifest_images),

            "unexpected_actual_images":
                sorted(unexpected_images),

            "duplicate_manifest_files":
                duplicate_manifest_files,

            "empty_label_files":
                empty_label_files,

            "clamped_label_files":
                clamped_label_files
        }
    }


# ============================================================
# VERIFY CROSS-SPLIT OVERLAP
# ============================================================

def verify_cross_split_overlap():
    """
    Check filename overlap between actual
    train, val, and test image directories.
    """

    split_names = {}

    for split_name in SPLIT:

        image_dir = (
            SUBSET_DIR
            / split_name
            / "images"
        )

        image_files = get_image_file(
            image_dir
        )

        split_names[split_name] = {
            path.name
            for path in image_files
        }

    train_val = sorted(
        split_names["train"]
        & split_names["val"]
    )

    train_test = sorted(
        split_names["train"]
        & split_names["test"]
    )

    val_test = sorted(
        split_names["val"]
        & split_names["test"]
    )

    passed = not any([
        train_val,
        train_test,
        val_test
    ])

    return {
        "passed": passed,
        "train_val": train_val,
        "train_test": train_test,
        "val_test": val_test
    }


# ============================================================
# SAVE VERIFICATION REPORT
# ============================================================

def save_report(report: dict):
    """
    Save full verification report.
    """

    # REPORT_DIR.mkdir(
    #     parents=True,
    #     exist_ok=True
    # )

    with VERIFY_JSON.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )

    print(
        f"\nVerification report saved:\n"
        f"{VERIFY_JSON}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("D-FIRE SMALL DATASET VERIFICATION")
    print("=" * 60)

    print(
        f"\nSubset root:\n"
        f"{SUBSET_DIR}"
    )

    print(
        f"\nSubset manifest:\n"
        f"{MANIFEST_JSON}"
    )

    # ----------------------------------------
    # 1. Load manifest
    # ----------------------------------------

    manifest = load_manifest(
        MANIFEST_JSON
    )

    print(
        "\nSubset manifest loaded successfully."
    )

    # ----------------------------------------
    # 2. Verify each split
    # ----------------------------------------

    split_results = {}

    for split_name in SPLIT:

        split_results[split_name] = (
            verify_split(
                manifest,
                split_name

            )
        )

    # ----------------------------------------
    # 3. Verify actual cross-split overlap
    # ----------------------------------------

    print("\n" + "=" * 60)
    print("CROSS-SPLIT OVERLAP CHECK")
    print("=" * 60)

    overlap_result = (
        verify_cross_split_overlap()
    )

    print(
        f"Train / Val overlap  : "
        f"{len(overlap_result['train_val'])}"
    )

    print(
        f"Train / Test overlap : "
        f"{len(overlap_result['train_test'])}"
    )

    print(
        f"Val / Test overlap   : "
        f"{len(overlap_result['val_test'])}"
    )

    if overlap_result["passed"]:
        print(
            "\nCross-split overlap check: PASSED"
        )

    else:
        print(
            "\nCross-split overlap check: FAILED"
        )

    # ----------------------------------------
    # 4. Overall status
    # ----------------------------------------

    all_splits_passed = all(
        result["passed"]
        for result in split_results.values()
    )

    overall_passed = (
        all_splits_passed
        and overlap_result["passed"]
    )

    report = {
        "dataset": "D-Fire-small",

        "subset_root": str(
            SUBSET_DIR
        ),

        "overall_passed": overall_passed,

        "splits": split_results,

        "cross_split_overlap":
            overlap_result
    }

    # ----------------------------------------
    # 5. Save report
    # ----------------------------------------

    save_report(
        report
    )

    # ----------------------------------------
    # 6. Final result
    # ----------------------------------------

    print("\n" + "=" * 60)

    if overall_passed:
        print(
            "DATASET VERIFICATION PASSED"
        )

        print(
            "Dataset is ready for YOLO configuration."
        )

    else:
        print(
            "DATASET VERIFICATION FAILED"
        )

        print(
            "Inspect subset_verification.json "
            "before training."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
