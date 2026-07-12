from pathlib import Path
from collections import Counter
import json

PARENT_DIR = Path(__file__).resolve().parent.parent

ROOT = PARENT_DIR / "data" / "raw" / "D-Fire"
REPORT_DIR = PARENT_DIR / "scripts" / "reports"
ANALYSIS_JSON = REPORT_DIR / "dataset_analysis.json"
MANIFEST_JSON = REPORT_DIR / "dataset_manifest.json"

SMOKE_ID=0
FIRE_ID=1

IMAGE_EXT= {".jpg", ".png",".jpeg"}


# Function : Read one class file from one YOLO label File
#
def read_categories(label_path:Path):
    class_id=set()

    if not label_path.exists():
        return class_id
    with label_path.open("r", encoding ="utf-8") as f:
        for linenumber , line in enumerate(f,start =1):
            line =line.strip()
            if not line :
                continue
            parts =line.split() # give a list of strings
            if len(parts) !=5:
                print(f"Invalid line {linenumber} in {label_path}: {line}")
                continue
            try :
                id_values =int(parts[0])
            except ValueError:
                print(f"Invalid class id in line {linenumber} in {label_path}:{line}")
                continue
            class_id.add(id_values)
    return class_id


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


def analyze_split(split_name:str):
    image_dir = ROOT / split_name /"images"
    label_dir = ROOT / split_name / "labels"

    print(f"Analyzing {split_name} split ...")
    print(f"Image dir :{image_dir}")
    print(f"Label dir :{label_dir}")

    count= Counter()
    manifest = {
        "fire_only":[],
        "smoke_only":[],
        "fire_and_smoke":[],
        "negative_or_background":[]
    }
    audit_manifest={
        "missing_label_files":[],
        "unknown_class_images":[],
    }
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist:{image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    image_files =[
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXT
    ]


    missing_labels_files=0
    unknown_class_image =0
    unknown_class_id =Counter()

    for path in image_files:
        label_path = label_dir / f"{path.stem}.txt"

        if not label_path.exists():
            missing_labels_files +=1
            audit_manifest["missing_label_files"].append(path.name)
            continue
        class_id = read_categories(label_path)
        unexpected_id = {id for id in class_id if id not in (SMOKE_ID,FIRE_ID)}
        if unexpected_id:
            unknown_class_image +=1
            unknown_class_id.update(unexpected_id)
            audit_manifest["unknown_class_images"].append({
                "image":path.name,
                "unknown_id": sorted(unexpected_id)
            })
            continue
        category = classify_image(class_id)

        count[category] +=1
        manifest[category].append(path.name)

    total_images=len(image_files)

    print("\n"+"="*40)
    print(f"D_FIRE Dataset Analysis for {split_name.upper()} split")
    print("="*40)

    print(f"Total images : {total_images}")
    print(f"FIRE_ONLY images : {count['fire_only']}")
    print(f"SMOKE_ONLY images : {count['smoke_only']}")
    print(f"FIRE_AND_SMOKE images : {count['fire_and_smoke']}")
    print(f"NEGATIVE_OR_BACKGROUND images : {count['negative_or_background']}")

    print("-"*40)

    print("Audit Report:")
    print(f"Missing label files : {missing_labels_files}")
    print(f"Images with unknown class ids : {unknown_class_image}")

    if unknown_class_id:
        print("Unknown class ids found:")
        for id , count in unknown_class_id.items():
            print(f"Class ID {id}: {count} occurrences")

    print("\n Sanity check:")
    print("-"*40)

    categories_total = sum(count.values())
    if categories_total + missing_labels_files+ unknown_class_image == total_images:
        print("Sanity check passed: Total images match sum of categories, missing labels, and unknown class ids.")
    else:
        print("Sanity check failed: Total images do not match sum of categories, missing labels, and unknown class ids.")
    return {
        "total_images": total_images,
        "categories":{
            "fire_only": count["fire_only"],
            "smoke_only": count["smoke_only"],
            "fire_and_smoke": count["fire_and_smoke"],
            "negative_or_background": count["negative_or_background"]
        },
        "audit":{
            "missing_label_files": missing_labels_files,
            "unknown_class_images": unknown_class_image,
            "unknown_class_ids": dict(unknown_class_id)
        },
        "manifest": manifest,
        "audit_manifest": audit_manifest

    }

def main():
    print("="*65)
    print("D-FIRE Dataset Analysis")
    print("="*65)

    train_result = analyze_split("train")
    test_result = analyze_split("test")

    analysis_result ={
        "dataset":"D-Fire Dataset",
        "class_mapping":{
            "0":"smoke",
            "1":"fire"
        },
        "train":{
            "total_images": train_result["total_images"],
            "categories": train_result["categories"],
            "audit": train_result["audit"]
        },
        "test":{
            "total_images": test_result["total_images"],
            "categories": test_result["categories"],
            "audit": test_result["audit"]
        },
        "combined":{
            "total_images":train_result["total_images"] + test_result["total_images"],
            "categories":{
                "fire_only": train_result["categories"]["fire_only"] + test_result["categories"]["fire_only"],
                "smoke_only": train_result["categories"]["smoke_only"] + test_result["categories"]["smoke_only"],
                "fire_and_smoke": train_result["categories"]["fire_and_smoke"] + test_result["categories"]["fire_and_smoke"],
                "negative_or_background": train_result["categories"]["negative_or_background"] + test_result["categories"]["negative_or_background"]
            },

        }
    }

    category_manifest={
        "dataset":"D-Fire Dataset",
        "train":train_result["manifest"],
        "test":test_result["manifest"],
        "audit":{
            "train":train_result["audit_manifest"],
            "test":test_result["audit_manifest"]
        }
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with ANALYSIS_JSON.open("w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2)
    with MANIFEST_JSON.open("w", encoding="utf-8") as f:
        json.dump(category_manifest, f, indent=4)
    print("\n JSON FILE SAVED SUCCESSFULLY")
    print(f"Analysis Report :{ANALYSIS_JSON}")
    print(f"Manifest Report :{MANIFEST_JSON}")


if __name__ == "__main__":
    main()
