#!/usr/bin/env python3
import os
import json
import random
import shutil

def main():
    # === Configure Paths ===
    BASE_DIR = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/datasets"
    IMAGES_DIR = os.path.join(BASE_DIR, "prepped")        # Folder with all images
    ANNOTATIONS_FILE = os.path.join(BASE_DIR, "result.json")  
    TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, "images/train")
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, "images/val")
    TRAIN_ANNOTATIONS_FILE = os.path.join(BASE_DIR, "annotations/train.json")
    VAL_ANNOTATIONS_FILE = os.path.join(BASE_DIR, "annotations/val.json")

    # === Split Ratio ===
    SPLIT_RATIO = 0.8

    # === Create output directories ===
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "annotations"), exist_ok=True)

    # === Load the original COCO annotations ===
    with open(ANNOTATIONS_FILE, "r") as f:
        coco_data = json.load(f)

    # === Shuffle and split the dataset ===
    all_images = coco_data["images"]
    random.shuffle(all_images)
    split_idx = int(len(all_images) * SPLIT_RATIO)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    # === Map image IDs to their annotations ===
    image_id_to_annotations = {img["id"]: [] for img in coco_data["images"]}
    for ann in coco_data["annotations"]:
        image_id_to_annotations[ann["image_id"]].append(ann)

    # === Helper function to copy + write JSON ===
    def prepare_annotations(image_list, out_json_file, output_images_dir):
        subset_annotations = {
            "images": [],
            "annotations": [],
            "categories": coco_data["categories"],
        }

        for img_dict in image_list:
            # Use os.path.basename to avoid any subfolder in file_name
            raw_name = img_dict["file_name"]        # e.g., "images/1e244d1b.jpg" or just "1e244d1b.jpg"
            file_name = os.path.basename(raw_name)  # e.g., "1e244d1b.jpg"

            src_path = os.path.join(IMAGES_DIR, file_name)
            dst_path = os.path.join(output_images_dir, file_name)

            if not os.path.isfile(src_path):
                print(f"Warning: Source image not found: {src_path}")
                continue

            # Copy the file
            shutil.copy(src_path, dst_path)

            # Add the relevant data to subset
            # (We should still store the original `img_dict` if it doesn't break anything else, 
            #  but we might want to update file_name to the 'basename' so that the new JSON is consistent.)
            updated_img_dict = dict(img_dict)
            updated_img_dict["file_name"] = file_name

            subset_annotations["images"].append(updated_img_dict)
            subset_annotations["annotations"].extend(image_id_to_annotations[img_dict["id"]])

        # Write the JSON
        with open(out_json_file, "w") as f:
            json.dump(subset_annotations, f, indent=4)

    # === Prepare training & validation sets ===
    prepare_annotations(train_images, TRAIN_ANNOTATIONS_FILE, TRAIN_IMAGES_DIR)
    prepare_annotations(val_images, VAL_ANNOTATIONS_FILE, VAL_IMAGES_DIR)

    # === Print summary ===
    print("Dataset split complete!")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")


if __name__ == "__main__":
    main()
