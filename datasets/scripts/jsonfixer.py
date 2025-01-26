import json

# Load your COCO JSON file
with open("/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/datasets/annotations/val.json", "r") as f:
    coco_data = json.load(f)

# Convert all image IDs to strings
for img in coco_data["images"]:
    img["id"] = str(img["id"])  # Ensure IDs are strings

for ann in coco_data["annotations"]:
    ann["image_id"] = str(ann["image_id"])  # Ensure IDs match the `images`

# Save the fixed JSON
with open("/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/datasets/annotations/val.json", "w") as f:
    json.dump(coco_data, f, indent=4)
