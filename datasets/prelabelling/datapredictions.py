import os
import glob
import json
import cv2
import uuid

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

class MyModel:
    def __init__(self, model_path, config_path):
        self.model = self.load_model(model_path, config_path)
    
    def load_model(self, model_path, config_path):
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cpu"
        return DefaultPredictor(cfg)
    
    def predict_image(self, image):
        """
        Run inference on a single image and return a list of annotation objects
        formatted in Label Studio's expected schema.
        """
        outputs = self.model(image)
        instances = outputs["instances"].to("cpu")
        annotations = []
        img_height, img_width = image.shape[:2]
        for i in range(len(instances.pred_boxes)):
            bbox_arr = instances.pred_boxes[i].tensor.numpy()[0]
            x1, y1, x2, y2 = float(bbox_arr[0]), float(bbox_arr[1]), float(bbox_arr[2]), float(bbox_arr[3])
            category_id = int(instances.pred_classes[i])
            score = float(instances.scores[i])
            annotation = {
                "id": str(uuid.uuid4()),
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": (x1 / img_width) * 100,
                    "y": (y1 / img_height) * 100,
                    "width": ((x2 - x1) / img_width) * 100,
                    "height": ((y2 - y1) / img_height) * 100,
                    "rectanglelabels": [f"Class {category_id}"]
                },
                "score": score
            }
            annotations.append(annotation)
        return annotations

# Paths for the model and config files
model_path = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/trained/model_final.pth"
config_path = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/scripts/config_home.yaml"

# Instantiate the model
my_model = MyModel(model_path, config_path)

# Folder containing images to be processed
images_folder = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/datasets/images/prepped/"
# Get list of image file paths (adjust pattern if needed)
image_files = glob.glob(os.path.join(images_folder, "*"))

# Base URL where your images are hosted.
base_url = "http://localhost:8000/"

tasks = []
for image_file in image_files:
    image = cv2.imread(image_file)
    if image is None:
        print(f"Warning: Could not read image: {image_file}")
        continue

    # Get annotations for the image
    annotation_results = my_model.predict_image(image)
    
    # Build an absolute URL using the image's basename
    file_name = os.path.basename(image_file)
    image_url = base_url + file_name
    
    # Build a task dictionary in Label Studio's JSON schema
    task = {
        "data": {
            "image": image_url
        },
        "predictions": [
            {
                "model_version": "v1.0",
                "result": annotation_results
            }
        ]
    }
    tasks.append(task)

# Save the tasks to a JSON file
output_json_path = "prelabeled_tasks.json"
with open(output_json_path, "w") as f:
    json.dump(tasks, f, indent=4)

print(f"Pre-labeled tasks saved to {output_json_path}")
