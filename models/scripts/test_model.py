import os
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# -----------------------------------------------------------------------------
# 1. Setup Paths and Configurations
# -----------------------------------------------------------------------------
# Path to your trained model
MODEL_WEIGHTS = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/trained/model_final.pth"  # Update with the path to your downloaded model
CONFIG_PATH = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/scripts/config.yaml"  # Update with the path to your config.yaml

# Path to test images
TEST_IMAGES_DIR = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/datasets/images/test"  # Update with the folder containing test images

# Output directory for results
OUTPUT_DIR = "./output/test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Register Dataset (Optional: Only if COCO format is needed)
# -----------------------------------------------------------------------------
# If your test images are annotated in COCO format, register them
# Uncomment and modify the following block if applicable
# register_coco_instances("my_test_dataset", {}, "/path/to/test.json", TEST_IMAGES_DIR)
# MetadataCatalog.get("my_test_dataset").thing_classes = [
#     "Address", "Date of Birth", "Dr Info", "Given Name", "Medicare Number", 
#     "Phone Number", "Request Date", "Sex", "Surname"
# ]

# -----------------------------------------------------------------------------
# 3. Load Configuration and Model
# -----------------------------------------------------------------------------
cfg = get_cfg()
cfg.merge_from_file(CONFIG_PATH)
cfg.MODEL.WEIGHTS = MODEL_WEIGHTS  # Path to the trained model
cfg.MODEL.DEVICE = "cpu"  # Change to "cuda" if running on GPU

predictor = DefaultPredictor(cfg)

# -----------------------------------------------------------------------------
# 4. Perform Inference on Test Images

def run_inference_on_images(image_dir, output_dir, predictor, cfg):
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read image {image_name}. Skipping.")
            continue

        # Perform inference
        outputs = predictor(image)

        # Extract predictions
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
        scores = instances.scores.numpy() if instances.has("scores") else None
        classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None

        # Get metadata
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])  # Use training metadata
        class_names = metadata.thing_classes  # List of category names

        # Keep only the highest-confidence prediction per category
        category_best_predictions = {}  # Dictionary to store the best prediction per category
        for i in range(len(classes)):
            category = class_names[classes[i]]  # Get category name
            if category not in category_best_predictions or scores[i] > category_best_predictions[category]["score"]:
                category_best_predictions[category] = {
                    "box": boxes[i],
                    "score": scores[i]
                }

        # Create a new list of filtered predictions
        filtered_boxes = []
        filtered_scores = []
        filtered_classes = []
        
        for category, data in category_best_predictions.items():
            filtered_boxes.append(data["box"])
            filtered_scores.append(data["score"])
            filtered_classes.append(class_names.index(category))

        # Convert filtered predictions back to tensor format
        instances.pred_boxes.tensor = torch.tensor(filtered_boxes)
        instances.scores = torch.tensor(filtered_scores)
        instances.pred_classes = torch.tensor(filtered_classes)

        # Visualize results
        v = Visualizer(image[:, :, ::-1], metadata, scale=0.5)
        v = v.draw_instance_predictions(instances)

        # Save the visualization
        output_path = os.path.join(output_dir, f"result_{image_name}")
        cv2.imwrite(output_path, v.get_image()[:, :, ::-1])
        print(f"Saved filtered result to {output_path}")


# Set class names manually if metadata is empty
MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = [
    "Address", "Date of Birth", "Dr Info", "Given Name", 
    "Medicare Number", "Phone Number", "Request Date", "Sex", "Surname"
]

# Run inference
run_inference_on_images(TEST_IMAGES_DIR, OUTPUT_DIR, predictor, cfg)
