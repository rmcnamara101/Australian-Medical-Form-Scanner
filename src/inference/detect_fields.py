# --------------------------------------------------------------------------------------------------------------------------
# This script is used to perform inference on a set of images and extract the bounding boxes of the detected fields.
#
# This script is designed to take as an input, a cropped image of a form, and will outpt a dictionary containing the
# field region information. To structure this data two new dataclasses are introduced. The ExtractedField class and
# the ExtractedForm class. The ExtractedField class contains the bounding box of the detected field, and the ExtractedForm
# class contains a dictionary of fields detected in the image.
#
# The output of this script is a dictionary where the key is the image name and the value is the ExtractedForm object.
# The output will be in the form:
#
# output = { 'image_name.jpg': 
#                               ExtractedForm(
#                                               fields={'field_name': [ExtractedField(bounding_box=(x1, y1, x2, y2))]
#                                                                                                                   } 
#                                                                                                                    }
#
# --------------------------------------------------------------------------------------------------------------------------

import os
import cv2
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# -----------------------------------------------------------------------------
# 1. Setup Paths and Configurations
# -----------------------------------------------------------------------------


MODEL_WEIGHTS = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/trained/model_final.pth"  # Update with trained model path
CONFIG_PATH = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/scripts/config_home.yaml"  # Update with model config path



# -----------------------------------------------------------------------------
# 2. Load Configuration and Model
# -----------------------------------------------------------------------------


cfg = get_cfg()
cfg.merge_from_file(CONFIG_PATH)
cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
cfg.MODEL.WEIGHTS_ONLY = True
cfg.MODEL.DEVICE = "cpu"  # Change to "cuda" if using a GPU

predictor = DefaultPredictor(cfg)

# Set class names manually if metadata is empty
MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = [
    "Address", "Date of Birth", "Dr Info", "Given Name", "Medicare Number", 
    "Phone Number", "Request Date", "Sex", "Surname"
]



# -----------------------------------------------------------------------------
# 3. Define Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class ExtractedField:
    bounding_box: Tuple[float, float, float, float]

@dataclass
class ExtractedForm:
    fields: Dict[str, List[ExtractedField]] = field(default_factory=dict)



# -----------------------------------------------------------------------------
# 4. Perform Inference and Extract Regions
# -----------------------------------------------------------------------------
def extract_regions(image_path: str, predictor, cfg) -> ExtractedForm:
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
    scores = instances.scores.numpy() if instances.has("scores") else None
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None
    
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    class_names = metadata.thing_classes
    
    extracted_form = ExtractedForm()
    category_best_predictions = {}
    
    for i in range(len(classes)):
        category = class_names[classes[i]]
        bbox = tuple(boxes[i].tolist())
        score = scores[i]
        
        if category not in category_best_predictions or score > category_best_predictions[category]["score"]:
            category_best_predictions[category] = {"bounding_box": bbox, "score": score}
    
    for category, data in category_best_predictions.items():
        extracted_form.fields[category] = ExtractedField(bounding_box=data["bounding_box"])
    
    return extracted_form

if __name__ == "__main__":
    image_dir = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/datasets/prepped/images"
    extracted_data = extract_regions(image_dir, predictor, cfg)
    print(extracted_data)
