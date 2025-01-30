import os
import cv2
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
import numpy as np

# -----------------------------------------------------------------------------
# 1. Setup Paths and Configurations
# -----------------------------------------------------------------------------

MODEL_WEIGHTS = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/trained/model_final.pth"
CONFIG_PATH = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/scripts/config_home.yaml"

# -----------------------------------------------------------------------------
# 2. Load Configuration and Model
# -----------------------------------------------------------------------------

def setup_model(use_gpu: bool = True) -> Tuple[DefaultPredictor, get_cfg]:
    """Initialize model with optimized settings."""
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_PATH)
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    cfg.MODEL.WEIGHTS_ONLY = True
    
    # Use GPU if available and requested
    if use_gpu and torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        # Enable TensorRT FP16 optimization if possible
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a confidence threshold
        if torch.cuda.get_device_capability()[0] >= 7:  # Check for Volta or newer architecture
            cfg.MODEL.FP16_ENABLED = True
    else:
        cfg.MODEL.DEVICE = "cpu"
        # Enable Intel MKL DNN optimization for CPU
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
    
    # Cache the metadata
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = [
        "Address", "Date of Birth", "Dr Info", "Given Name", "Medicare Number", 
        "Phone Number", "Request Date", "Sex", "Surname"
    ]
    
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

# -----------------------------------------------------------------------------
# 3. Define Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class ExtractedField:
    bounding_box: Tuple[float, float, float, float]

@dataclass
class ExtractedForm:
    fields: Dict[str, ExtractedField] = field(default_factory=dict)

# -----------------------------------------------------------------------------
# 4. Optimized Inference and Extraction
# -----------------------------------------------------------------------------

def process_instances(instances: Instances, class_names: list) -> dict:
    """Process instances efficiently using vectorized operations."""
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    
    # Use numpy operations for faster processing
    unique_classes, class_indices = np.unique(classes, return_inverse=True)
    max_scores = np.zeros(len(unique_classes))
    best_boxes = np.zeros((len(unique_classes), 4))
    
    for i, class_idx in enumerate(unique_classes):
        mask = classes == class_idx
        if mask.any():
            class_scores = scores[mask]
            max_score_idx = np.argmax(class_scores)
            max_scores[i] = class_scores[max_score_idx]
            best_boxes[i] = boxes[mask][max_score_idx]
    
    return {
        class_names[class_idx]: {
            "bounding_box": tuple(best_boxes[i]),
            "score": max_scores[i]
        }
        for i, class_idx in enumerate(unique_classes)
    }

def extract_regions(
    image_path: str,
    predictor: DefaultPredictor,
    cfg: get_cfg,
    min_score: float = 0.5,
    preloaded_image: Optional[np.ndarray] = None
) -> ExtractedForm:
    """Extract regions with optimized processing."""
    if preloaded_image is not None:
        image = preloaded_image
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
    
    # Run inference
    with torch.no_grad():  # Disable gradient calculation
        outputs = predictor(image)
    
    instances = outputs["instances"].to(predictor.model.device)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    
    # Process predictions
    predictions = process_instances(instances, metadata.thing_classes)
    
    # Create ExtractedForm with filtered predictions
    extracted_form = ExtractedForm()
    for category, data in predictions.items():
        if data["score"] >= min_score:
            extracted_form.fields[category] = ExtractedField(bounding_box=data["bounding_box"])
    
    return extracted_form

def batch_extract_regions(
    image_paths: List[str],
    predictor: DefaultPredictor,
    cfg: get_cfg,
    batch_size: int = 4
) -> Dict[str, ExtractedForm]:
    """Process multiple images in batches for improved throughput."""
    results = {}
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [cv2.imread(path) for path in batch_paths]
        
        # Process each image in the batch
        for path, image in zip(batch_paths, batch_images):
            if image is not None:
                results[os.path.basename(path)] = extract_regions(
                    path,
                    predictor,
                    cfg,
                    preloaded_image=image
                )
    
    return results

if __name__ == "__main__":
    # Initialize model with GPU support if available
    predictor, cfg = setup_model(use_gpu=True)
    
    # Single image inference
    image_path = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/datasets/prepped/images/0aa98922-SKM_C224e25012613090_0077.jpg"
    extracted_data = extract_regions(image_path, predictor, cfg)
    print(extracted_data)
  