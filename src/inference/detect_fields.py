import os
import cv2
import torch
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
    
    if use_gpu and torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        if torch.cuda.get_device_capability()[0] >= 7:
            cfg.MODEL.FP16_ENABLED = True
    else:
        cfg.MODEL.DEVICE = "cpu"
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
    
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = [
        "Address", "Date of Birth", "Dr Info", "Given Name", "Medicare Number", 
        "Phone Number", "Request Date", "Sex", "Surname"
    ]
    
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

# -----------------------------------------------------------------------------
# 3. Optimized Inference and Extraction
# -----------------------------------------------------------------------------

def process_instances(instances: Instances, class_names: list) -> Dict[str, Tuple[Tuple[float, float, float, float], float]]:
    """Process instances efficiently using vectorized operations."""
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    
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
        class_names[class_idx]: (tuple(best_boxes[i]), max_scores[i])
        for i, class_idx in enumerate(unique_classes)
    }

def extract_regions(
    image_path: str,
    predictor: DefaultPredictor,
    cfg: get_cfg,
    min_score: float = 0.5,
    preloaded_image: Optional[np.ndarray] = None
) -> Dict[str, Tuple[float, float, float, float]]:
    """Extract regions with optimized processing."""
    image = preloaded_image if preloaded_image is not None else cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    with torch.no_grad():
        outputs = predictor(image)
    
    instances = outputs["instances"].to(predictor.model.device)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    
    predictions = process_instances(instances, metadata.thing_classes)
    
    return {
        category: bbox
        for category, (bbox, score) in predictions.items()
        if score >= min_score
    }

def batch_extract_regions(
    image_paths: List[str],
    predictor: DefaultPredictor,
    cfg: get_cfg,
    batch_size: int = 4
) -> Dict[str, Dict[str, Tuple[float, float, float, float]]]:
    """Process multiple images in batches for improved throughput."""
    results = {}
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [cv2.imread(path) for path in batch_paths]
        
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
    predictor, cfg = setup_model(use_gpu=True)
    
    image_path = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/datasets/prepped/images/0aa98922-SKM_C224e25012613090_0077.jpg"
    extracted_data = extract_regions(image_path, predictor, cfg)
    print(extracted_data)
