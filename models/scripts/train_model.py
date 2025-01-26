import os

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# -----------------------------------------------------------------------------
# 1. Setup Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The root "datasets" folder you mentioned
DATASET_DIR = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/datasets"

# Paths to your COCO-JSON annotations
TRAIN_JSON = os.path.join(DATASET_DIR, "annotations", "train.json")
VAL_JSON   = os.path.join(DATASET_DIR, "annotations", "val.json")

# Paths to your image folders
TRAIN_IMAGES = os.path.join(DATASET_DIR, "images", "train")
VAL_IMAGES   = os.path.join(DATASET_DIR, "images", "val")

# Path to config.yaml (the one referencing the model zoo)
CONFIG_PATH  = os.path.join(BASE_DIR, "config.yaml")

# Output directory (where checkpoints & logs will go)
# You had a "MODELS_DIR" but let's define an explicit output directory:
OUTPUT_DIR = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/trained"

# -----------------------------------------------------------------------------
# 2. Register the Datasets with Detectron2
# -----------------------------------------------------------------------------
# We'll use "my_train_dataset" and "my_val_dataset" to match your config.yaml
register_coco_instances("my_train_dataset", {}, TRAIN_JSON, TRAIN_IMAGES)
register_coco_instances("my_val_dataset", {}, VAL_JSON, VAL_IMAGES)


MetadataCatalog.get("my_train_dataset").thing_classes = [
    "Address", "Date of Birth", "Dr Info", "Given Name", "Medicare Number", "Phone Number", "Request Date", "Sex", "Surname"
]

MetadataCatalog.get("my_val_dataset").thing_classes = [
    "Address", "Date of Birth", "Dr Info", "Given Name", "Medicare Number", "Phone Number", "Request Date", "Sex", "Surname"
]


# -----------------------------------------------------------------------------
# 3. Create & Customize the Detectron2 Config
# -----------------------------------------------------------------------------
def build_config():
    cfg = get_cfg()
    # Merge from your custom config.yaml, which inherits from model zoo
    cfg.merge_from_file(CONFIG_PATH)

    # If your config.yaml references model zoo weights, it will download them. 
    # If you want to override with a local model, do so here, e.g.:
    # cfg.MODEL.WEIGHTS = "/path/to/some/pretrained_model.pkl"

    cfg.DATASETS.TRAIN = ("my_train_dataset",)
    cfg.DATASETS.TEST = ("my_val_dataset",)

    # Ensure the number of classes matches your dataset
    # If config.yaml already has `MODEL.ROI_HEADS.NUM_CLASSES = 9`, 
    # this line might be redundant. But let's be explicit.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9

    # Adjust training settings if needed
    # e.g., number of iterations, base LR, batch size, etc.
    # If already set in config.yaml, you can skip or override them here:
    # cfg.SOLVER.MAX_ITER = 2000
    # cfg.SOLVER.BASE_LR = 0.001

    # Where to store training logs and checkpoints:
    cfg.OUTPUT_DIR = OUTPUT_DIR

    cfg.MODEL.DEVICE = "mps"

    return cfg

# -----------------------------------------------------------------------------
# 4. Main Training Loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = build_config()
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
