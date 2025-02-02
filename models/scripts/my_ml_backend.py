import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import json

app = Flask(__name__)

LABEL_CONFIG = {
    0: {"value": "Class 0", "background": "#0000FF"},  # Address
    1: {"value": "Class 1", "background": "#FFFF00"},  # Date of Birth
    2: {"value": "Class 2", "background": "#00FFFF"},  # Dr Info
    3: {"value": "Class 3", "background": "#00FF00"},  # Given Name
    4: {"value": "Class 4", "background": "#FF0000"},  # Medicare Number
    5: {"value": "Class 5", "background": "#800080"},  # Phone Number
    6: {"value": "Class 6", "background": "#FFA500"},  # Request Date
    7: {"value": "Class 7", "background": "#008080"},  # Sex
    8: {"value": "Class 8", "background": "#FF00FF"},  # Surname
}

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
    
    def predict(self, tasks, **kwargs):
        # If only one task is provided, aggregate all annotations in a single dict.
        if len(tasks) == 1:
            task = tasks[0]
            aggregated_results = []  # this will hold all rectangle annotations for the task

            # Process the image for the task
            image_url = task["data"].get("image")
            image_path = image_url.replace("/data/upload/", "/Users/rileymcnamara/Library/Application Support/label-studio/media/upload/")
            image = cv2.imread(image_path)
            if image is None:
                # Optionally log an error and continue
                return {"model_version": "v1.0", "results": []}

            outputs = self.model(image)
            instances = outputs["instances"].to("cpu")

            # Loop over detections and accumulate results
            for i in range(len(instances.pred_boxes)):
                bbox_arr = instances.pred_boxes[i].tensor.numpy()[0]
                x1, y1, x2, y2 = float(bbox_arr[0]), float(bbox_arr[1]), float(bbox_arr[2]), float(bbox_arr[3])
                category_id = int(instances.pred_classes[i])
                score = float(instances.scores[i])
                annotation = {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": float((x1 / image.shape[1]) * 100),
                        "y": float((y1 / image.shape[0]) * 100),
                        "width": float(((x2 - x1) / image.shape[1]) * 100),
                        "height": float(((y2 - y1) / image.shape[0]) * 100),
                        "rectanglelabels": [f"Class {category_id}"]
                    },
                    "score": score
                }
                aggregated_results.append(annotation)

            # Return a single prediction dict for this task
            return {"model_version": "v1.0", "result": aggregated_results}

        else:
            # For multiple tasks, build a list with one aggregated dict per task.
            responses = []
            for task in tasks:
                aggregated_results = []
                image_url = task["data"].get("image")
                image_path = image_url.replace("/data/upload/", "/your/local/path/")
                image = cv2.imread(image_path)
                if image is None:
                    responses.append({"model_version": "v1.0", "results": []})
                    continue

                outputs = self.model(image)
                instances = outputs["instances"].to("cpu")
                for i in range(len(instances.pred_boxes)):
                    bbox_arr = instances.pred_boxes[i].tensor.numpy()[0]
                    x1, y1, x2, y2 = float(bbox_arr[0]), float(bbox_arr[1]), float(bbox_arr[2]), float(bbox_arr[3])
                    category_id = int(instances.pred_classes[i])
                    score = float(instances.scores[i])
                    annotation = {
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "x": float((x1 / image.shape[1]) * 100),
                            "y": float((y1 / image.shape[0]) * 100),
                            "width": float(((x2 - x1) / image.shape[1]) * 100),
                            "height": float(((y2 - y1) / image.shape[0]) * 100),
                            "rectanglelabels": [f"Class {category_id}"]
                        },
                        "score": score
                    }
                    aggregated_results.append(annotation)
                responses.append({"result": aggregated_results})
            return responses



model = MyModel(
    "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/trained/model_final.pth",
    "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/scripts/config_home.yaml"
)

@app.route('/setup', methods=['POST'])
def setup():
    return jsonify({
        "model_version": "detectron2 v1",
        "labels": list(LABEL_CONFIG.values()),
        "status": "ok"
    })

@app.route('/predict', methods=['POST'])
def predict():
    tasks = request.json.get('tasks', [])
    if not tasks:
        # Return an empty dict or dict with empty results if no tasks provided.
        return {}

    # If there's only one task, return the prediction dict directly.
    if len(tasks) == 1:
        print("correct")
        prediction = model.predict(tasks)
        # prediction should already be a dict like:
        # { "model_version": "v1.0", "results": [ ... annotation objects ... ] }
        return prediction
    else:
        print("incorrect")
        # For multiple tasks, your model.predict() should return a list of dicts.
        predictions = model.predict(tasks)
        return predictions



@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)