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
        predictions = []
        for task in tasks:
            image_url = task["data"]["image"]
            image_path = image_url.replace("/data/upload/", "/Users/rileymcnamara/Library/Application Support/label-studio/media/upload/")
            
            print(f"Original image URL: {image_url}")
            print(f"Converted image path: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                print(f"🚨 OpenCV failed to read image: {image_path}")
                continue  # Skip this image

            # Run the model
            outputs = self.model(image)
            instances = outputs["instances"].to("cpu")

            print(f"🔍 Model Outputs: {instances}")  # Debugging

            results = []
            for i in range(len(instances.pred_boxes)):
                bbox = instances.pred_boxes[i].tensor.numpy()[0]
                category_id = int(instances.pred_classes[i])
                score = float(instances.scores[i])

                print(f"🟢 Prediction {i}: BBox={bbox}, Category={category_id}, Score={score}")

                # Convert bbox to Label Studio's format
                result = {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": (bbox[0] / image.shape[1]) * 100,  # Convert to percentage
                        "y": (bbox[1] / image.shape[0]) * 100,
                        "width": ((bbox[2] - bbox[0]) / image.shape[1]) * 100,
                        "height": ((bbox[3] - bbox[1]) / image.shape[0]) * 100,
                        "rectanglelabels": [f"Class {category_id}"]  # Modify this if you have class names
                    },
                    "score": score  # Optional: Confidence score
                }
                results.append(result)

            predictions.append({
                "result": results
            })

        print(f"🔍 Final Predictions Sent to Label Studio: {json.dumps(predictions, indent=2)}")  # Debugging
        return predictions


model = MyModel(
    "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/trained/model_final.pth",
    "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/scripts/config_home.yaml"
)

@app.route('/setup', methods=['POST'])
def setup():
    return jsonify({"status": "ok"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    tasks = request.json.get('tasks', [])
    predictions = model.predict(tasks)
    return jsonify(predictions)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
