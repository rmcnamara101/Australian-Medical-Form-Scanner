from flask import Flask, request, Response
import cv2
import json
import uuid
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

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
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(cfg)

    def predict(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        
        result = []
        for i in range(len(instances.pred_boxes)):
            bbox = instances.pred_boxes[i].tensor.numpy()[0].tolist()
            score = float(instances.scores[i])
            category_id = int(instances.pred_classes[i])
            
            # Convert bbox to relative coordinates
            x1, y1, x2, y2 = bbox
            annotation = {
                "id": str(uuid.uuid4()),
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": 100 * x1 / image.shape[1],
                    "y": 100 * y1 / image.shape[0],
                    "width": 100 * (x2 - x1) / image.shape[1],
                    "height": 100 * (y2 - y1) / image.shape[0],
                    "rectanglelabels": [f"Class {category_id}"]
                },
                "score": score
            }
            result.append(annotation)
        
        return result

model = MyModel(
    "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/trained/model_final.pth",
    "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/models/scripts/config_home.yaml"
)

@app.route('/predict', methods=['POST'])
def predict():
    tasks = request.json.get('tasks', [])
    predictions = []
    
    for task in tasks:
        image_url = task["data"]["image"]
        image_path = image_url.replace(
            "/data/upload/", 
            "/Users/rileymcnamara/Library/Application Support/label-studio/media/upload/"
        )
        
        result = model.predict(image_path)
        predictions.append({
            "result": result,
            "score": max([r["score"] for r in result]) if result else 0,
            "model_version": "detectron2 v1"
        })

    return {"predictions": predictions}

@app.route('/setup', methods=['POST'])
def setup():
    return {
        "model_version": "detectron2 v1",
        "labels": list(LABEL_CONFIG.values()),
        "status": "ok"
    }

@app.route('/health', methods=['GET'])
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090)