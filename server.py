from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

MODEL_DIR = "saved_model/breed_classifier"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_DIR)
class_names = None
try:
    import json, os
    classes_file = os.path.join(MODEL_DIR, "classes.json")
    if os.path.exists(classes_file):
        with open(classes_file, "r") as f:
            class_names = json.load(f)
except Exception:
    class_names = None

app = Flask(__name__)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error":"no file part"}), 400
    file = request.files["file"]
    img_bytes = file.read()
    x = preprocess_image(img_bytes)
    preds = model.predict(x)[0]
    idx = int(preds.argmax())
    conf = float(preds[idx])
    label = class_names[idx] if class_names else str(idx)
    return jsonify({"pred_idx": idx, "pred_label": label, "confidence": conf})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

