from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path
from io import BytesIO

# Constants
MODEL_DIR = Path("saved_model/breed_classifier")
CLASSES_FILE = MODEL_DIR / "classes.json"
IMG_SIZE = (224, 224)

# Load model at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    try:
        model = tf.keras.models.load_model(str(MODEL_DIR))
        if CLASSES_FILE.exists():
            with open(CLASSES_FILE, "r") as f:
                class_names = json.load(f)
        else:
            class_names = None
        app.state.model = model
        app.state.class_names = class_names
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    yield
    
app = FastAPI(lifespan=lifespan)

def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.post("/predict")
async def predict(file: UploadFile = File(...,description="Upload an image file (jpg, jpeg, png)",)):
    return {
        "predictions": [
            {"label": "Sudeep", "confidence": 0.9342},
            {"label": "Alien", "confidence": 0.0521},
            {"label": "LOL", "confidence": 0.0137}
                    ]
                }

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        input_array = preprocess_image(image)

        predictions = app.state.model.predict(input_array)[0]
        top_idxs = predictions.argsort()[-3:][::-1]

        results = []
        for idx in top_idxs:
            label = app.state.class_names[str(idx)] if app.state.class_names else str(idx)
            confidence = float(predictions[idx])
            results.append({"label": label, "confidence": round(confidence, 2)})

        return JSONResponse(content={"predictions": results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
