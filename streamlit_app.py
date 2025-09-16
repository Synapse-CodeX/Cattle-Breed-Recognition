
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import tempfile
import json
from io import BytesIO
from pathlib import Path

MODEL_DIR = Path("saved_model/breed_classifier")
MODEL_URL = st.secrets.get("MODEL_URL", "")  # set this in Streamlit Cloud secrets if using remote model
CLASSES_FILE = MODEL_DIR / "classes.json"
IMG_SIZE = (224, 224)

@st.cache_resource(ttl=60*60*24)
def download_and_load_model(model_url: str, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(model_url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download model: {r.status_code}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with open(tmp.name, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    import zipfile
    with zipfile.ZipFile(tmp.name, "r") as z:
        z.extractall(local_dir)
    return tf.keras.models.load_model(str(local_dir))

@st.cache_resource(ttl=60*60*24)
def load_model_local_or_remote():
    if MODEL_DIR.exists() and (MODEL_DIR / "saved_model.pb").exists():
        model = tf.keras.models.load_model(str(MODEL_DIR))
    else:
        if not MODEL_URL:
            raise RuntimeError("No local model found and MODEL_URL not set. Upload model to repo or set MODEL_URL in secrets.")
        model = download_and_load_model(MODEL_URL, MODEL_DIR)
    return model

@st.cache_data(ttl=60*60*24)
def load_class_names():
    if CLASSES_FILE.exists():
        with open(CLASSES_FILE, "r") as f:
            return json.load(f)
    else:
        return None

def preprocess_image_pil(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    return arr

st.set_page_config(page_title="Cattle Breed Classifier", layout="centered")
st.title("Cattle Breed Recognition")
st.write("Upload an image of a cow/bull and the model will predict the breed.")

uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
if uploaded is not None:
    try:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_column_width=True)
        with st.spinner("Loading model..."):
            model = load_model_local_or_remote()
            class_names = load_class_names()
        input_arr = preprocess_image_pil(img)
        preds = model.predict(input_arr)[0]
        top_idxs = preds.argsort()[-3:][::-1]
        st.subheader("Top predictions")
        for idx in top_idxs:
            label = class_names[idx] if class_names else str(idx)
            st.write(f"- **{label}** — {preds[idx]*100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Upload an image to get prediction.")
