import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://localhost:8000/predict"  # test on local

st.set_page_config(page_title="Cattle Breed Classifier", layout="centered")

st.title("Cattle Breed Classifier")
st.write("Upload an image of cattle, and the model will predict the breed!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    #uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify"):
        with st.spinner("Classifying..."):
            try:
                # request tobackend
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    predictions = response.json()["predictions"]

                    st.subheader("Top Predictions:")
                    for pred in predictions:
                        st.write(f"{pred['label']} = {pred['confidence']*100:.2f}%")

                        # Confidence bar
                        st.progress(pred["confidence"])

                else:
                    st.error(f"Error: {response.json()['detail']}")

            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
