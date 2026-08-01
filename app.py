import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.title("CT Kidney Disease Classifier")
st.write("CT scan image upload karo, model bata dega disease ka type")

@st.cache_resource
def load_my_model():
    model_path = hf_hub_download(repo_id="M10001/my-model", filename="kidney_disease_model.h5")
    return load_model(model_path)

model = load_my_model()

class_names = ["Cyst", "Normal", "Stone", "Tumor"]

uploaded_file = st.file_uploader("CT scan image upload karo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded CT Scan", width=300)

    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    predicted_class = class_names[predicted_class_idx]

    st.success(f"Prediction: **{predicted_class}** (Confidence: {confidence:.2f}%)")
