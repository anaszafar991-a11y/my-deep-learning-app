import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.title("CT Kidney Disease Classifier")
st.write("Upload a CT scan image and the model will classify the disease type.")

st.warning("⚠️ **Note:** This model is trained exclusively on CT Kidney scan images (categories: Cyst, Normal, Stone, Tumor). Uploading any other type of image (e.g. a regular photo, X-ray, or scan of a different organ) will produce unreliable or incorrect results.")

@st.cache_resource
def load_my_model():
    model_path = hf_hub_download(repo_id="M10001/my-model", filename="kidney_disease_model.h5")
    return load_model(model_path)

model = load_my_model()

class_names = ["Cyst", "Normal", "Stone", "Tumor"]

uploaded_file = st.file_uploader("Upload a CT scan image", type=["jpg", "jpeg", "png"])

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

    if confidence < 60:
        st.error(f"❓ Low confidence result ({confidence:.2f}%). This image may not be a valid CT kidney scan, or the scan quality is unclear. Please try uploading a clearer CT scan image.")
    else:
        st.success(f"Prediction: **{predicted_class}** (Confidence: {confidence:.2f}%)")

    st.caption("Disclaimer: This tool is intended for educational and demonstration purposes only and does not constitute medical advice or diagnosis. Please consult a qualified healthcare professional for any medical concerns.")
