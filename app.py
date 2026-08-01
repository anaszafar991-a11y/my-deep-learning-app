import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="CT Kidney Disease Classifier",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 16px;
        color: #555555;
        margin-bottom: 25px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f7f4;
        border: 1px solid #c8e6d8;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🩺 CT Kidney Disease Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered classification for CT kidney scan images</p>', unsafe_allow_html=True)

with st.expander("ℹ️ About this tool", expanded=False):
    st.write("""
    This tool uses a deep learning model trained to classify CT kidney scan images 
    into four categories: **Cyst, Normal, Stone, and Tumor**.
    
    **Important:** This model is trained exclusively on CT Kidney scan images. 
    Uploading any other type of image will produce unreliable results.
    """)

@st.cache_resource
def load_my_model():
    model_path = hf_hub_download(repo_id="M10001/my-model", filename="kidney_disease_model.h5")
    return load_model(model_path)

with st.spinner("Loading model..."):
    model = load_my_model()

class_names = ["Cyst", "Normal", "Stone", "Tumor"]

st.markdown("### 📤 Upload CT Scan")
uploaded_file = st.file_uploader("Supported formats: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded CT Scan", use_container_width=True)

    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array)

    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    predicted_class = class_names[predicted_class_idx]

    with col2:
        st.markdown("### 🔍 Result")
        if confidence < 60:
            st.error(f"**Low Confidence:** {confidence:.2f}%\n\nThis image may not be a valid CT kidney scan, or the scan quality is unclear.")
        else:
            st.markdown(f"""
                <div class="result-box">
                    <h3 style="color:#1E3A5F; margin-bottom:5px;">{predicted_class}</h3>
                    <p style="color:#555;">Confidence: <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("**Class Probabilities:**")
        for i, class_name in enumerate(class_names):
            st.progress(float(prediction[0][i]), text=f"{class_name}: {prediction[0][i]*100:.1f}%")

st.markdown("---")
st.caption("⚕️ Disclaimer: This tool is intended for educational and demonstration purposes only and does not constitute medical advice or diagnosis. Please consult a qualified healthcare professional for any medical concerns.")
