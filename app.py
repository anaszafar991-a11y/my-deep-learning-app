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
    .explanation-box {
        padding: 18px;
        border-radius: 10px;
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🩺 CT Kidney Disease Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-assisted classification for CT kidney scan images</p>', unsafe_allow_html=True)

with st.expander("ℹ️ About this tool", expanded=False):
    st.write("""
    This tool uses a deep learning model trained to classify CT kidney scan images 
    into four categories: **Cyst, Normal, Stone, and Tumor**.

    **Important:** This is a prototype AI tool intended to assist screening, not to 
    replace professional medical diagnosis. This model has not undergone formal 
    clinical validation or regulatory approval. All results must be reviewed and 
    confirmed by a qualified radiologist or physician.
    """)

class_info = {
    "Normal": {
        "summary": "No visible abnormalities detected.",
        "detail": "The uploaded CT scan shows kidney structure consistent with a normal, healthy kidney. No signs of cysts, stones, or tumors were detected by the model."
    },
    "Cyst": {
        "summary": "Findings consistent with a renal cyst.",
        "detail": "The model has identified a fluid-filled sac-like structure in the kidney, a pattern typically associated with a renal cyst. Cysts are often benign but should be evaluated by a physician, especially if large or symptomatic."
    },
    "Stone": {
        "summary": "Findings consistent with a kidney stone.",
        "detail": "The model has detected a dense, mineral-like formation in the kidney, consistent with the presence of a kidney stone (nephrolithiasis). Size, location, and symptoms should be clinically assessed."
    },
    "Tumor": {
        "summary": "Findings consistent with a renal mass/tumor.",
        "detail": "The model has identified an irregular mass-like structure in the kidney, which may indicate a tumor. This finding requires urgent clinical evaluation, including further imaging and possibly biopsy, to determine whether the mass is benign or malignant."
    }
}

def is_likely_ct_scan(image, color_threshold=15):
    """CT scans are grayscale. Reject images with significant color content."""
    img_array = np.array(image.convert("RGB"))
    r = img_array[:, :, 0].astype(int)
    g = img_array[:, :, 1].astype(int)
    b = img_array[:, :, 2].astype(int)
    color_diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean()
    return color_diff < color_threshold

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

    if not is_likely_ct_scan(image):
        with col2:
            st.markdown("### 🔍 Result")
            st.error("🚫 **This does not appear to be a CT kidney scan.**\n\nThe uploaded image does not match the visual characteristics of a CT scan (CT scans are grayscale). Please upload a valid CT kidney scan image.")
    else:
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

        if confidence >= 60:
            info = class_info[predicted_class]
            st.markdown(f"""
                <div class="explanation-box">
                    <b>📋 Finding Summary:</b> {info['summary']}<br><br>
                    <b>Details:</b> {info['detail']}
                </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("⚕️ Disclaimer: This tool is intended for educational and demonstration purposes only and does not constitute medical advice or diagnosis. All findings must be verified by a qualified healthcare professional.")
