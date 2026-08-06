import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from datetime import datetime
from fpdf import FPDF
import tempfile
import os

st.set_page_config(
    page_title="Kidney Disease Classifier",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #0B3D5C;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 16px;
        color: #5A6B7B;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 22px;
        border-radius: 12px;
        background: linear-gradient(135deg, #E8F5F0 0%, #DCEEF5 100%);
        border: 1px solid #B8DCE8;
        text-align: center;
    }
    .explanation-box {
        padding: 18px;
        border-radius: 10px;
        background-color: #F7F9FA;
        border-left: 4px solid #0B7285;
        margin-top: 15px;
    }
    .history-card {
        padding: 10px 14px;
        border-radius: 8px;
        background-color: #F7F9FA;
        border: 1px solid #E0E4E8;
        margin-bottom: 8px;
    }
    section[data-testid="stSidebar"] {
        background-color: #0B3D5C;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🩺 Kidney AI")
    st.markdown("---")

    st.markdown("### 🔬 Scan Type")
    scan_type = st.radio(
        "Select the type of scan you are uploading:",
        ["CT Scan", "MRI Scan"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🧾 Patient Information (optional)")
    st.caption("Filled fields will appear on the downloadable PDF report.")
    patient_name = st.text_input("Patient Name")
    patient_age = st.text_input("Age")
    patient_gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])

    st.markdown("---")
    st.markdown("### About")
    st.write(
        "This tool uses a deep learning model to classify CT/MRI kidney scan "
        "images into four categories: Cyst, Normal, Stone, and Tumor."
    )
    st.markdown("### Model Info")
    st.write("**Type:** CNN (Deep Learning)")
    st.write(f"**Input:** {scan_type} (grayscale)")
    st.write("**Classes:** Cyst, Normal, Stone, Tumor")
    st.markdown("---")
    st.markdown("### ⚠️ Important")
    st.write(
        "This is a prototype AI tool for screening assistance only. "
        "It has not undergone formal clinical validation. All results "
        "must be reviewed by a qualified radiologist or physician."
    )
    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">🩺 Kidney Disease Classifier</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">AI-assisted classification for {scan_type.lower()} kidney images</p>', unsafe_allow_html=True)

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

def is_likely_scan_image(image, color_threshold=15):
    img_array = np.array(image.convert("RGB"))
    r = img_array[:, :, 0].astype(int)
    g = img_array[:, :, 1].astype(int)
    b = img_array[:, :, 2].astype(int)
    color_diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean()
    return color_diff < color_threshold


def generate_pdf_report(scan_type, patient_name, patient_age, patient_gender,
                         image, predicted_class, confidence, class_names,
                         prediction, info):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(11, 61, 92)
    pdf.cell(0, 12, f"{scan_type} Kidney Disease Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 107, 123)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(4)
    pdf.set_draw_color(184, 220, 232)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(11, 61, 92)
    pdf.cell(0, 8, "Patient Information", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, f"Name: {patient_name or 'N/A'}", ln=True)
    pdf.cell(0, 7, f"Age: {patient_age or 'N/A'}", ln=True)
    pdf.cell(0, 7, f"Gender: {patient_gender or 'N/A'}", ln=True)
    pdf.cell(0, 7, f"Scan Type: {scan_type}", ln=True)
    pdf.ln(4)

    img_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            image.save(tmp_img.name)
            img_path = tmp_img.name
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(11, 61, 92)
        pdf.cell(0, 8, "Uploaded Scan", ln=True)
        pdf.image(img_path, w=80)
        pdf.ln(4)
    finally:
        if img_path and os.path.exists(img_path):
            os.unlink(img_path)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(11, 61, 92)
    pdf.cell(0, 8, "AI Analysis Result", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, f"Prediction: {predicted_class}", ln=True)
    pdf.cell(0, 7, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Class Probabilities:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for i, cname in enumerate(class_names):
        pdf.cell(0, 6, f"    {cname}: {prediction[0][i] * 100:.1f}%", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Finding Summary:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, info["summary"])
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Details:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, info["detail"])
    pdf.ln(6)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(
        0, 5,
        "Disclaimer: This tool is intended for educational and demonstration "
        "purposes only and does not constitute medical advice or diagnosis. "
        "All findings must be verified by a qualified healthcare professional."
    )

    return bytes(pdf.output())

@st.cache_resource
def load_my_model():
    model_path = hf_hub_download(repo_id="M10001/my-model", filename="kidney_disease_model.h5")
    return load_model(model_path)

with st.spinner("Loading model..."):
    model = load_my_model()

class_names = ["Cyst", "Normal", "Stone", "Tumor"]

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- UPLOAD ----------------
st.markdown(f"### 📤 Upload {scan_type}")
uploaded_file = st.file_uploader("Supported formats: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption=f"Uploaded {scan_type}", use_container_width=True)

    if not is_likely_scan_image(image):
        with col2:
            st.markdown("### 🔍 Result")
            st.error(f"🚫 **This does not appear to be a valid {scan_type.lower()}.**\n\nThe uploaded image does not match the visual characteristics of a {scan_type.lower()} (these scans are typically grayscale). Please upload a valid {scan_type.lower()} image.")
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
                st.error(f"**Low Confidence:** {confidence:.2f}%\n\nThis image may not be a valid {scan_type.lower()}, or the scan quality is unclear.")
            else:
                st.markdown(f"""
                    <div class="result-box">
                        <h3 style="color:#0B3D5C; margin-bottom:5px;">{predicted_class}</h3>
                        <p style="color:#5A6B7B;">Confidence: <b>{confidence:.2f}%</b></p>
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

            st.session_state.history.insert(0, {
                "filename": uploaded_file.name,
                "prediction": predicted_class,
                "confidence": f"{confidence:.2f}%",
                "time": datetime.now().strftime("%H:%M:%S")
            })

            # ---------------- PDF REPORT ----------------
            pdf_bytes = generate_pdf_report(
                scan_type=scan_type,
                patient_name=patient_name,
                patient_age=patient_age,
                patient_gender=patient_gender,
                image=image,
                predicted_class=predicted_class,
                confidence=confidence,
                class_names=class_names,
                prediction=prediction,
                info=info
            )

            report_filename = f"{scan_type.replace(' ', '_')}_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=report_filename,
                mime="application/pdf"
            )

# ---------------- HISTORY ----------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕒 Scan History (this session)")
    for item in st.session_state.history:
        st.markdown(f"""
            <div class="history-card">
                <b>{item['filename']}</b> — {item['prediction']} ({item['confidence']}) 
                <span style="color:#999; float:right;">{item['time']}</span>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("⚕️ Disclaimer: This tool is intended for educational and demonstration purposes only and does not constitute medical advice or diagnosis. All findings must be verified by a qualified healthcare professional.")
