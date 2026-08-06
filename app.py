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
    st.markdown("### 🧾 Patient Information")
    st.caption("Used to generate the clinical report below.")
    patient_name = st.text_input("Patient Name")
    col_a, col_b = st.columns(2)
    with col_a:
        patient_age = st.text_input("Age")
    with col_b:
        patient_gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
    patient_id = st.text_input("Patient ID / MRN")
    referring_physician = st.text_input("Referring Physician")
    clinical_history = st.text_area("Clinical Indication / History", height=70,
                                     placeholder="e.g. Flank pain, routine screening, follow-up...")

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

def build_report_text(scan_type, patient_name, patient_age, patient_gender, patient_id,
                       referring_physician, clinical_history, predicted_class, confidence,
                       class_names, prediction, info):
    now = datetime.now()
    prob_lines = "\n".join(
        f"    {cname:<10} : {prediction[0][i] * 100:5.1f}%"
        for i, cname in enumerate(class_names)
    )
    lines = [
        "==================================================",
        "            AI KIDNEY IMAGING REPORT",
        "==================================================",
        f"Report Date   : {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Exam Type     : {scan_type}",
        "",
        "PATIENT INFORMATION",
        "--------------------------------------------------",
        f"Name                 : {patient_name or 'N/A'}",
        f"Age / Gender         : {patient_age or 'N/A'} / {patient_gender or 'N/A'}",
        f"Patient ID / MRN     : {patient_id or 'N/A'}",
        f"Referring Physician  : {referring_physician or 'N/A'}",
        "",
        "CLINICAL INDICATION",
        "--------------------------------------------------",
        clinical_history or "Not provided.",
        "",
        "TECHNIQUE",
        "--------------------------------------------------",
        f"AI-assisted classification of {scan_type.lower()} kidney images using a "
        "convolutional neural network (CNN) trained to identify Cyst, Normal, "
        "Stone, and Tumor patterns.",
        "",
        "FINDINGS",
        "--------------------------------------------------",
        info["summary"],
        info["detail"],
        "",
        "Class Probabilities:",
        prob_lines,
        "",
        "IMPRESSION",
        "--------------------------------------------------",
        f"{predicted_class} (AI confidence: {confidence:.2f}%)",
        "",
        "RECOMMENDATION",
        "--------------------------------------------------",
        "This is an AI-assisted screening result, not a confirmed diagnosis. "
        "Correlate clinically and confirm with a qualified radiologist or "
        "physician before making any treatment decision.",
        "",
        "==================================================",
        "This report was generated by an AI screening tool for educational/",
        "demonstration purposes and does not constitute medical advice.",
        "==================================================",
    ]
    return "\n".join(lines)


def is_likely_scan_image(image, color_threshold=15):
    img_array = np.array(image.convert("RGB"))
    r = img_array[:, :, 0].astype(int)
    g = img_array[:, :, 1].astype(int)
    b = img_array[:, :, 2].astype(int)
    color_diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean()
    return color_diff < color_threshold


def generate_pdf_report(scan_type, patient_name, patient_age, patient_gender, patient_id,
                         referring_physician, clinical_history, image, predicted_class,
                         confidence, class_names, prediction, info):
    now = datetime.now()
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(11, 61, 92)
    pdf.cell(0, 10, "AI Kidney Imaging Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 107, 123)
    pdf.cell(0, 6, f"{scan_type}  |  Generated {now.strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(3)
    pdf.set_draw_color(184, 220, 232)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    def section_title(text):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(11, 61, 92)
        pdf.cell(0, 8, text, ln=True)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(0, 0, 0)

    def wrapped(text):
        pdf.multi_cell(0, 6, text)
        pdf.set_x(pdf.l_margin)

    def field_row(label, value):
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.cell(45, 6.5, label, ln=False)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.cell(0, 6.5, str(value) if value else "N/A", ln=True)
        pdf.set_x(pdf.l_margin)

    section_title("Patient Information")
    field_row("Name:", patient_name)
    field_row("Age / Gender:", f"{patient_age or 'N/A'} / {patient_gender or 'N/A'}")
    field_row("Patient ID / MRN:", patient_id)
    field_row("Referring Physician:", referring_physician)
    pdf.ln(3)

    section_title("Clinical Indication")
    wrapped(clinical_history or "Not provided.")
    pdf.ln(3)

    section_title("Technique")
    wrapped(
        f"AI-assisted classification of {scan_type.lower()} kidney images using a "
        "convolutional neural network (CNN) trained to identify Cyst, Normal, "
        "Stone, and Tumor patterns."
    )
    pdf.ln(3)

    img_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            image.save(tmp_img.name)
            img_path = tmp_img.name
        section_title("Scan Image")
        pdf.image(img_path, w=70)
        pdf.set_x(pdf.l_margin)
        pdf.ln(3)
    finally:
        if img_path and os.path.exists(img_path):
            os.unlink(img_path)

    section_title("Findings")
    wrapped(info["summary"])
    wrapped(info["detail"])
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(0, 6.5, "Class Probabilities:", ln=True)
    pdf.set_font("Helvetica", "", 10.5)
    for i, cname in enumerate(class_names):
        pdf.cell(0, 6, f"    {cname}: {prediction[0][i] * 100:.1f}%", ln=True)
    pdf.ln(3)

    section_title("Impression")
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(11, 61, 92)
    pdf.cell(0, 7, f"{predicted_class}  (AI confidence: {confidence:.2f}%)", ln=True)
    pdf.set_x(pdf.l_margin)
    pdf.ln(2)

    section_title("Recommendation")
    wrapped(
        "This is an AI-assisted screening result, not a confirmed diagnosis. "
        "Correlate clinically and confirm with a qualified radiologist or "
        "physician before making any treatment decision."
    )
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    wrapped(
        "This report was generated by an AI screening tool for educational/demonstration "
        "purposes and does not constitute medical advice or diagnosis."
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

            # ---------------- FULL REPORT (TEXT) ----------------
            report_text = build_report_text(
                scan_type=scan_type,
                patient_name=patient_name,
                patient_age=patient_age,
                patient_gender=patient_gender,
                patient_id=patient_id,
                referring_physician=referring_physician,
                clinical_history=clinical_history,
                predicted_class=predicted_class,
                confidence=confidence,
                class_names=class_names,
                prediction=prediction,
                info=info
            )

            st.markdown("### 📝 Full Clinical Report")
            st.text_area("Report (copyable)", report_text, height=420, label_visibility="collapsed")

            # ---------------- PDF REPORT ----------------
            pdf_bytes = generate_pdf_report(
                scan_type=scan_type,
                patient_name=patient_name,
                patient_age=patient_age,
                patient_gender=patient_gender,
                patient_id=patient_id,
                referring_physician=referring_physician,
                clinical_history=clinical_history,
                image=image,
                predicted_class=predicted_class,
                confidence=confidence,
                class_names=class_names,
                prediction=prediction,
                info=info
            )

            base_filename = f"{scan_type.replace(' ', '_')}_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{base_filename}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            with dl_col2:
                st.download_button(
                    label="📃 Download Text Report",
                    data=report_text,
                    file_name=f"{base_filename}.txt",
                    mime="text/plain",
                    use_container_width=True
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
