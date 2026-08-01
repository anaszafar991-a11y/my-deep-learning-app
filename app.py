import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
import numpy as np

st.title("Mera Deep Learning Model")

@st.cache_resource
def load_my_model():
    model_path = hf_hub_download(repo_id="TUMHARA-USERNAME/my-model", filename="my_model.h5")
    return load_model(model_path)

model = load_my_model()

uploaded_file = st.file_uploader("Image upload karo", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file)
    st.write("Prediction: ...")