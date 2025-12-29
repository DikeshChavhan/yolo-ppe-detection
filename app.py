from ultralytics import YOLO
import streamlit as st
from PIL import Image
from utils import draw_results

st.set_page_config(page_title="PPE Detection App", layout="centered")
st.title("🦺 PPE Detection using YOLO11")

st.write("Upload an image to detect Helmet / Vest")

# Load model once
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Running detection…")
    results = model(image)[0]

    output = draw_results(results)
    st.image(output, caption="Prediction", use_column_width=True)

    st.success("✅ Detection complete")
else:
    st.info("Upload an image to begin.")
