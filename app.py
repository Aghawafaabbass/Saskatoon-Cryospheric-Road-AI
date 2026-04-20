
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Saskatoon Winter AI", layout="wide")
st.title("❄️ Saskatoon Cryospheric Road Safety System")

@st.cache_resource
def load_model():
    return YOLO("best.pt") # Local path for GitHub/Streamlit Cloud

model = load_model()

uploaded_file = st.file_uploader("Upload a road image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    results = model.predict(source=img_array, conf=0.25)
    res_plotted = results.plot()
    st.image(res_plotted, use_column_width=True)
    st.success("Analysis Complete!")
