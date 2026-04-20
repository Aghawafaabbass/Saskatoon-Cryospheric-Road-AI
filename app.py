import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import datetime

# --- Page Config ---
st.set_page_config(page_title="Saskatoon Winter AI", layout="wide", page_icon="❄️")

# --- UI Styling (Visibility Fix) ---
st.markdown("""
    <style>
    /* Metric boxes fix for Dark Mode */
    [data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #464b5c;
        padding: 15px;
        border-radius: 15px;
    }
    [data-testid="stMetricLabel"] {
        color: #00d4ff !important;
        font-weight: bold;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    /* Main title styling */
    .main-title {
        color: #00d4ff;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
    }
    footer {visibility: hidden;}
    .footer-text {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: white;
        text-align: center;
        padding: 8px;
        font-size: 14px;
        z-index: 100;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar: Features ---
with st.sidebar:
    st.title("📍 Saskatoon Hub")
    st.metric("City", "Saskatoon, SK")
    st.metric("Current Temp", "-14°C", "❄️ Snowy")
    
    st.write("---")
    st.subheader("👨‍💻 Developer")
    st.success("Agha Wafa Abbas")
    
    st.write("---")
    st.info("System: YOLOv8 Engine\nOS: Debian Trixie")

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error("Model 'best.pt' missing from repository!")
        return None

model = load_model()

# --- Main Interface ---
st.markdown('<p class="main-title">❄️ Saskatoon Cryospheric Road Safety System</p>', unsafe_allow_html=True)
st.write("---")

# Layout for Uploader and Stats
col_up, col_stat = st.columns([2, 1])

with col_up:
    uploaded_file = st.file_uploader("📷 Upload Road Image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None and model is not None:
    # 1. Processing
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    with st.spinner('AI analyzing road conditions...'):
        results = model.predict(source=img_array, conf=0.25)
        res_plotted = results[0].plot()
        
    # 2. Results Layout
    res_col1, res_col2 = st.columns([1.5, 1])

    with res_col1:
        st.subheader("🔍 AI Vision")
        st.image(res_plotted, caption="Detection Result", use_container_width=True)

    with res_col2:
        st.subheader("🛡️ Safety Dashboard")
        names = model.names
        detected = [names[int(box.cls)] for box in results[0].boxes]
        
        if detected:
            condition = ", ".join(set(detected))
            st.write(f"**Detected Conditions:** {condition}")
            
            hazard_keywords = ['snow', 'ice', 'slush', 'black ice']
            is_hazardous = any(x.lower() in hazard_keywords for x in detected)
            
            if is_hazardous:
                st.error("🚨 **DANGER: Hazardous Road!**")
                st.warning("Advice: Speed limit 30-40 km/h. High risk of skidding.")
            else:
                st.success("✅ **SAFE: Road Clear**")
                st.write("Normal driving conditions.")
            
            # --- New Feature: Download Report ---
            report_data = f"Saskatoon Road Safety Report\nDate: {datetime.datetime.now()}\nCondition: {condition}\nHazard: {is_hazardous}"
            st.download_button("📥 Download Analysis Report", report_data, file_name="Road_Report.txt")
            
        else:
            st.info("No hazards detected.")
            is_hazardous = False

    # --- 3. Interactive Map ---
    st.write("---")
    st.subheader("📍 Incident Map (Saskatoon Area)")
    m = folium.Map(location=[52.1332, -106.6700], zoom_start=12)
    folium.Marker(
        [52.1332, -106.6700], 
        popup="Analysis Location", 
        icon=folium.Icon(color='red' if is_hazardous else 'green', icon='info-sign')
    ).add_to(m)
    st_folium(m, width="100%", height=350)

# --- Professional Footer ---
st.markdown(f"""
    <div class="footer-text">
        System Active | Developed by <b>Agha Wafa Abbas</b> | {datetime.datetime.now().year}
    </div>
    """, unsafe_allow_html=True)
