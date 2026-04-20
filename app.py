import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium

# Page Setup
st.set_page_config(page_title="Saskatoon Winter AI", layout="wide", page_icon="❄️")

# --- UI Styling ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    footer {visibility: hidden;}
    .footer-text {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: white;
        text-align: center;
        padding: 10px;
        font-family: sans-serif;
        z-index: 100;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar: Weather & Info ---
st.sidebar.title("📍 Location Info")
st.sidebar.metric("City", "Saskatoon, SK")
st.sidebar.metric("Current Temp", "-14°C", "Snowy")
st.sidebar.info("This system uses AI to detect hazardous road conditions in real-time.")
st.sidebar.write("---")
st.sidebar.markdown("**Developed by:**")
st.sidebar.subheader("👨‍💻 Agha Wafa Abbas")

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error("Model 'best.pt' nahi mili. Please check your GitHub files.")
        return None

model = load_model()

# --- Main App ---
st.title("❄️ Saskatoon Cryospheric Road Safety System")
st.write("---")

uploaded_file = st.file_uploader("📷 Upload Road Image for Analysis...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    with st.spinner('AI is analyzing the road conditions...'):
        results = model.predict(source=img_array, conf=0.25)[0] # Get first result
        res_plotted = results.plot()
        
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔍 AI Detection Result")
        st.image(res_plotted, caption="Processed Image", use_container_width=True)

    with col2:
        st.subheader("🛡️ Safety Dashboard")
        names = model.names
        detected = [names[int(box.cls)] for box in results.boxes]
        
        if detected:
            st.write(f"**Conditions Found:** {', '.join(set(detected))}")
            hazard_keywords = ['snow', 'ice', 'slush', 'black ice']
            is_hazardous = any(x.lower() in hazard_keywords for x in detected)
            
            if is_hazardous:
                st.error("🚨 **HAZARD ALERT!**")
                st.write("- Drive below 40 km/h")
                st.write("- Increase braking distance")
            else:
                st.success("✅ **ROAD CLEAR**")
                st.write("Normal winter driving conditions.")
        else:
            st.info("No hazards detected.")
            is_hazardous = False

    # --- Map Section ---
    st.write("---")
    st.subheader("📍 Road Location (Saskatoon Area)")
    m = folium.Map(location=[52.1332, -106.6700], zoom_start=12)
    folium.Marker(
        [52.1332, -106.6700], 
        popup="Hazard Location", 
        icon=folium.Icon(color='red' if is_hazardous else 'green')
    ).add_to(m)
    st_folium(m, width="100%", height=300)

# --- Professional Footer ---
st.markdown("""
    <div class="footer-text">
        Developed with ❤️ by <b>Agha Wafa Abbas</b> | Saskatoon Winter Safety Project
    </div>
    """, unsafe_allow_html=True)
