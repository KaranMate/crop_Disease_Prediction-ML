import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
from catboost import Pool
import tensorflow as tf

# ======================================================
# 1. UI CUSTOMIZATION (White Background & Green Buttons)
# ======================================================
st.set_page_config(page_title="AgroSmart AI", page_icon="🌾", layout="wide")

st.markdown("""
    <style>
    /* Force White Background */
    .stApp {
        background-color: white !important;
    }
    
    /* Style All Buttons to Green */
    div.stButton > button:first-child {
        background-color: #28a745 !important;
        color: white !important;
        border-radius: 8px;
        border: none;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    
    /* Button Hover Effect */
    div.stButton > button:first-child:hover {
        background-color: #218838 !important;
        border: 1px solid #1e7e34;
    }

    /* Make Text highly readable on white */
    h1, h2, h3, p {
        color: #1e1e1e !important;
    }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# 2. MULTI-LANGUAGE DATA
# ======================================================
LANG_DATA = {
    "English": {
        "title": "🌾 Smart Crop Diagnostic Station",
        "tab1": "📊 Data-Based Analysis",
        "tab2": "📷 Image-Based Analysis",
        "btn": "Run AI Diagnosis",
        "res_head": "Diagnosis Result",
        "sol_head": "Treatment Plan",
        "upload_label": "Upload a leaf photo"
    },
    "Hindi": {
        "title": "🌾 स्मार्ट फसल निदान केंद्र",
        "tab1": "📊 डेटा आधारित विश्लेषण",
        "tab2": "📷 फोटो आधारित विश्लेषण",
        "btn": "जांच शुरू करें",
        "res_head": "जांच का परिणाम",
        "sol_head": "उपचार योजना",
        "upload_label": "पत्ती का फोटो अपलोड करें"
    }
}

# Language Toggle (High Visibility at the Top)
lang_choice = st.radio("🌐 Language / भाषा चुनें", ["English", "Hindi"], horizontal=True)
t = LANG_DATA[lang_choice]

# ======================================================
# 3. MODELS & SOLUTIONS
# ======================================================
@st.cache_resource
def load_assets():
    cb = joblib.load("catboost_crop_disease_model.pkl")
    le = joblib.load("crop_disease_label_encoder.pkl")
    cnn = tf.keras.models.load_model("crop_disease_cnn.h5")
    return cb, le, cnn

cb_model, label_encoder, cnn_model = load_assets()

# SOLUTION DATABASE
DISEASE_SOLUTIONS = {
    "Fungal": {"msg": "Spray Potassium + Fungicide. Reduce watering.", "clr": "#d9534f"},
    "Bacterial": {"msg": "Apply Copper-based spray. Prune dead leaves.", "clr": "#f0ad4e"},
    "Viral": {"msg": "No cure. Uproot and burn the plant immediately.", "clr": "#c9302c"},
    "Pest": {"msg": "Apply Neem Oil or Organic Pesticide.", "clr": "#ec971f"},
    "Healthy": {"msg": "No treatment needed. Maintain health.", "clr": "#5cb85c"}
}

# ======================================================
# 4. MAIN INTERFACE
# ======================================================
st.title(t["title"])
tab1, tab2 = st.tabs([t["tab1"], t["tab2"]])

with tab1:
    st.header(t["tab1"])
    # (Data Input Logic...)
    if st.button(t["btn"], key="btn1"):
        # Prediction Logic for CatBoost
        st.write("Processing Data...")

with tab2:
    st.header(t["tab2"])
    up_file = st.file_uploader(t["upload_label"], type=["jpg", "png", "jpeg"])

    if up_file:
        img = Image.open(up_file)
        st.image(img, width=300, caption="Uploaded Image")
        
        if st.button(t["btn"], key="btn2"):
            # 1. Preprocess
            img_res = img.resize((224, 224))
            img_arr = np.array(img_res) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            
            # 2. Predict (CRITICAL: Class Order must match training folders)
            preds = cnn_model.predict(img_arr)
            
            # Alphabetical order is usually: Bacterial, Fungal, Healthy, Pest, Viral
            classes = ["Bacterial", "Fungal", "Healthy", "Pest", "Viral"]
            res_name = classes[np.argmax(preds)]
            
            # 3. Show Result and Solution Box
            info = DISEASE_SOLUTIONS.get(res_name, DISEASE_SOLUTIONS["Healthy"])
            
            st.markdown(f"""
                <div style="background-color:white; padding:20px; border-radius:10px; border: 2px solid {info['clr']}; margin-top:20px;">
                    <h2 style="color:{info['clr']};">{t['res_head']}: {res_name}</h2>
                    <h4 style="color:black;">🩺 {t['sol_head']}</h4>
                    <p style="font-size:18px; color:#333;">{info['msg']}</p>
                </div>
            """, unsafe_allow_html=True)
