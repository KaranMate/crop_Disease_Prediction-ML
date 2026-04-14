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
        height: 3.5em;
        width: 100%;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    
    /* Button Hover Effect */
    div.stButton > button:first-child:hover {
        background-color: #218838 !important;
        border: 1px solid #1e7e34;
    }

    /* Make Text highly readable on white */
    h1, h2, h3, h4, p, label {
        color: #1e1e1e !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: bold;
        font-size: 16px;
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
        "upload_label": "Upload a leaf photo",
        "conf": "Confidence Score"
    },
    "Hindi": {
        "title": "🌾 स्मार्ट फसल निदान केंद्र",
        "tab1": "📊 डेटा आधारित विश्लेषण",
        "tab2": "📷 फोटो आधारित विश्लेषण",
        "btn": "जांच शुरू करें",
        "res_head": "जांच का परिणाम",
        "sol_head": "उपचार योजना",
        "upload_label": "पत्ती का फोटो अपलोड करें",
        "conf": "सटीकता स्कोर"
    }
}

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

model, label_encoder, cnn_model = load_assets()

DISEASE_SOLUTIONS = {
    "Fungal": {"msg": "Spray Potassium + Fungicide (Mancozeb). Reduce watering.", "clr": "#d9534f"},
    "Bacterial": {"msg": "Apply Copper-based spray (COC). Prune infected leaves.", "clr": "#f0ad4e"},
    "Viral": {"msg": "No cure. Remove and burn the plant immediately to prevent spread.", "clr": "#c9302c"},
    "Pest": {"msg": "Apply Neem Oil or Imidacloprid insecticide.", "clr": "#ec971f"},
    "Healthy": {"msg": "No treatment needed. Keep maintaining current care.", "clr": "#28a745"}
}

# ======================================================
# 4. MAIN INTERFACE
# ======================================================
st.title(t["title"])
tab1, tab2 = st.tabs([t["tab1"], t["tab2"]])

with tab1:
    st.header(t["tab1"])
    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("Crop", ["Tomato", "Potato", "Rice", "Maize"])
        temp = st.slider("Temperature (°C)", 10, 50, 25)
        hum = st.slider("Humidity (%)", 10, 100, 60)
    with col2:
        spots = st.radio("Leaf Spots?", ["No", "Yes"])
        wilt = st.radio("Wilting?", ["No", "Yes"])

    if st.button(t["btn"], key="btn1"):
        # Simulated Data Logic
        input_data = pd.DataFrame({
            "Crop": [crop], "Temperature(C)": [temp], "Humidity(%)": [hum],
            "Rainfall(mm)": [100], "Soil_pH": [6.5], "Soil_Moisture(%)": [40],
            "Leaf_Spots": [1 if spots=="Yes" else 0], "Wilting": [1 if wilt=="Yes" else 0]
        })
        res_idx = int(model.predict(Pool(input_data, cat_features=["Crop"])).flatten()[0])
        res_name = label_encoder.inverse_transform([res_idx])[0]
        
        info = DISEASE_SOLUTIONS.get(res_name, DISEASE_SOLUTIONS["Healthy"])
        st.success(f"### {t['res_head']}: {res_name}")
        st.info(f"**{t['sol_head']}:** {info['msg']}")

with tab2:
    st.header(t["tab2"])
    up_file = st.file_uploader(t["upload_label"], type=["jpg", "png", "jpeg"])

    if up_file:
        img = Image.open(up_file)
        st.image(img, width=400, caption="Uploaded Image")
        
        if st.button(t["btn"], key="btn2"):
            # 1. Preprocess
            img_res = img.resize((224, 224))
            img_arr = np.array(img_res) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            
            # 2. Predict
            preds = cnn_model.predict(img_arr)
            confidence = np.max(preds) * 100
            
            # Match class names alphabetically (standard Keras folder loading)
            classes = ["Bacterial", "Fungal", "Healthy", "Pest", "Viral"]
            res_name = classes[np.argmax(preds)]
            
            # 3. Enhanced Solution Card
            info = DISEASE_SOLUTIONS.get(res_name, DISEASE_SOLUTIONS["Healthy"])
            
            st.markdown(f"""
                <div style="background-color:white; padding:25px; border-radius:15px; border: 3px solid {info['clr']}; margin-top:20px; box-shadow: 0px 4px 15px rgba(0,0,0,0.05);">
                    <h2 style="color:{info['clr']}; margin-bottom:0;">{t['res_head']}: {res_name}</h2>
                    <p style="color:gray;">{t['conf']}: {confidence:.2f}%</p>
                    <hr style="border: 0.5px solid #eee;">
                    <h4 style="color:black;">🩺 {t['sol_head']}</h4>
                    <p style="font-size:19px; color:#333; line-height:1.6;">{info['msg']}</p>
                </div>
            """, unsafe_allow_html=True)
