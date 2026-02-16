import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
from catboost import Pool
import tensorflow as tf

# ======================================================
# 1. PAGE CONFIG & LANGUAGE
# ======================================================
st.set_page_config(page_title="AgroSmart AI", page_icon="🌾", layout="wide")

# Multilingual Content
LANG_DATA = {
    "English": {
        "title": "🌾 Smart Crop Diagnostic Station",
        "tab1": "📊 Data-Based Analysis",
        "tab2": "📷 Image-Based Analysis",
        "btn": "Analyze Now",
        "result_head": "Diagnosis Result",
        "sol_head": "Recommended Treatment",
        "upload_label": "Upload a photo of the infected leaf",
        "sev": "Severity"
    },
    "Hindi": {
        "title": "🌾 स्मार्ट फसल निदान केंद्र",
        "tab1": "📊 डेटा आधारित विश्लेषण",
        "tab2": "📷 फोटो आधारित विश्लेषण",
        "btn": "अभी विश्लेषण करें",
        "result_head": "निदान परिणाम",
        "sol_head": "अनुशंसित उपचार",
        "upload_label": "संक्रमित पत्ती का फोटो अपलोड करें",
        "sev": "गंभीरता"
    }
}

# Language Selector (Placed at top for visibility)
lang_choice = st.radio("🌐 Choose Language / भाषा चुनें", ["English", "Hindi"], horizontal=True)
t = LANG_DATA[lang_choice]

# ======================================================
# 2. LOAD MODELS
# ======================================================
@st.cache_resource
def load_all():
    cb = joblib.load("catboost_crop_disease_model.pkl")
    le = joblib.load("crop_disease_label_encoder.pkl")
    cnn = tf.keras.models.load_model("crop_disease_cnn.h5")
    return cb, le, cnn

model, label_encoder, cnn_model = load_all()

# UNIFIED KNOWLEDGE BASE (Disease -> Solution)
DISEASE_SOLUTIONS = {
    "Fungal": {"sev": "High", "sol": "Apply Fungicide + Reduce moisture.", "clr": "#ff4b4b"},
    "Bacterial": {"sev": "Medium", "sol": "Use Copper-based sprays + Prune infected parts.", "clr": "#ffa500"},
    "Viral": {"sev": "Extreme", "sol": "No cure. Remove plant immediately to stop spread.", "clr": "#ff0000"},
    "Pest": {"sev": "High", "sol": "Spray Neem Oil + Apply organic pesticides.", "clr": "#f1c40f"},
    "Healthy": {"sev": "None", "sol": "Keep doing what you're doing! Crop is healthy.", "clr": "#2ecc71"}
}

# ======================================================
# 3. UI TABS
# ======================================================
st.title(t["title"])
tab1, tab2 = st.tabs([t["tab1"], t["tab2"]])

# --- TAB 1: CATBOOST (FIELD DATA) ---
with tab1:
    st.header(t["tab1"])
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        crop = st.selectbox("Crop", ["Tomato", "Potato", "Rice", "Maize"])
        temp = st.slider("Temperature (°C)", 10, 50, 25)
        hum = st.slider("Humidity (%)", 10, 100, 60)
        rain = st.slider("Rainfall (mm)", 0, 500, 150)
        ph = st.slider("Soil pH", 4.0, 9.0, 6.5)
        moist = st.slider("Moisture (%)", 0, 100, 40)
        spots = st.radio("Leaf Spots?", ["No", "Yes"])
        wilt = st.radio("Wilting?", ["No", "Yes"])
        
    if st.button(t["btn"], key="btn_data"):
        data = pd.DataFrame({
            "Crop": [crop], "Temperature(C)": [temp], "Humidity(%)": [hum],
            "Rainfall(mm)": [rain], "Soil_pH": [ph], "Soil_Moisture(%)": [moist],
            "Leaf_Spots": [1 if spots=="Yes" else 0], "Wilting": [1 if wilt=="Yes" else 0]
        })
        pool = Pool(data=data, cat_features=["Crop"])
        res_idx = int(model.predict(pool).flatten()[0])
        res_name = label_encoder.inverse_transform([res_idx])[0]
        
        info = DISEASE_SOLUTIONS.get(res_name, DISEASE_SOLUTIONS["Healthy"])
        st.success(f"### {t['result_head']}: {res_name}")
        st.info(f"**{t['sol_head']}:** {info['sol']}")

# --- TAB 2: CNN (IMAGE UPLOAD) ---
with tab2:
    st.header(t["tab2"])
    up_file = st.file_uploader(t["upload_label"], type=["jpg", "jpeg", "png"])

    if up_file:
        img = Image.open(up_file)
        st.image(img, width=400, caption="Uploaded Image")
        
        if st.button(t["btn"], key="btn_img"):
            # Process Image
            img_res = img.resize((224, 224))
            img_arr = np.array(img_res) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            
            # Predict
            preds = cnn_model.predict(img_arr)
            # CNN classes must match the order in train_cnn.py
            classes = ["Bacterial", "Fungal", "Healthy", "Pest", "Viral"]
            res_name = classes[np.argmax(preds)]
            
            # Display Result + Solution
            info = DISEASE_SOLUTIONS.get(res_name, DISEASE_SOLUTIONS["Healthy"])
            
            st.markdown(f"### {t['result_head']}: <span style='color:{info['clr']}'>{res_name}</span>", unsafe_allow_html=True)
            
            # THE SOLUTION BOX
            st.markdown(f"""
                <div style="background-color:{info['clr']}20; padding:20px; border-radius:10px; border-left: 10px solid {info['clr']};">
                    <h4>✅ {t['sol_head']}</h4>
                    <p style="font-size:18px;">{info['sol']}</p>
                    <strong>{t['sev']}:</strong> {info['sev']}
                </div>
            """, unsafe_allow_html=True)
