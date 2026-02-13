import streamlit as st
import pandas as pd
import joblib
from catboost import Pool

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Crop Disease Prediction",
    page_icon="🌾",
    layout="wide"
)

# ======================================================
# LOAD MODEL & ENCODER
# ======================================================
@st.cache_resource
def load_assets():
    # Ensure these files are in the same folder as this script
    model = joblib.load("catboost_crop_disease_model.pkl")
    encoder = joblib.load("crop_disease_label_encoder.pkl")
    return model, encoder

model, label_encoder = load_assets()

# ======================================================
# DISEASE KNOWLEDGE BASE
# ======================================================
DISEASE_INFO = {
    "Fungal": {"severity": "High", "fertilizer": "Potassium + Fungicide", "color": "#ff4b4b"},
    "Bacterial": {"severity": "Medium", "fertilizer": "NPK + Copper Spray", "color": "#ffa500"},
    "Viral": {"severity": "Very High", "fertilizer": "Remove Plant + Compost", "color": "#ff0000"},
    "Pest": {"severity": "Medium", "fertilizer": "Neem Oil + Nitrogen", "color": "#f1c40f"},
    "Healthy": {"severity": "None", "fertilizer": "Normal Maintenance", "color": "#2ecc71"}
}

# ======================================================
# SIDEBAR INPUTS
# ======================================================
st.sidebar.title("🌱 Input Parameters")

crop = st.sidebar.selectbox(
    "Select Crop",
    ["Tomato", "Potato", "Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Apple", "Groundnut", "Mango"]
)

st.sidebar.subheader("🌦 Environment")
temp = st.sidebar.slider("🌡 Temperature (°C)", 10.0, 50.0, 25.0)
hum = st.sidebar.slider("💧 Humidity (%)", 10.0, 100.0, 50.0)
rain = st.sidebar.slider("🌧 Rainfall (mm)", 0.0, 500.0, 150.0)

st.sidebar.subheader("🧪 Soil & Health")
ph = st.sidebar.slider("🧪 Soil pH", 3.0, 10.0, 6.5)
moisture = st.sidebar.slider("💦 Soil Moisture (%)", 0.0, 100.0, 40.0)

# New inputs required by your model
leaf_spots = st.sidebar.radio("🍂 Visible Leaf Spots?", ["No", "Yes"])
wilting = st.sidebar.radio("🥀 Plant Wilting?", ["No", "Yes"])

predict_btn = st.sidebar.button("🔍 Predict Disease")

# ======================================================
# MAIN UI
# ======================================================
st.markdown("<h1 style='text-align:center;'>🌾 Smart Crop Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

if predict_btn:
    # 🔴 NAMES MUST MATCH YOUR NOTEBOOK EXACTLY
    input_df = pd.DataFrame({
        "Crop": [crop],
        "Temperature(C)": [temp],      # Added (C)
        "Humidity(%)": [hum], 
        "Rainfall(mm)": [rain],        # Added (mm)
        "Soil_pH": [ph],
        "Soil_Moisture(%)": [moisture], # Added missing feature
        "Leaf_Spots": [1 if leaf_spots == "Yes" else 0], # Added missing feature
        "Wilting": [1 if wilting == "Yes" else 0]        # Added missing feature
    })

    # Prepare Pool for CatBoost
    # (Only 'Crop' was marked as cat_feature in your notebook)
    input_pool = Pool(data=input_df, cat_features=["Crop"])

    # Prediction
    raw_pred = model.predict(input_pool)
    
    # Handle the array shape (flattening to get the single integer)
    pred_idx = int(raw_pred.flatten()[0])
    disease = label_encoder.inverse_transform([pred_idx])[0]

    # Get UI info
    info = DISEASE_INFO.get(disease, DISEASE_INFO["Healthy"])

    # Display Results
    res_col1, res_col2, res_col3 = st.columns(3)
    
    res_col1.metric("Disease Detected", disease)
    res_col2.metric("Severity Level", info["severity"])
    res_col3.write(f"**Fertilizer:** \n {info['fertilizer']}")

    st.success(f"Prediction Complete for {crop}!")
    
    st.markdown("---")
    st.subheader("📋 Data Sent to Model")
    st.write(input_df)

else:
    st.info("Adjust the sliders and click Predict to see results.")