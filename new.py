import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# --- Configuration & Styling ---
st.set_page_config(page_title="Crop Classification", page_icon="🌱", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Constants ---
IMG_SIZE = (224, 224) 
MODEL_PATH = 'best_crop_model.keras'
CLASS_NAMES_PATH = 'class_names.txt'

@st.cache_resource
def load_class_names():
    """Loads class names dynamically from the text file generated during training."""
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            # Read lines and remove any trailing whitespace/newlines
            return [line.strip() for line in f.readlines() if line.strip()]
    else:
        st.warning(f"⚠️ '{CLASS_NAMES_PATH}' not found. Please ensure it is in the same directory.")
        return []

@st.cache_resource
def load_prediction_model():
    """Loads the Keras model once and caches it."""
    if os.path.exists(MODEL_PATH):
        try:
            return load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning(f"Model file '{MODEL_PATH}' not found. Please ensure it is in the same directory.")
        return None

def predict(img, model, class_names):
    """Preprocesses image and returns prediction results."""
    # Resize and convert to array
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    
    # Expand dimensions to create batch (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Note: Removed img_array / 255.0 because EfficientNetB0 expects 0-255 pixel 
    # values and applies its own internal normalization.
    
    predictions = model.predict(img_array)
    
    # The training script already included a softmax activation in the final Dense layer,
    # so `predictions[0]` already contains the probabilities summing to 1.
    class_idx = np.argmax(predictions[0])
    confidence = 100 * predictions[0][class_idx] 
    
    # Safely get the class name
    if class_idx < len(class_names):
        label = class_names[class_idx]
    else:
        label = f"Unknown Class (Index {class_idx})"
        
    return label, confidence

# --- Initialize Data ---
CLASS_NAMES = load_class_names()
model = load_prediction_model()

# --- UI Layout ---
st.title("🌱 Crop Identification AI")
st.write("Upload a clear photo of a crop to identify its type using the deep learning model.")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.info("This app uses a Keras `.keras` model to classify agricultural crops from images.")
    if CLASS_NAMES:
        st.write(f"Supported classes ({len(CLASS_NAMES)}):")
        with st.expander("View all classes"):
            for c in CLASS_NAMES:
                st.write(f"- {c}")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        if model is not None and len(CLASS_NAMES) > 0:
            st.subheader("Analysis")
            with st.spinner('Analyzing crop patterns...'):
                label, conf = predict(img, model, CLASS_NAMES)
            
            # Display results
            st.success(f"**Result: {label}**")
            st.metric(label="Confidence", value=f"{conf:.2f}%")
            
            # Progress bar visual
            st.progress(int(conf))
        elif len(CLASS_NAMES) == 0:
            st.error("Class names are missing. Cannot interpret predictions.")
        else:
            st.error("Model is not loaded. Cannot perform prediction.")

else:
    st.write("---")
    st.write("Please upload an image file to begin.")

# Footer
st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes. Always consult an agronomist for critical farming decisions.")