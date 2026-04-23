from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from catboost import Pool

app = Flask(__name__)
CORS(app)

# Load assets
cb_model = joblib.load('catboost_crop_disease_model.pkl')
le = joblib.load('crop_disease_label_encoder.pkl') # Use this for BOTH models
cnn_model = tf.keras.models.load_model('best_crop_model.keras')

# Global treatment database
solutions = {
    "Fungal": {"msg": "Spray Potassium + Fungicide (Mancozeb). Reduce watering.", "clr": "#d9534f"},
    "Bacterial": {"msg": "Apply Copper-based spray (COC). Prune infected leaves.", "clr": "#f0ad4e"},
    "Viral": {"msg": "No cure. Remove and burn the plant immediately to prevent spread.", "clr": "#c9302c"},
    "Pest": {"msg": "Apply Neem Oil or Imidacloprid insecticide.", "clr": "#ec971f"},
    "Healthy": {"msg": "No treatment needed. Keep maintaining current care.", "clr": "#28a745"},
    "Unknown": {"msg": "Please upload a clear photo of a plant leaf.", "clr": "#6c757d"}
}

@app.route('/predict/data', methods=['POST'])
def predict_data():
    data = request.json
    df = pd.DataFrame([data])
    pred = cb_model.predict(Pool(df, cat_features=['Crop']))
    disease = le.inverse_transform(pred.astype(int).flatten())[0]
    return jsonify({'disease': disease, 'solution': solutions.get(disease, solutions["Healthy"])})

@app.route('/predict/image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    img = Image.open(file).convert('RGB') # Fix for PNG transparency
    img = img.resize((224, 224))
    
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    
    preds = cnn_model.predict(img_arr)
    idx = np.argmax(preds)
    confidence = float(preds[0][idx])
    
    # THRESHOLD LOGIC: Reject if confidence is low (e.g., a laptop)
    if confidence < 0.50:
        return jsonify({
            'disease': 'Unknown Object',
            'solution': solutions["Unknown"],
            'confidence': confidence
        })

    # Use label_encoder to get the name to ensure sync with CatBoost
    disease = le.inverse_transform([idx])[0]
    
    return jsonify({
        'disease': disease, 
        'solution': solutions.get(disease, solutions["Healthy"]), 
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)