import tensorflow as tf
from tensorflow.keras import layers, models, applications
import os

def train_crop_model(data_dir="dataset"):
    # 1. Load and Preprocess Data
    # Ensure you have folders: dataset/Bacterial, dataset/Fungal, etc.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32,
        label_mode='categorical'
    )

    # 2. Use Transfer Learning (MobileNetV2) - This is the key logic fix
    base_model = applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False # Freeze the pre-trained weights

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(5, activation='softmax') # 5 Classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. Train the model (Actual learning happens here)
    print("🚀 Training started... Make sure your dataset folder is ready!")
    if os.path.exists(data_dir):
        model.fit(train_ds, epochs=10) 
        model.save("crop_disease_cnn.h5")
        print("✅ Success! Trained model saved as 'crop_disease_cnn.h5'")
    else:
        # If no dataset, we save a template (but training is required for accuracy)
        model.save("crop_disease_cnn.h5")
        print("⚠️ No 'dataset' folder found. Saved untrained template. Predictions will be random until trained.")

if __name__ == "__main__":
    train_crop_model()
