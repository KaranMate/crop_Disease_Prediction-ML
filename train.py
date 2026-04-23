import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
import os

def build_and_train_model():
    # ==========================================
    # 1. SETUP AND HYPERPARAMETERS
    # ==========================================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the root folders you want to pull data from
    train_folder_names = ["PlantVillage", "Train"]
    val_folder_names = ["Validation"]
    
    img_size = (224, 224)
    batch_size = 32
    seed = 42

    print("📂 Scanning directories...")

    # ==========================================
    # 2. SMART CLASS DISCOVERY
    # ==========================================
    all_folders = train_folder_names + val_folder_names
    unique_classes = set()
    
    for folder_name in all_folders:
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.exists(folder_path):
            # Grab all subdirectories (excluding any hidden folders like .ipynb_checkpoints)
            classes_in_folder = [f for f in os.listdir(folder_path) 
                                 if os.path.isdir(os.path.join(folder_path, f)) 
                                 and not f.startswith('.')]
            unique_classes.update(classes_in_folder)

    # Sort alphabetically for consistent one-hot encoding
    class_names = sorted(list(unique_classes))
    num_classes = len(class_names)
    
    if num_classes == 0:
        print("❌ Error: No classes found. Please check your folder structure.")
        return
        
    print(f"\n✅ Master Class List ({num_classes} Classes): {class_names}")

    # ==========================================
    # 3. CUSTOM DATASET PIPELINE (Crash-Proof)
    # ==========================================
    def create_custom_dataset(folder_names, is_training=False):
        all_image_paths = []
        all_labels = []
        
        # Create a dictionary to map class names to numbers
        class_to_index = {name: i for i, name in enumerate(class_names)}

        for folder_name in folder_names:
            folder_path = os.path.join(base_dir, folder_name)
            if not os.path.exists(folder_path):
                continue
                
            for class_name in os.listdir(folder_path):
                class_dir = os.path.join(folder_path, class_name)
                
                # Only process if it's a valid directory in our master list
                if os.path.isdir(class_dir) and class_name in class_to_index:
                    label_idx = class_to_index[class_name]
                    
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            all_image_paths.append(os.path.join(class_dir, img_name))
                            all_labels.append(label_idx)

        if len(all_image_paths) == 0:
            return None

        # Convert to TensorFlow Dataset
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        label_ds = tf.data.Dataset.from_tensor_slices(all_labels)
        ds = tf.data.Dataset.zip((path_ds, label_ds))

        # Shuffle only the training data
        if is_training:
            ds = ds.shuffle(buffer_size=len(all_image_paths), seed=seed)

        # Map file paths to actual images
        def process_path(file_path, label):
            # Read and decode the image (decode_image safely handles jpg, png, bmp)
            img_raw = tf.io.read_file(file_path)
            img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
            img = tf.image.resize(img, img_size)
            img.set_shape([img_size[0], img_size[1], 3]) # Lock the shape for EfficientNet
            
            # Convert label to one-hot encoding
            label = tf.one_hot(label, depth=num_classes)
            return img, label

        # Apply processing in parallel
        AUTOTUNE = tf.data.AUTOTUNE
        ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(AUTOTUNE)
        
        print(f"📦 Loaded {len(all_image_paths)} images from {folder_names}")
        return ds

    print("\n🚀 Building Data Pipelines...")
    train_ds = create_custom_dataset(train_folder_names, is_training=True)
    val_ds = create_custom_dataset(val_folder_names, is_training=False)

    if train_ds is None or val_ds is None:
        print("❌ Error: Failed to load training or validation images.")
        return

    # ==========================================
    # 4. DATA AUGMENTATION
    # ==========================================
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
    ])

    # ==========================================
    # 5. BUILD THE ARCHITECTURE (EfficientNetB0)
    # ==========================================
    print("\n🏗️ Building EfficientNetB0 Model...")
    base_model = applications.EfficientNetB0(
        input_shape=img_size + (3,), 
        include_top=False, 
        weights='imagenet'
    )
    
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=img_size + (3,)),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    # ==========================================
    # 6. CALLBACKS
    # ==========================================
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=4, 
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2, 
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath='best_crop_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # ==========================================
    # 7. PHASE 1: TRAIN THE HEAD
    # ==========================================
    print("\n⭐ PHASE 1: Training classification head...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10, 
        callbacks=[early_stopping, lr_scheduler, checkpoint]
    )

    # ==========================================
    # 8. PHASE 2: FINE-TUNING
    # ==========================================
    print("\n🔧 PHASE 2: Fine-tuning deeper layers...")
    base_model.trainable = True
    
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stopping, lr_scheduler, checkpoint]
    )

    print("\n✨ TRAINING COMPLETE! Best model saved as 'best_crop_model.keras'")
    
    class_names_path = os.path.join(base_dir, "class_names.txt")
    with open(class_names_path, "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"📝 Master class list saved to '{class_names_path}'")

if __name__ == "__main__":
    build_and_train_model()