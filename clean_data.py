import os
from PIL import Image

def clean_corrupted_images():
    # Gets the folder where this script is running
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folders_to_check = ["PlantVillage", "Train", "Validation"]
    
    removed_count = 0
    scanned_count = 0

    print("🧹 Starting dataset cleanup. This might take a minute...\n")

    for folder_name in folders_to_check:
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            continue

        # Walk through all subfolders
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(root, file)
                    scanned_count += 1
                    
                    try:
                        # Try to open and verify the file headers
                        with Image.open(file_path) as img:
                            img.verify()
                    except Exception as e:
                        # If it fails, the image is corrupted. Delete it.
                        print(f"🗑️ Deleting corrupted file: {file_path}")
                        os.remove(file_path)
                        removed_count += 1

    print(f"\n✨ Cleanup Complete!")
    print(f"📊 Scanned: {scanned_count} images.")
    print(f"💥 Removed: {removed_count} corrupted images.")

if __name__ == "__main__":
    clean_corrupted_images()