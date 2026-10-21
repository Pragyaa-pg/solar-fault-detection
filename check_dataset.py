import os

DATASET_PATH = r"D:\projects\solar-fault-detection\InfraredSolarModules"

print("--- Root Folder Contents ---")
if os.path.exists(DATASET_PATH):
    for item in os.listdir(DATASET_PATH):
        print(" -", item)
else:
    print("Folder path not found. Please check folder name.")

IMAGE_DIR = os.path.join(DATASET_PATH, "images")
print("\n--- Image Samples ---")
if os.path.exists(IMAGE_DIR):
    images = os.listdir(IMAGE_DIR)[:10]  # Get first 10 image names
    for img in images:
        print(" -", img)
else:
    print("Images folder not found.")