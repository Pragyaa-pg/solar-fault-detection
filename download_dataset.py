import json
import os
import shutil
from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ---- 0. Auto-detect the metadata file location ----
# This searches the current directory and subfolders for module_metadata.json,
# so it works no matter how the zip/git-clone nested the folders.
matches = list(Path(".").rglob("module_metadata.json"))

if not matches:
    raise FileNotFoundError(
        "Could not find module_metadata.json anywhere under the current directory. "
        "Make sure you've extracted/cloned the InfraredSolarModules dataset here, "
        "and that you're running this script from the right folder."
    )

JSON_PATH = matches[0]
DATA_DIR = JSON_PATH.parent          # the folder that directly contains images/ and the json
IMAGES_DIR = DATA_DIR / "images"

print(f"Found metadata at: {JSON_PATH.resolve()}")
print(f"Using DATA_DIR: {DATA_DIR.resolve()}")

# ---- 1. Load metadata ----
with open(JSON_PATH, "r") as f:
    metadata = json.load(f)

print(f"Total images in metadata: {len(metadata)}")

# Convert to a DataFrame for easy analysis
df = pd.DataFrame([
    {
        "image_id": k,
        "filepath": DATA_DIR / v["image_filepath"],
        "class": v["anomaly_class"],
    }
    for k, v in metadata.items()
])

print(df.head())

# ---- 2. Class distribution ----
class_counts = df["class"].value_counts()
print("\nClass distribution:")
print(class_counts)

class_counts.plot(kind="bar", figsize=(10, 5), title="Class distribution")
plt.ylabel("Number of images")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()

# ---- 3. Preview a few sample images per class ----
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()
for ax, cls in zip(axes, df["class"].unique()):
    sample_path = df[df["class"] == cls].iloc[0]["filepath"]
    img = Image.open(sample_path)
    ax.imshow(img, cmap="gray")
    ax.set_title(cls)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_images.png")
plt.show()

# ---- 4. (Optional) Reorganize images into class subfolders ----
# Useful for frameworks like Keras' ImageDataGenerator / flow_from_directory,
# or PyTorch's ImageFolder, which expect folder-per-class structure.
OUTPUT_DIR = Path("InfraredSolarModules_by_class")
OUTPUT_DIR.mkdir(exist_ok=True)

for _, row in df.iterrows():
    class_dir = OUTPUT_DIR / row["class"]
    class_dir.mkdir(parents=True, exist_ok=True)
    dest = class_dir / os.path.basename(row["filepath"])
    if not dest.exists():
        shutil.copy(row["filepath"], dest)

print("\nReorganized into class folders at:", OUTPUT_DIR.resolve())

# ---- 5. Train/val/test split (stratified) ----
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["class"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["class"], random_state=42
)

print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)