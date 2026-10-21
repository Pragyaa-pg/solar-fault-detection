import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = r"D:\projects\solar-fault-detection\InfraredSolarModules"
JSON_PATH = os.path.join(BASE_DIR, "module_metadata.json")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# 1. Load JSON Metadata
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# 2. Extract image paths and corresponding defect labels
records = []
for key, item in data.items():
    # Construct filename (e.g. 0.jpg, 1.jpg)
    img_filename = f"{item['image_filepath']}"  
    full_path = os.path.join(IMAGES_DIR, os.path.basename(img_filename))
    
    # Class label is inside 'anomaly_class'
    label = item.get("anomaly_class")
    
    if os.path.exists(full_path) and label:
        records.append({"image_path": full_path, "label": label})

df = pd.DataFrame(records)
print(f"Total valid images mapped: {len(df)}")
print(f"Classes found ({df['label'].nunique()}): {df['label'].unique().tolist()}")

# 3. Split into 80% Train, 10% Validation, 10% Test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

# 4. Save to CSV files
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"\nSuccess! Saved:")
print(f" - train.csv ({len(train_df)} samples)")
print(f" - val.csv ({len(val_df)} samples)")
print(f" - test.csv ({len(test_df)} samples)")