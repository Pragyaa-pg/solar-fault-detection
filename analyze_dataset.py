import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ---- 0. Auto-detect metadata (same as before) ----
matches = list(Path(".").rglob("module_metadata.json"))
if not matches:
    raise FileNotFoundError("module_metadata.json nahi mili. Pehle dataset extract karo.")
JSON_PATH = matches[0]
DATA_DIR = JSON_PATH.parent

with open(JSON_PATH, "r") as f:
    metadata = json.load(f)

df = pd.DataFrame([
    {"image_id": k, "filepath": DATA_DIR / v["image_filepath"], "class": v["anomaly_class"]}
    for k, v in metadata.items()
])

print(f"Total images: {len(df)}\n")

# =========================================================
# 1. CLASS DISTRIBUTION ANALYSIS
# =========================================================
print("=" * 60)
print("1. CLASS DISTRIBUTION")
print("=" * 60)

class_counts = df["class"].value_counts()
print(class_counts)
print(f"\nImbalance ratio (majority/minority): {class_counts.max() / class_counts.min():.1f}x")

fig, ax = plt.subplots(figsize=(10, 5))
class_counts.plot(kind="bar", ax=ax, color="steelblue")
ax.set_ylabel("Number of Images")
ax.set_title("Class Distribution")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("1_class_distribution.png")
plt.close()
print("Saved: 1_class_distribution.png\n")

# =========================================================
# 2. PIXEL INTENSITY / BRIGHTNESS ANALYSIS
# =========================================================
print("=" * 60)
print("2. PIXEL INTENSITY ANALYSIS (sample-based for speed)")
print("=" * 60)

# Manual sampling (groupby().apply() ka safer alternative)
sample_frames = []
for cls in df["class"].unique():
    class_df = df[df["class"] == cls]
    n_samples = min(len(class_df), 200)
    sample_frames.append(class_df.sample(n_samples, random_state=42))

sample_df = pd.concat(sample_frames, ignore_index=True)

brightness_data = defaultdict(list)
error_count = 0

for _, row in sample_df.iterrows():
    try:
        img = np.array(Image.open(row["filepath"]).convert("L"))  # grayscale
        brightness_data[row["class"]].append(img.mean())
    except Exception as e:
        error_count += 1
        if error_count <= 5:  # sirf pehli 5 errors dikhao, spam avoid karne ke liye
            print(f"Error reading {row['filepath']}: {e}")

if error_count > 5:
    print(f"... aur {error_count - 5} aur errors (total: {error_count})")

# Mean brightness per class
print("\nMean brightness per class:")
for cls, vals in brightness_data.items():
    if vals:  # empty na ho
        print(f"  {cls:20s}: {np.mean(vals):.2f} (std: {np.std(vals):.2f})")

# Boxplot comparison (matplotlib version-safe)
fig, ax = plt.subplots(figsize=(12, 6))
labels_list = list(brightness_data.keys())
values_list = list(brightness_data.values())

try:
    # Newer matplotlib (3.9+)
    ax.boxplot(values_list, tick_labels=labels_list)
except TypeError:
    # Older matplotlib
    ax.boxplot(values_list, labels=labels_list)

ax.set_ylabel("Mean Pixel Brightness")
ax.set_title("Brightness Distribution by Class")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("2_brightness_by_class.png")
plt.close()
print("\nSaved: 2_brightness_by_class.png\n")
# =========================================================
# 3. DATA QUALITY CHECK (corrupted, duplicates, dimensions)
# =========================================================
print("=" * 60)
print("3. DATA QUALITY CHECK")
print("=" * 60)

corrupted = []
dimensions = Counter()
hashes = defaultdict(list)

# Testing ke liye pehle sirf 2000 images pe chalao (fast)
# Jab confirm ho jaye ki sahi kaam kar raha hai, tab check_df = df kar dena (full dataset)
check_df = df.sample(min(2000, len(df)), random_state=42)

total = len(check_df)
print(f"Checking {total} images...\n")

for i, (_, row) in enumerate(check_df.iterrows()):
    # Har 200 images ke baad progress dikhao
    if i % 200 == 0:
        print(f"  Progress: {i}/{total} images checked...")

    try:
        img = Image.open(row["filepath"])
        dimensions[img.size] += 1
        img_hash = hashlib.md5(img.tobytes()).hexdigest()
        hashes[img_hash].append(row["image_id"])
    except Exception as e:
        corrupted.append((row["filepath"], str(e)))

print(f"\n✅ Done checking {total} images.\n")

print(f"Corrupted/unreadable images: {len(corrupted)}")
if corrupted:
    for path, err in corrupted[:5]:
        print(f"  - {path}: {err}")

print(f"\nImage dimensions found:")
for dim, count in dimensions.most_common():
    print(f"  {dim}: {count} images")

duplicates = {h: ids for h, ids in hashes.items() if len(ids) > 1}
print(f"\nDuplicate images found: {len(duplicates)} groups")
if duplicates:
    total_dupes = sum(len(ids) - 1 for ids in duplicates.values())
    print(f"  Total redundant copies: {total_dupes}")
# =========================================================
# 4. VISUAL CLASS COMPARISON (with colormap for clarity)
# =========================================================
print("\n" + "=" * 60)
print("4. VISUAL COMPARISON")
print("=" * 60)

classes = sorted(df["class"].unique())
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for ax, cls in zip(axes, classes):
    sample_path = df[df["class"] == cls].iloc[0]["filepath"]
    img = Image.open(sample_path).convert("L")
    im = ax.imshow(img, cmap="inferno")  # heatmap-style, defects clearly dikhenge
    ax.set_title(cls, fontsize=11)
    ax.axis("off")

plt.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Intensity")
plt.suptitle("Sample Images per Class (Inferno Colormap)", fontsize=14)
plt.savefig("4_visual_comparison.png", bbox_inches="tight")
plt.close()
print("Saved: 4_visual_comparison.png")

print("\n✅ Analysis complete! Check the generated PNG files.")

# Debug: ek single file manually check karo
test_row = df.iloc[0]
print("Testing path:", test_row["filepath"])
print("Path exists?", test_row["filepath"].exists())