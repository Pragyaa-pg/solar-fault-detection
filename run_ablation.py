import os
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ============================================================
# ★★★ CHANGE THESE PATHS ACCORDING TO YOUR FOLDER ★★★
# ============================================================
DATA_DIR = Path("./InfraredSolarModules")               # folder containing images/ and metadata
META_FILE = DATA_DIR / "module_metadata.json"

# If your structure is different, examples:
# DATA_DIR = Path("D:/Datasets/InfraredSolarModules")
# META_FILE = DATA_DIR / "module_metadata.json"

# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 40                    # reduce to 25-30 if you want faster runs
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 456]             # 3 seeds for mean ± std
TEST_SIZE = 0.1                    # ~2000 test images

CLASS_NAMES = [
    "Cell", "Cell-Multi", "Cracking", "Diode", "Diode-Multi",
    "Hot-Spot", "Hot-Spot-Multi", "No-Anomaly", "Offline-Module",
    "Shadowing", "Soiling", "Vegetation"
]

# ============================================================
# DATASET CLASS
# ============================================================
class SolarDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def load_splits():
    print(f"Loading metadata from: {META_FILE}")
    with open(META_FILE, "r") as f:
        meta = json.load(f)

    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    samples = []

    for key, info in meta.items():
        if "image_filepath" in info:
            rel_path = info["image_filepath"]
        else:
            rel_path = f"images/{key}.jpg"

        img_path = DATA_DIR / rel_path
        if not img_path.exists():
            img_path = DATA_DIR / "images" / f"{key}.jpg"
        if not img_path.exists():
            continue

        label = class_to_idx[info["anomaly_class"]]
        samples.append((str(img_path), label))

    print(f"Total valid images: {len(samples)}")

    paths, labels = zip(*samples)
    train_p, test_p, train_l, test_l = train_test_split(
        paths, labels, test_size=TEST_SIZE, stratify=labels, random_state=42
    )
    print(f"Train samples: {len(train_p)} | Test samples: {len(test_p)}")
    return list(zip(train_p, train_l)), list(zip(test_p, test_l))

# ============================================================
# TRANSFORMS
# ============================================================
def get_transform(augment=False):
    t = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if augment:
        t += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(contrast=0.2),
        ]
    t += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(t)

# ============================================================
# MODEL CREATION
# ============================================================
def create_model(name):
    if name == "mobilenet":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        raise ValueError("Model must be 'mobilenet' or 'resnet50'")
    return model.to(DEVICE)

# ============================================================
# SINGLE TRAINING RUN
# ============================================================
def run_one_experiment(model_name, config, train_samples, test_samples, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_aug = config in ["Aug_only", "Full"]
    use_wrs = config in ["WRS_only", "Full"]

    train_tf = get_transform(augment=use_aug)
    test_tf  = get_transform(augment=False)

    train_ds = SolarDataset(train_samples, train_tf)
    test_ds  = SolarDataset(test_samples, test_tf)

    train_labels = [s[1] for s in train_samples]

    if use_wrs:
        counts = Counter(train_labels)
        weights = [1.0 / counts[y] for y in train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = create_model(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_macro_f1 = 0.0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"{model_name}-{config} | Ep {epoch+1}/{NUM_EPOCHS}", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images.to(DEVICE))
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        current_f1 = f1_score(all_labels, all_preds, average="macro")
        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            best_state = model.state_dict().copy()

    # Final evaluation with best model
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    return {
        "model": model_name,
        "config": config,
        "seed": seed,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if not META_FILE.exists():
        raise FileNotFoundError(f"Cannot find metadata file:\n{META_FILE}\nPlease correct the path.")

    train_samples, test_samples = load_splits()

    # Both models
    models_list = ["mobilenet", "resnet50"]
    configs = ["Baseline", "Aug_only", "WRS_only", "Full"]

    all_results = []

    for model_name in models_list:
        for config in configs:
            for seed in SEEDS:
                print(f"\n{'='*70}")
                print(f"Training → {model_name.upper()} | {config} | seed = {seed}")
                print(f"{'='*70}")
                
                result = run_one_experiment(model_name, config, train_samples, test_samples, seed)
                all_results.append(result)
                
                print(f"Result → Accuracy: {result['accuracy']:.4f} | "
                      f"Macro-F1: {result['macro_f1']:.4f} | "
                      f"Weighted-F1: {result['weighted_f1']:.4f}")

    # ----------------------------------------------------------
    # Create Table IV
    # ----------------------------------------------------------
    df = pd.DataFrame(all_results)

    summary = df.groupby(["model", "config"]).agg(
        Acc_mean=("accuracy", "mean"),
        Acc_std=("accuracy", "std"),
        F1_mean=("macro_f1", "mean"),
        F1_std=("macro_f1", "std"),
        WF1_mean=("weighted_f1", "mean"),
        WF1_std=("weighted_f1", "std")
    ).reset_index()

    def format_mean_std(mean, std):
        return f"{mean:.2f} ± {std:.2f}"

    summary["Accuracy"] = summary.apply(lambda x: format_mean_std(x["Acc_mean"], x["Acc_std"]), axis=1)
    summary["Macro-F1"] = summary.apply(lambda x: format_mean_std(x["F1_mean"], x["F1_std"]), axis=1)
    summary["Weighted-F1"] = summary.apply(lambda x: format_mean_std(x["WF1_mean"], x["WF1_std"]), axis=1)

    final_table = summary[["model", "config", "Accuracy", "Macro-F1", "Weighted-F1"]].copy()
    
    # Nice names
    final_table["model"] = final_table["model"].replace({
        "mobilenet": "MobileNetV3",
        "resnet50": "ResNet-50"
    })
    final_table["config"] = final_table["config"].replace({
        "Baseline": "Baseline",
        "Aug_only": "+Aug only",
        "WRS_only": "+WRS only",
        "Full": "Full (WRS + Aug)"
    })
    final_table.columns = ["Model", "Configuration", "Accuracy", "Macro-F1", "Weighted-F1"]

    print("\n\n" + "="*90)
    print("TABLE IV – Ablation Study Results (Mean ± Std)")
    print("="*90)
    print(final_table.to_string(index=False))
    print("="*90)

    # Save files
    final_table.to_csv("Table_IV_Ablation.csv", index=False)
    df.to_csv("ablation_all_runs_raw.csv", index=False)
    
    print("\nFiles saved successfully:")
    print("  → Table_IV_Ablation.csv          (ready for paper)")
    print("  → ablation_all_runs_raw.csv      (all individual runs)")