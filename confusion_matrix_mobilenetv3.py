import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================================================
# Dataset
# =========================================================
class SolarDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, class_to_idx, transform):
        self.df = pd.read_csv(csv_path)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        label = self.class_to_idx[row["label"]]
        return img, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_df = pd.read_csv("train.csv")
classes = sorted(train_df["label"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
num_classes = len(classes)

print(f"Classes ({num_classes}): {classes}")

val_dataset = SolarDataset("val.csv", class_to_idx, transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# =========================================================
# Model loader
# =========================================================
def get_model(model_name, num_classes):
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

# =========================================================
# Get predictions + probabilities
# =========================================================
def get_predictions(model_name, weight_file):
    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(weight_file, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )

# =========================================================
# Confusion Matrix (Counts + Normalized)
# =========================================================
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # row-normalized

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- Counts ---
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=axes[0], cbar=True
    )
    axes[0].set_title(f"Confusion Matrix (Counts)\n{model_name}", fontsize=13)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # --- Normalized (%) ---
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=axes[1], cbar=True, vmin=0, vmax=1
    )
    axes[1].set_title(f"Confusion Matrix (Normalized %)\n{model_name}", fontsize=13)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    filename = f"confusion_matrix_{model_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")

    # Accuracy from CM
    acc = np.trace(cm) / np.sum(cm)
    print(f"{model_name} Accuracy: {acc*100:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

# =========================================================
# ROC Curve (Per-class + Macro + Micro)
# =========================================================
def plot_roc_curve(y_true, y_probs, model_name):
    # Binarize labels for one-vs-rest
    y_bin = label_binarize(y_true, classes=range(num_classes))
    if num_classes == 2:
        y_bin = np.hstack((1 - y_bin, y_bin))  # safety for binary case

    fpr, tpr, roc_auc = {}, {}, {}

    # Per-class ROC
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # ---------- Plot ----------
    plt.figure(figsize=(10, 8))

    # Micro & Macro
    plt.plot(fpr["micro"], tpr["micro"],
             label=f"Micro-average (AUC = {roc_auc['micro']:.3f})",
             color="deeppink", linestyle=":", linewidth=3)

    plt.plot(fpr["macro"], tpr["macro"],
             label=f"Macro-average (AUC = {roc_auc['macro']:.3f})",
             color="navy", linestyle=":", linewidth=3)

    # Per-class curves
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.8,
                 label=f"{classes[i]} (AUC = {roc_auc[i]:.3f})")

    # Diagonal
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves - {model_name}", fontsize=14)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"roc_curve_{model_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")
    print(f"Macro AUC: {roc_auc['macro']:.4f} | Micro AUC: {roc_auc['micro']:.4f}\n")

# =========================================================
# MAIN
# =========================================================
models_to_eval = [
    ("mobilenet_v3", "mobilenet_v3_solar_fault_model.pth"),
    ("resnet50", "resnet50_solar_fault_model.pth"),
]

for model_name, weight_file in models_to_eval:
    print("=" * 60)
    print(f"Processing: {model_name}")
    print("=" * 60)

    y_true, y_pred, y_probs = get_predictions(model_name, weight_file)

    plot_confusion_matrix(y_true, y_pred, model_name)
    plot_roc_curve(y_true, y_probs, model_name)

print("✅ Done! Both confusion matrices (counts + normalized) and full ROC curves generated.")