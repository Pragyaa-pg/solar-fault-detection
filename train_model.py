import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

# =========================================================
# STEP 1: Device check (GPU hai ya sirf CPU)
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================================================
# STEP 2: Dataset class (image + label ko load karega)
# =========================================================
class SolarDataset(Dataset):
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

# =========================================================
# STEP 3: Data transforms — CHANGED: sirf halka flip, heavy
# augmentation (rotation, color-jitter) hata diya
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

# =========================================================
# STEP 4: CSV files load karo (jo pehle bana chuke ho)
# =========================================================
train_df_check = pd.read_csv("train.csv")
classes = sorted(train_df_check["label"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
num_classes = len(classes)
print(f"Classes ({num_classes}): {classes}")

train_dataset = SolarDataset("train.csv", class_to_idx, transform)
val_dataset = SolarDataset("val.csv", class_to_idx, transform)

# =========================================================
# STEP 5: Class imbalance ke weights calculate karo (loss ke liye)
# =========================================================
class_counts = train_df_check["label"].map(class_to_idx).value_counts().sort_index()
weights = 1.0 / class_counts.values
weights = weights / weights.sum() * num_classes
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
print(f"Class weights (imbalance handling): {class_weights}")

# =========================================================
# STEP 6: DataLoaders — CHANGED: sampler HATA diya, ab sirf
# class-weighted loss se hi imbalance handle hoga
# =========================================================
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# =========================================================
# STEP 7: Model choose karo (yahan switch kar sakte ho)
# =========================================================
def get_model(model_name, num_classes):
    if model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError("model_name must be 'resnet50' or 'mobilenet_v3'")
    return model

# --------- CHANGED: ab mobilenet_v3 improve kar rahe hain -----------
MODEL_NAME = "resnet50"
# -----------------------------------------------------------------------

model = get_model(MODEL_NAME, num_classes).to(device)

# =========================================================
# STEP 8: Loss, optimizer aur scheduler set karo
# =========================================================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# NAYA: Learning rate scheduler — plateau hone par LR kam karega
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

# =========================================================
# STEP 9: TRAINING LOOP (ab best-checkpoint tracking ke saath)
# =========================================================
NUM_EPOCHS = 25
best_val_acc = 0.0   # NAYA: best accuracy track karne ke liye

for epoch in range(NUM_EPOCHS):
    # ---- Training phase ----
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ---- Validation phase ----
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average="macro")
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} "
          f"| Val Accuracy: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}% | LR: {current_lr:.6f}")

    # NAYA: Scheduler ko update karo
    scheduler.step(val_acc)

    # NAYA: Best model checkpointing — sirf tab save hoga jab naya best mile
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{MODEL_NAME}_solar_fault_model.pth")
        print(f"  → New best model saved (Val Acc: {val_acc*100:.2f}%)")

print(f"\nBest validation accuracy achieved: {best_val_acc*100:.2f}%")

# =========================================================
# STEP 10: Best checkpoint reload karke final report banao
# =========================================================
model.load_state_dict(torch.load(f"{MODEL_NAME}_solar_fault_model.pth", weights_only=True))
model.eval()
val_preds, val_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

print("\n" + "=" * 60)
print("FINAL CLASSIFICATION REPORT (Best Checkpoint)")
print("=" * 60)
target_names = [idx_to_class[i] for i in range(num_classes)]
print(classification_report(val_labels, val_preds, target_names=target_names))

print(f"\nModel saved as: {MODEL_NAME}_solar_fault_model.pth (best checkpoint)")