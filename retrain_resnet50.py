import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SolarDataset(Dataset):
    def __init__(self, csv_path, class_to_idx, transform):
        self.df = pd.read_csv(csv_path)
        self.class_to_idx = class_to_idx
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        return img, self.class_to_idx[row["label"]]

# ORIGINAL transform — NO heavy augmentation (jo 77% wale run mein tha)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_df = pd.read_csv("train.csv")
classes = sorted(train_df["label"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
num_classes = len(classes)

train_dataset = SolarDataset("train.csv", class_to_idx, transform)
val_dataset = SolarDataset("val.csv", class_to_idx, transform)

# ORIGINAL loader — shuffle=True, NO sampler
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Class weights (loss ke liye, ye tha original 77% run mein)
class_counts = train_df["label"].map(class_to_idx).value_counts().sort_index()
weights = 1.0 / class_counts.values
weights = weights / weights.sum() * num_classes
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

model = models.resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

NUM_EPOCHS = 10   # original wala number

for epoch in range(NUM_EPOCHS):
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
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {train_loss/len(train_loader):.4f} "
          f"| Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}%")

print("\n" + "="*60)
print("FINAL CLASSIFICATION REPORT")
print("="*60)
target_names = [idx_to_class[i] for i in range(num_classes)]
print(classification_report(val_labels, val_preds, target_names=target_names))

torch.save(model.state_dict(), "resnet50_solar_fault_model.pth")
print("\nModel saved (OVERWRITES the bad 64% version with a fresh, correct one)")