import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from statsmodels.stats.contingency_tables import mcnemar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SolarDataset(torch.utils.data.Dataset):
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

transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_df = pd.read_csv("train.csv")
classes = sorted(train_df["label"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)

val_dataset = SolarDataset("val.csv", class_to_idx, transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def get_model(name):
    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m

def get_preds(name, path):
    model = get_model(name)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, lbls in val_loader:
            images = images.to(device)
            out = model(images)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            labels.extend(lbls.numpy())
    return np.array(preds), np.array(labels)

preds_mobilenet, labels = get_preds("mobilenet_v3", "mobilenet_v3_solar_fault_model.pth")
preds_resnet, _ = get_preds("resnet50", "resnet50_solar_fault_model.pth")

correct_mobilenet = (preds_mobilenet == labels)
correct_resnet = (preds_resnet == labels)

# Contingency table for McNemar's test
both_correct = np.sum(correct_mobilenet & correct_resnet)
mobilenet_only = np.sum(correct_mobilenet & ~correct_resnet)
resnet_only = np.sum(~correct_mobilenet & correct_resnet)
both_wrong = np.sum(~correct_mobilenet & ~correct_resnet)

table = [[both_correct, mobilenet_only],
         [resnet_only, both_wrong]]

result = mcnemar(table, exact=False, correction=True)

print("Contingency Table:")
print(f"  Both correct: {both_correct}")
print(f"  MobileNetV3 correct, ResNet50 wrong: {mobilenet_only}")
print(f"  ResNet50 correct, MobileNetV3 wrong: {resnet_only}")
print(f"  Both wrong: {both_wrong}")
print(f"\nMcNemar's test statistic: {result.statistic:.4f}")
print(f"p-value: {result.pvalue:.4f}")

if result.pvalue < 0.05:
    print("\n=> Statistically significant difference between the two models (p < 0.05)")
else:
    print("\n=> No statistically significant difference between the two models (p >= 0.05)")