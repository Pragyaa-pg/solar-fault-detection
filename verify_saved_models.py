import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

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

for name, path in [("mobilenet_v3", "mobilenet_v3_solar_fault_model.pth"),
                    ("resnet50", "resnet50_solar_fault_model.pth")]:
    model = get_model(name)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()

    preds, labels = [], []
    with torch.no_grad():
        for images, lbls in val_loader:
            images = images.to(device)
            out = model(images)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            labels.extend(lbls.numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    print(f"{name}: Accuracy = {acc*100:.2f}%, Macro-F1 = {f1*100:.2f}%")