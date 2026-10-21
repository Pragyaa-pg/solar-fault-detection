"""
Ablation runner for the solar-fault paper.

Usage (from your venv, project folder):
    python ablation_runs.py --model resnet50
    python ablation_runs.py --model mobilenet_v3
    python ablation_runs.py --model resnet50 --include_full   # sanity check run

Every place marked  >>> TODO <<<  must be copied from YOUR training script so
that these runs use exactly the same setup as your "Proposed Full" results.
Results are appended to ablation_results.csv
"""

import argparse, csv, json, os, random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import models, transforms

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
JSON_PATH   = r"D:\projects\solar-fault-detection\InfraredSolarModules\module_metadata.json"
IMAGES_ROOT = r"D:\projects\solar-fault-detection\InfraredSolarModules\images"

CLASS_NAMES = ["Cell", "Cell-Multi", "Cracking", "Diode", "Diode-Multi",
               "Hot-Spot", "Hot-Spot-Multi", "No-Anomaly", "Offline-Module",
               "Shadowing", "Soiling", "Vegetation"]
NUM_CLASSES = len(CLASS_NAMES)

# >>> TODO <<< copy these from your training script
EPOCHS, LR, BATCH_SIZE, IMG_SIZE = 10, 1e-4, 32, 224
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------
# Dataset (reads module_metadata.json)
# ---------------------------------------------------------------
class SolarDataset(Dataset):
    def __init__(self, transform=None):
        with open(JSON_PATH) as f:
            meta = json.load(f)
        c2i = {c: i for i, c in enumerate(CLASS_NAMES)}
        self.samples = [(os.path.basename(v["image_filepath"]), c2i[v["anomaly_class"]])
                        for v in meta.values() if v["anomaly_class"] in c2i]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        name, label = self.samples[i]
        img = Image.open(os.path.join(IMAGES_ROOT, name)).convert("RGB")
        return (self.transform(img) if self.transform else img), label


def get_split_indices(dataset):
    """
    >>> TODO <<<  PASTE YOUR EXACT SPLIT LOGIC HERE (same seed as training).
    Must return (train_idx, val_idx) as lists of integers. The val set must be
    the same 2000 images your reports were computed on.
    Example (ONLY if this is what you used in training):
        from sklearn.model_selection import train_test_split
        labels = [l for _, l in dataset.samples]
        train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.1,
                                              stratify=labels, random_state=42)
    """
    raise NotImplementedError("Paste your training split code in get_split_indices()")


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------
def build_model(name):
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    else:
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    return m.to(device)


# ---------------------------------------------------------------
# One full training run for one protocol
# ---------------------------------------------------------------
def run_protocol(model_name, use_aug, use_wrs):
    set_seed(SEED)

    # >>> TODO <<< make this augmentation identical to your "Proposed Full" one
    if use_aug:
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(), NORM])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), NORM])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), NORM])

    train_ds_full = SolarDataset(train_tf)
    val_ds_full = SolarDataset(val_tf)
    train_idx, val_idx = get_split_indices(train_ds_full)
    train_ds, val_ds = Subset(train_ds_full, train_idx), Subset(val_ds_full, val_idx)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    if use_wrs:
        train_labels = np.array([train_ds_full.samples[i][1] for i in train_idx])
        class_count = np.bincount(train_labels, minlength=NUM_CLASSES)
        weights = (1.0 / class_count)[train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(model_name)
    criterion = nn.CrossEntropyLoss()          # plain CE, as stated in the paper
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1, best_state = -1, None
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        model.eval(); yt, yp = [], []
        with torch.no_grad():
            for x, y in val_loader:
                yp.extend(model(x.to(device)).argmax(1).cpu().numpy()); yt.extend(y.numpy())
        f1 = f1_score(yt, yp, average="macro")
        print(f"  epoch {epoch+1}/{EPOCHS}  val acc={accuracy_score(yt, yp):.4f}  macro-F1={f1:.4f}")
        if f1 > best_f1:                        # best checkpoint on macro-F1
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state); model.eval(); yt, yp = [], []
    with torch.no_grad():
        for x, y in val_loader:
            yp.extend(model(x.to(device)).argmax(1).cpu().numpy()); yt.extend(y.numpy())
    per_class = f1_score(yt, yp, average=None, labels=list(range(NUM_CLASSES)))
    return {
        "acc": accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, average="macro"),
        "hotspot_f1": per_class[CLASS_NAMES.index("Hot-Spot")],
        "soiling_f1": per_class[CLASS_NAMES.index("Soiling")],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet50", "mobilenet_v3"], required=True)
    ap.add_argument("--include_full", action="store_true",
                    help="also run Proposed Full (aug + WRS) as a sanity check")
    args = ap.parse_args()

    protocols = {"Baseline": (False, False),
                 "Augmentation only": (True, False),
                 "WRS only": (False, True)}
    if args.include_full:
        protocols["Proposed Full"] = (True, True)

    new_file = not os.path.exists("ablation_results.csv")
    with open("ablation_results.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["model", "protocol", "acc", "macro_f1", "hotspot_f1", "soiling_f1"])
        for name, (aug, wrs) in protocols.items():
            print(f"\n===== {args.model} | {name} =====")
            r = run_protocol(args.model, aug, wrs)
            w.writerow([args.model, name] + [f"{r[k]:.2f}" for k in
                       ("acc", "macro_f1", "hotspot_f1", "soiling_f1")])
            f.flush()
            print(f"RESULT {name}: {r}")