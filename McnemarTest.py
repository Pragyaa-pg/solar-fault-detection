"""
McNemar's Test (with continuity correction) between ResNet50 and MobileNetV3.
Prints the full 2x2 contingency table (including discordant pair counts b and c)
and the corrected test statistic + p-value.

Assumes: same val_loader used for both models' evaluation (same transforms,
same order of samples, shuffle=False).
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from statsmodels.stats.contingency_tables import mcnemar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 12

# ------------------------------------------------------------------
# IMPORTANT: val_loader must NOT shuffle, so both models see images
# in the exact same order — otherwise the paired comparison is invalid.
# If your loader has shuffle=True, change it to shuffle=False here.
# ------------------------------------------------------------------
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)   # <-- your val_loader

def get_predictions(model, loader):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
    return np.array(y_true), np.array(y_pred)

# ------------------------------------------------------------------
# Load ResNet50
# ------------------------------------------------------------------
resnet = models.resnet50(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet.load_state_dict(torch.load("resnet50_solar_fault_model.pth", map_location=device))
resnet.to(device)

# ------------------------------------------------------------------
# Load MobileNetV3
# ------------------------------------------------------------------
mobilenet = models.mobilenet_v3_large(weights=None)
mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, num_classes)
mobilenet.load_state_dict(torch.load("mobilenet_v3_solar_fault_model.pth", map_location=device))
mobilenet.to(device)

# ------------------------------------------------------------------
# Get predictions from both models (same val_loader, same order)
# ------------------------------------------------------------------
y_true_r, y_pred_resnet = get_predictions(resnet, val_loader)
y_true_m, y_pred_mobilenet = get_predictions(mobilenet, val_loader)

# Sanity check — both runs should have seen identical ground truth in the same order
assert np.array_equal(y_true_r, y_true_m), "val_loader order mismatch between the two runs!"
y_true = y_true_r

resnet_correct = (y_pred_resnet == y_true)
mobilenet_correct = (y_pred_mobilenet == y_true)

# ------------------------------------------------------------------
# Build the 2x2 contingency table
#            MobileNet correct | MobileNet wrong
# ResNet correct      a        |       b
# ResNet wrong         c       |       d
# ------------------------------------------------------------------
a = np.sum(resnet_correct & mobilenet_correct)
b = np.sum(resnet_correct & ~mobilenet_correct)   # ResNet right, MobileNet wrong
c = np.sum(~resnet_correct & mobilenet_correct)   # ResNet wrong, MobileNet right
d = np.sum(~resnet_correct & ~mobilenet_correct)

table = [[a, b], [c, d]]

print("Contingency table:")
print(f"                MobileNet Correct   MobileNet Wrong")
print(f"ResNet Correct  {a:>17}   {b:>15}")
print(f"ResNet Wrong    {c:>17}   {d:>15}")
print()
print(f"Discordant pairs — b (ResNet right, MobileNet wrong): {b}")
print(f"Discordant pairs — c (ResNet wrong, MobileNet right): {c}")
print()

# ------------------------------------------------------------------
# McNemar's test WITH continuity correction (standard for this sample size)
# ------------------------------------------------------------------
result = mcnemar(table, exact=False, correction=True)
print(f"McNemar's test statistic (with continuity correction): {result.statistic:.4f}")
print(f"p-value: {result.pvalue:.4f}")