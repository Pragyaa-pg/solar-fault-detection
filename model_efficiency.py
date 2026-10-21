import os
import time
import torch
import torch.nn as nn
from torchvision import models
from thop import profile   # pip install thop

# ============================================================
# Configuration
# ============================================================
IMG_SIZE = 224
NUM_CLASSES = 12
DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_CPU = torch.device("cpu")

# ============================================================
# Create the two models (same as in your training)
# ============================================================
def create_mobilenet():
    model = models.mobilenet_v3_large(weights=None)   # or DEFAULT if you want pretrained
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    return model

def create_resnet50():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

# ============================================================
# Measurement functions
# ============================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6   # in millions

def model_size_mb(model, filename="temp_model.pth"):
    torch.save(model.state_dict(), filename)
    size = os.path.getsize(filename) / (1024 * 1024)  # MB
    os.remove(filename)
    return size

def count_flops(model, device):
    model = model.to(device)
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return flops / 1e9   # in GFLOPs

def measure_latency(model, device, warmup=20, runs=100):
    model = model.to(device)
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / runs * 1000
    return avg_ms

# ============================================================
# Run measurements
# ============================================================
models_to_test = {
    "MobileNetV3": create_mobilenet(),
    "ResNet-50": create_resnet50()
}

# Your known Macro-F1 values (from training)
macro_f1 = {
    "MobileNetV3": 0.64,
    "ResNet-50": 0.62
}

print(f"{'Model':<15} {'Params (M)':<12} {'Size (MB)':<10} {'FLOPs (G)':<10} "
      f"{'CPU (ms)':<10} {'GPU (ms)':<10} {'Macro-F1':<10}")
print("-" * 85)

results = []

for name, model in models_to_test.items():
    params = count_parameters(model)
    size = model_size_mb(model)
    flops = count_flops(model, DEVICE_GPU if torch.cuda.is_available() else DEVICE_CPU)
    
    latency_cpu = measure_latency(model, DEVICE_CPU)
    
    if torch.cuda.is_available():
        latency_gpu = measure_latency(model, DEVICE_GPU)
    else:
        latency_gpu = None

    print(f"{name:<15} {params:<12.1f} {size:<10.1f} {flops:<10.2f} "
          f"{latency_cpu:<10.1f} {latency_gpu if latency_gpu else 'N/A':<10} {macro_f1[name]:<10.2f}")

    results.append({
        "Model": name,
        "Params (M)": round(params, 1),
        "Size (MB)": round(size, 1),
        "FLOPs (G)": round(flops, 2),
        "Latency CPU (ms)": round(latency_cpu, 1),
        "Latency GPU (ms)": round(latency_gpu, 1) if latency_gpu else "N/A",
        "Macro-F1": macro_f1[name]
    })

# ============================================================
# Optional: save as CSV / Markdown table
# ============================================================
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("deployment_metrics.csv", index=False)
print("\nSaved: deployment_metrics.csv")

# Print Markdown table (copy-paste ready for paper)
print("\nMarkdown table:")
print(df.to_markdown(index=False))