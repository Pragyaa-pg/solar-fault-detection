"""
Per-class F1-score Comparison chart: MobileNetV3 vs ResNet-50

TWO WAYS TO USE THIS:

OPTION A — you already have the per-class F1 scores
    (e.g. printed by sklearn's classification_report on your own trained models)
    -> just fill in f1_mobilenet and f1_resnet below and run.

OPTION B — you have raw predictions saved (y_true, y_pred for each model)
    -> use the compute_f1_per_class() helper to get the F1 lists automatically,
       instead of typing them by hand.
"""

import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "Cell", "Cell-Multi", "Cracking", "Diode", "Diode-Multi",
    "Hot-Spot", "Hot-Spot-Multi", "No-Anomaly", "Offline-Module",
    "Shadowing", "Soiling", "Vegetation"
]


# ------------------------------------------------------------------
# OPTION A: fill these in with YOUR OWN model's per-class F1 scores,
# in the same order as CLASS_NAMES above.
# You get these numbers from sklearn's classification_report(y_true, y_pred)
# for each model — the "f1-score" column, one value per class.
# ------------------------------------------------------------------
f1_mobilenet = [0.66, 0.49, 0.67, 0.96, 0.88, 0.59, 0.60, 0.92, 0.63, 0.70, 0.47, 0.69]  # <-- replace with your numbers
f1_resnet    = [0.70, 0.51, 0.73, 0.98, 0.88, 0.65, 0.57, 0.93, 0.70, 0.74, 0.58, 0.73]  # <-- replace with your numbers


# ------------------------------------------------------------------
# OPTION B: if you have raw y_true/y_pred arrays saved (e.g. from a
# classification_report or your own eval loop), use this instead to
# compute the per-class F1 lists automatically. Uncomment and adapt.
# ------------------------------------------------------------------
# from sklearn.metrics import f1_score
#
# def compute_f1_per_class(y_true, y_pred, num_classes=12):
#     return f1_score(y_true, y_pred, average=None, labels=range(num_classes))
#
# f1_mobilenet = compute_f1_per_class(y_true_mobilenet, y_pred_mobilenet)
# f1_resnet    = compute_f1_per_class(y_true_resnet, y_pred_resnet)


def plot_f1_comparison(class_names, f1_a, f1_b,
                        label_a="MobileNetV3", label_b="ResNet-50",
                        filename="f1_comparison.png"):
    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, f1_a, width, label=label_a)
    ax.bar(x + width / 2, f1_b, width, label=label_b)

    ax.set_ylabel("F1-Score")
    ax.set_title(f"Per-class F1-score Comparison: {label_a} vs {label_b}")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()
    print(f"Saved {filename}")


if __name__ == "__main__":
    plot_f1_comparison(CLASS_NAMES, f1_mobilenet, f1_resnet)