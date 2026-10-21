# Solar Fault Detection

Deep learning based fault detection in solar panels using infrared (thermal) images. The project trains and compares lightweight and standard CNN models (MobileNetV3, ResNet50) on the InfraredSolarModules dataset, and evaluates them on accuracy, F1-score, statistical significance and deployment efficiency.

## Highlights

- Multi-class classification of solar module anomalies from infrared images
- Model comparison: **MobileNetV3** vs **ResNet50**
- Ablation study to measure the effect of different training choices
- Statistical validation using **McNemar's test**
- Efficiency analysis (model size, inference speed) for real-world deployment

## Dataset

This project uses the **InfraredSolarModules** dataset.

- Download from: **[ADD DATASET LINK HERE]**
- Or run `python download_dataset.py`
- Extract it into the project root so you get an `InfraredSolarModules/` folder

> The dataset and trained model weights are not included in this repository because of their size.

## Results

### Class distribution
![Class Distribution](1_class_distribution.png)

### Brightness by class
![Brightness by Class](2_brightness_by_class.png)

### Visual comparison
![Visual Comparison](4_visual_comparison.png)

### F1-score comparison
![F1 Comparison](f1_comparison.png)

| Model | Accuracy | F1 (macro) | Params | Inference time |
|-------|----------|------------|--------|----------------|
| MobileNetV3 | [ADD] | [ADD] | [ADD] | [ADD] |
| ResNet50 | [ADD] | [ADD] | [ADD] | [ADD] |

Detailed numbers are in `ablation_results.csv`, `efficiency_summary.csv` and `deployment_metrics.csv`.

## Project Structure

```
solar-fault-detection/
├── download_dataset.py          # Download the dataset
├── check_dataset.py             # Sanity check on dataset files
├── analyze_dataset.py           # Class distribution, brightness analysis plots
├── create_splits.py             # Creates train.csv / val.csv / test.csv
├── train_model.py               # Main training script
├── retrain_resnet50.py          # ResNet50 retraining
├── run_ablation.py              # Ablation experiments
├── ablationRun.py               # Ablation runner
├── verify_saved_models.py       # Verify saved model weights
├── confusion_matrix_mobilenetv3.py  # Confusion matrix for MobileNetV3
├── plotF1Comparision.py         # F1 comparison plot
├── statistical_test.py          # Statistical significance test
├── McnemarTest.py               # McNemar's test
├── model_efficiency.py          # Model size and speed analysis
├── train.csv / val.csv / test.csv   # Dataset splits
└── *.csv, *.png                 # Results and plots
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Pragyaa-pg/solar-fault-detection.git
cd solar-fault-detection

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac

# 3. Install dependencies
pip install -r requirements.txt
```

## How to Run

Run the scripts in this order:

```bash
python download_dataset.py        # 1. get the dataset
python analyze_dataset.py         # 2. explore the data
python create_splits.py           # 3. create train/val/test splits
python train_model.py             # 4. train the model
python retrain_resnet50.py        # 5. train the ResNet50 baseline
python run_ablation.py            # 6. ablation study
python confusion_matrix_mobilenetv3.py   # 7. evaluation plots
python statistical_test.py        # 8. statistical tests
python McnemarTest.py
python model_efficiency.py        # 9. efficiency analysis
python plotF1Comparision.py       # 10. final comparison plot
```

A GPU is recommended. The scripts fall back to CPU automatically if CUDA is not available.

## Tech Stack

Python, PyTorch, torchvision, scikit-learn, pandas, NumPy, Pillow, Matplotlib

## Author

**Pragya Gupta**
- GitHub: [@Pragyaa-pg](https://github.com/Pragyaa-pg)
- LinkedIn: [pragya-4568aa376](https://linkedin.com/in/pragya-4568aa376)

## Acknowledgements

- InfraredSolarModules dataset: **[ADD CREDIT / CITATION HERE]**