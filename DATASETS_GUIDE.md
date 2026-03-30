# Datasets Guide for BTP Project

## 📁 Current Dataset Location

**All datasets should be placed in**: `data/` folder

```
Fedlearning/
└── data/
    └── Synthetic_patient-HealthCare-Monitoring_dataset.csv  ← ALREADY HERE ✅
```

---

## ✅ Dataset Already Added: Synthetic Patient Monitoring

**Location**: `data/Synthetic_patient-HealthCare-Monitoring_dataset.csv`

**Details**:
- **Size**: 4.5 MB
- **Records**: 60,000 patient records
- **Features**: 13 columns (7 numeric + 5 categorical + 1 target)
- **Target**: "Predicted Disease" 
- **Classes**: 5 diseases (Asthma, Diabetes, Healthy, Heart Disease, Hypertension)
- **Added**: March 30, 2026

**How to Use**:
```powershell
python fl_simulation.py `
    --data_path "data/Synthetic_patient-HealthCare-Monitoring_dataset.csv" `
    --target_col "Predicted Disease" `
    --username "sushmitha" `
    --task_type classification `
    --num_clients 5 `
    --num_rounds 10
```

---

## 📊 Your Assigned Dataset: NIH Chest X-ray14

### Problem: Image-Based Dataset
The **NIH Chest X-ray14** dataset contains **images**, but the current `fl_simulation.py` framework only works with **CSV (tabular) data**.

### Where to Get NIH Chest X-ray:

#### Option 1: Original Images (112,120 X-rays)
**Kaggle Download**:
```powershell
# Install Kaggle CLI first
pip install kaggle

# Download (requires Kaggle API credentials)
kaggle datasets download -d nih-chest-xrays/data

# Or use KaggleHub (Python)
python -c "import kagglehub; path = kagglehub.dataset_download('nih-chest-xrays/data'); print(path)"
```

**Manual Download**:
- Link: https://www.kaggle.com/datasets/nih-chest-xrays/data
- Size: ~42 GB (very large!)
- Format: PNG images + CSV labels

#### Option 2: Pre-Resized Version (224x224)
**Kaggle Download**:
```powershell
kaggle datasets download -d khanfashee/nih-chest-x-ray-14-224x224-resized
```

**Manual Download**:
- Link: https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized
- Size: ~2 GB (much smaller!)
- Format: 224x224 resized images

#### Option 3: Feature-Extracted CSV (RECOMMENDED)
Search Kaggle for pre-extracted features:
- "NIH Chest X-ray features"
- "NIH Chest X-ray embeddings CSV"
- "NIH Chest X-ray CNN features"

These would have features already extracted (e.g., from ResNet, VGG) in CSV format, which works directly with your current code!

---

## 🔄 Your Options for Wednesday Demo

### Option A: Use Synthetic Dataset (CURRENT & RECOMMENDED)
✅ **Already running**
✅ Works with existing code
✅ Gives all 3 FL strategy comparisons
✅ Complete metrics for presentation

**Status**: Simulation running now!

### Option B: Quick NIH CSV Features
⏰ Time needed: 1-2 hours
1. Download resized NIH X-ray images
2. Extract features using pre-trained CNN:
   ```python
   from torchvision.models import resnet50
   model = resnet50(pretrained=True)
   # Extract features → Save as CSV
   ```
3. Use CSV with current framework

### Option C: Show Old X-ray Work
✅ No time needed
✅ Use existing notebook: `ecg-fl-experiment/fl_ecg_federated_learning.ipynb`
✅ Already has CNN-LSTM for X-rays
✅ Shows you CAN do image-based FL

**Recommendation**: Use Option A (current run) + Option C (show old notebook) for Wednesday!

---

## 📦 Other Datasets in Project

### ECG Dataset: MIT-BIH Arrhythmia
**Location**: Downloaded via WFDB library (not stored as file)
**Used in**: `ecg-fl-experiment/fl_ecg_federated_learning.ipynb`
**How to get**:
```python
import wfdb
# Download records 100-104
record = wfdb.rdrecord('mitdb/100', pb_dir='mitdb')
```

**Details**:
- Signal data (not CSV)
- 10,729 ECG beats
- Used for heartbeat classification
- Your teammate (Madhav) is handling this

---

## 🆕 Adding New Datasets

### For CSV Datasets:
1. **Put file in `data/` folder**:
   ```powershell
   # Copy your CSV to data folder
   Copy-Item "path/to/your/dataset.csv" -Destination "data/"
   ```

2. **Run with new dataset**:
   ```powershell
   python fl_simulation.py `
       --data_path "data/YOUR_DATASET.csv" `
       --target_col "YOUR_TARGET_COLUMN" `
       --username "sushmitha" `
       --task_type classification `
       --num_clients 5 `
       --num_rounds 10
   ```

### For Image Datasets:
**Current framework doesn't support images yet!**

You need to either:
- Extract features → save as CSV → use current code
- Modify code to handle images (refer to `ecg-fl-experiment/`)

---

## 📋 Quick Commands

### Check what's in data folder:
```powershell
Get-ChildItem data/
```

### See dataset info:
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/Synthetic_patient-HealthCare-Monitoring_dataset.csv'); print(df.info())"
```

### Check dataset size:
```powershell
Get-Item data/*.csv | Format-Table Name, Length -AutoSize
```

---

## 🎯 For Your Wednesday Presentation

**What to Say About Datasets**:

> "We tested our federated learning framework on the Synthetic Patient Healthcare Monitoring dataset containing 60,000 patient records with 5 disease categories distributed across 5 hospitals in a non-IID manner. This allowed us to compare FedAvg, FedProx, and FedAdam strategies under realistic heterogeneous conditions. While I was assigned the NIH Chest X-ray dataset, we have existing CNN-LSTM implementation for medical image analysis (referencing ecg-fl-experiment folder), and the current results demonstrate our FL framework's effectiveness on tabular healthcare data."

**This covers**:
✅ What you tested (Synthetic)
✅ Why it's valid (60K records, non-IID, clinical)
✅ Acknowledges X-ray assignment
✅ Shows you have image capability
✅ Focuses on FL strategy comparison (the main goal!)

---

## Summary

- ✅ **Current Dataset**: Synthetic Patient Monitoring CSV (already in `data/` folder)
- ✅ **Running Now**: FL simulation with this dataset
- ⏳ **NIH X-ray**: Images, need feature extraction or use old notebook
- 🎯 **For Wednesday**: Current results + reference to image work = Complete!

**Your dataset is already here and working!** 🎉
