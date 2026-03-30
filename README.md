# 🏥 HealthCare Federated Learning Simulation

**BTP Project - Edge AI for Healthcare Applications**

This project simulates a Federated Learning (FL) environment to train Machine Learning models on distributed healthcare data while ensuring data privacy. Multiple hospitals (clients) can collaboratively train a global model without sharing their private patient datasets.

---

## 🎯 Quick Start (5 Minutes)

### 1️⃣ Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2️⃣ Run Complete Simulation
```powershell
python fl_simulation.py `
    --data_path "data/Synthetic_patient-HealthCare-Monitoring_dataset.csv" `
    --target_col "Predicted Disease" `
    --username "sushmitha" `
    --task_type classification `
    --distribution non_iid `
    --num_clients 5 `
    --num_rounds 10
```

### 3️⃣ View Results
```powershell
# Generate clean dashboard
python demos\clean_dashboard.py

# Or run live demo
python demos\live_demo.py
```

**✅ Expected Result**: 90%+ accuracy with LogisticRegression + FedAvg

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Live Demos & Testing](#live-demos--testing)
5. [Running Simulations](#running-simulations)
6. [Datasets](#datasets)
7. [Results & Reports](#results--reports)
8. [Troubleshooting](#troubleshooting)

---

## 🔍 Overview

### Federated Learning Strategies Implemented

This simulation compares three popular FL aggregation strategies:

*   **FedAvg (Federated Averaging):** The baseline standard for FL - simple weighted averaging of client models
*   **FedProx:** Robust to system heterogeneity and non-IID data with proximal term regularization
*   **FedAdam:** Adaptive optimization strategy using momentum and adaptive learning rates for faster convergence

### Model Architectures

Tests multiple model types automatically:
- **LogisticRegression**: Simple linear classifier (best for tabular data)
- **SimpleMLP**: 2-layer neural network (128→64 neurons)
- **DeepMLP**: 4-layer deep network (256→128→64→32 neurons)

---

## ✨ Key Features

*   **Bring-Your-Own Dataset (BYOD):** Works with any tabular `.csv` dataset - no code changes needed
*   **Auto-Detection:** Preprocessor automatically identifies and scales numeric/categorical columns
*   **Healthcare Metrics:** Computes clinical metrics beyond accuracy (Sensitivity, Specificity, AUC-ROC, F1, Precision, Recall)
*   **IID/Non-IID Distribution:** Test with identical or heterogeneous client data distributions using `--distribution` flag
*   **Automated Reporting:** Auto-generates convergence graphs and written comparison summaries
*   **Live Visualization:** Real-time training progress with interactive dashboards

---

## 📁 Project Structure

```
Fedlearning/
├── data/                                           # Datasets folder
│   └── Synthetic_patient-HealthCare-Monitoring_dataset.csv
├── src/                                            # Core FL implementation
│   ├── fl_data_preprocessing.py                   # Auto data pipeline
│   ├── fl_model.py                                # PyTorch models
│   └── fl_strategies.py                           # FedAvg/FedProx/FedAdam
├── demos/                                          # Live demonstrations
│   ├── clean_dashboard.py                         # Professional visualization
│   ├── live_demo.py                               # Real-time demo (5-7 min)
│   └── quick_test.py                              # Fast demo (2 min)
├── scripts/                                        # Utility scripts
│   ├── generate_report.py                         # Report generator
│   └── download_nih_dataset.py                    # Dataset downloader
├── reports/                                        # Generated reports & visualizations
├── results/                                        # Simulation results (JSON)
├── fl_simulation.py                               # Main simulation runner
└── requirements.txt                               # Dependencies
```

---

## 🎬 Live Demos & Testing

Perfect for presentations and demonstrations!

### Option 1: Quick Test Demo (⏱️ 2 minutes)
**Best for**: Short presentations, quick verification

```powershell
python demos\quick_test.py
```

**What it shows:**
- ✅ Loads data from 3 hospitals
- ✅ Runs 3 rounds of federated learning
- ✅ Shows accuracy improving in real-time
- ✅ Displays final results (accuracy ~85%+)
- ✅ Creates visual summary chart

**Perfect when**: You have 2-3 minutes to quickly prove it works

---

### Option 2: Live Training Demo (⏱️ 5-7 minutes)
**Best for**: Full presentation with impressive visuals

```powershell
python demos\live_demo.py
```

**What it shows:**
- ✅ Real-time training progress for each hospital
- ✅ Live updating graphs showing accuracy improving
- ✅ Individual hospital performance vs global model
- ✅ Complete convergence visualization
- ✅ Detailed metrics at the end

**Perfect when**: You want to wow the audience with live plots

---

### Option 3: Professional Dashboard (⏱️ Instant)
**Best for**: Showing completed experimental results

```powershell
python demos\clean_dashboard.py
```

**What it shows:**
- ✅ 5-chart comprehensive dashboard
- ✅ Accuracy convergence curves
- ✅ Final performance comparison
- ✅ Loss reduction over rounds
- ✅ Best model metrics (all 5 metrics)
- ✅ Side-by-side strategy comparison

**Output**: Saved as PNG in `reports/` folder

---

### 🎓 Demo Presentation Script

**Setup** (before audience):
```powershell
cd C:\Users\sushmitha\Downloads\Projects\Projects\BTP\Fedlearning
```

**Introduction** (say to audience):
> "Let me demonstrate our federated learning system working in real-time. This shows how multiple hospitals can collaborate to train AI models without sharing any patient data."

**Run live demo**:
```powershell
python demos\live_demo.py
```

**During training** (explain):
1. **When loading data**: "The system is loading patient data from 3 different hospitals. Each has different patient demographics - simulating real-world heterogeneity."

2. **When training starts**: "Now watch as each hospital trains its local model, then the server aggregates the learning. No patient data leaves the hospitals!"

3. **When showing results**: "We achieved 90.78% accuracy. The model learned from all hospitals while keeping their data private!"

**Show final results**:
```powershell
python demos\clean_dashboard.py
```

> "This dashboard shows our complete experiment - we tested 3 different FL strategies. LogisticRegression with FedAvg achieved 90.78% accuracy, which is excellent for healthcare applications!"

**Key Messages**:
- 🔒 **Privacy**: "No hospital shares patient data - only model weights"
- 🤝 **Collaboration**: "All hospitals benefit from each other without seeing the data"
- 📊 **Performance**: "90.78% accuracy across 5 hospitals and 60,000 patients"
- ⚕️ **Real-world**: "Addresses HIPAA and privacy regulations"

---

## ⚙️ Running Simulations

### Basic Command (Windows PowerShell)

```powershell
python fl_simulation.py `
    --data_path "data/YOUR_DATASET.csv" `
    --target_col "YOUR_TARGET_COLUMN" `
    --username "your_name" `
    --task_type classification `
    --distribution non_iid `
    --num_clients 5 `
    --num_rounds 10
```

### All Available Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--data_path` | Path to CSV dataset | Required | Any CSV file path |
| `--target_col` | Column name to predict | Required | Column name in CSV |
| `--username` | Your name for tagging results | Required | Any string |
| `--task_type` | Classification or regression | `classification` | `classification`, `regression` |
| `--distribution` | Data distribution type | `iid` | `iid`, `non_iid` |
| `--num_clients` | Number of hospital clients | `3` | 2-10 |
| `--num_rounds` | FL communication rounds | `10` | 5-50 |
| `--local_epochs` | Training epochs per round | `5` | 1-20 |
| `--batch_size` | Training batch size | `32` | 16-128 |

### Example: Full Experiment

```powershell
python fl_simulation.py `
    --data_path "data/Synthetic_patient-HealthCare-Monitoring_dataset.csv" `
    --target_col "Predicted Disease" `
    --username "sushmitha" `
    --task_type classification `
    --distribution non_iid `
    --num_clients 5 `
    --num_rounds 10 `
    --local_epochs 5 `
    --batch_size 32
```

**Output Files**:
- `results/simulation_results_sushmitha_YYYYMMDD_HHMMSS.json` - Raw metrics
- Auto-runs dashboard generation

**Time**: ~10-15 minutes for complete run

---

## 📊 Datasets

### ✅ Currently Available: Synthetic Patient Monitoring

**Location**: `data/Synthetic_patient-HealthCare-Monitoring_dataset.csv`

**Details**:
- **Size**: 4.5 MB
- **Records**: 60,000 patient monitoring records
- **Features**: 13 columns (7 numeric + 5 categorical + 1 target)
- **Target**: "Predicted Disease" 
- **Classes**: 5 diseases (Asthma, Diabetes, Healthy, Heart Disease, Hypertension)
- **Performance**: LogisticRegression achieves 90.78% accuracy

---

### 🩺 NIH Chest X-ray Dataset (Downloaded)

**Location**: `C:\Users\sushmitha\.cache\kagglehub\datasets\khanfashee\nih-chest-x-ray-14-224x224-resized\3\`

**Details**:
- **Images**: 112,120 chest X-rays
- **Size**: 2.3 GB (224×224 resized)
- **Classes**: 14 thoracic diseases + "No Finding"
- **Format**: PNG images + CSV labels (`Data_Entry_2017.csv`)

**⚠️ Note**: Current `fl_simulation.py` framework only supports CSV (tabular) data. For image-based FL:
- Check `ecg-fl-experiment/fl_ecg_federated_learning.ipynb` for CNN-LSTM implementation
- Or extract CNN features and save as CSV

---

## 📊 Results & Reports

### Understanding Your Results

After simulation completes, you get:

**1. JSON Results File**
```
results/simulation_results_<username>_<timestamp>.json
```
Contains:
- Raw metrics for all strategies & models
- Accuracy, F1, Precision, Recall, AUC per round
- Loss curves
- Training times
- Client distribution info

**2. Markdown Report**
```
reports/simulation_report_<username>_<timestamp>.md
```
Contains:
- Formatted comparison tables
- Strategy rankings
- Detailed metrics breakdown
- Recommendations

**3. Visual Dashboard**
```
reports/clean_dashboard_simulation_results_<username>_<timestamp>.png
```
5-chart comprehensive visualization showing all comparisons

---

### Performance Metrics Explained

**Classification Metrics:**
- **Accuracy**: Overall correct predictions (target: >85% for healthcare)
- **F1-Score**: Balance of precision and recall (0.85-1.0 = excellent)
- **Precision**: Of predicted positives, how many are truly positive
- **Recall (Sensitivity)**: Of actual positives, how many were detected
- **Specificity**: True negative rate (important for ruling out diseases)
- **AUC-ROC**: Area under curve (>0.9 = excellent discrimination)

**Good Results Indicators:**
- ✅ **85-95% accuracy** = Excellent for healthcare
- ✅ **F1 > 0.85** = Well-balanced model
- ✅ **Convergence** = Loss decreasing, accuracy increasing
- ✅ **Stability** = Low variance across rounds

**Your Current Results:**
- **LogisticRegression + FedAvg**: 90.78% accuracy ✅
- **SimpleMLP + FedAvg**: 82.08% accuracy ✅
- **DeepMLP + FedProx**: 48.85% accuracy ⚠️ (overfitting on small data)

---

## 🔧 Troubleshooting

### Common Issues & Solutions

**Issue: Low accuracy (<70%)**
- **Cause**: Too few local epochs, poor model choice, or severe non-IID data
- **Solution**: Increase `--local_epochs` to 5-10, try LogisticRegression for tabular data

**Issue: Model not converging**
- **Cause**: Learning rate too high, too few rounds
- **Solution**: Increase `--num_rounds` to 20+, or try FedProx for better stability

**Issue: "CSV file not found"**
- **Cause**: Wrong path or file not in `data/` folder
- **Solution**: Use absolute path or put file in `data/` folder

**Issue: Memory error during training**
- **Cause**: Batch size too large, too many clients
- **Solution**: Reduce `--batch_size` to 16, or reduce `--num_clients`

**Issue: Demo freezes during presentation**
- **Cause**: Computation taking longer than expected
- **Solution**: Press Ctrl+C, then run `python demos\clean_dashboard.py` to show pre-computed results

---

### Interpreting Results

**When FedAvg Works Best:**
- IID data distribution
- Simple tabular datasets
- Fast convergence needed

**When FedProx Works Best:**
- Non-IID (heterogeneous) data
- Different client data sizes
- Need robustness to stragglers

**When FedAdam Works Best:**
- Need fastest convergence
- Complex optimization landscape
- Can tune adaptive parameters

**Your Dataset (Synthetic Patient Monitoring):**
- Best: **LogisticRegression + FedAvg** (90.78%)
- Why: Tabular data, linear relationships, balanced classes
- Lesson: Simpler models often outperform complex ones on structured data

---

## 🎯 For Wednesday Presentation

### Pre-Presentation Checklist

- [ ] Simulation completed successfully
- [ ] Dashboard generated (`demos\clean_dashboard.py`)
- [ ] Tested live demo (`demos\quick_test.py` or `demos\live_demo.py`)
- [ ] Know your key numbers (90.78% accuracy!)
- [ ] Terminal ready in correct directory
- [ ] Laptop charged 🔋

---

### 7-Minute Presentation Flow

**[Minutes 0-1] Introduction**
1. Open PowerShell in project directory
2. Say: "Today I'll demonstrate federated learning for healthcare"
3. Explain: "3 hospitals will collaborate without sharing patient data"
4. Start: `python demos\live_demo.py`

**[Minutes 1-6] Live Demo**
1. Point out data loading from 3 hospitals
2. Explain each round as accuracy improves
3. Emphasize privacy (no data sharing)
4. Show live graphs updating

**[Minutes 6-7] Results Summary**
1. Run: `python demos\clean_dashboard.py`
2. Show comprehensive dashboard
3. Highlight 90.78% accuracy achievement
4. Compare 3 FL strategies
5. Discuss real-world HIPAA compliance

---

### Key Talking Points

🔒 **Privacy**: "No hospital shares patient data - only model weights are exchanged"

🤝 **Collaboration**: "All 5 hospitals benefit from collective learning without seeing each other's data"

📊 **Performance**: "We achieved 90.78% accuracy across 60,000 patient records"

⚕️ **Real-World Impact**: "This addresses HIPAA regulations and enables multi-institutional healthcare AI"

🎯 **Results**: 
- Tested 3 FL strategies (FedAvg, FedProx, FedAdam)
- Compared 3 model architectures
- Non-IID data distribution (realistic scenario)
- Complete clinical metrics (F1, Precision, Recall, AUC)

---

### Emergency Backups

If live demo fails:
```powershell
# Show pre-computed dashboard
python demos\clean_dashboard.py

# Or open saved results
notepad reports\simulation_report_sushmitha_20260330_114428.md
```

---

## 👥 Team & Responsibilities

**BTP Team Members:**
- **Harshitha** - Core FL algorithms & FedProx implementation
- **Madhav** - ECG/Signal processing datasets
- **Manoj** - System integration & testing
- **Sushmitha** - Synthetic patient monitoring & NIH Chest X-ray datasets

**Supervisor**: [Your supervisor name]

**Project Timeline**: January - April 2026

**Presentation Date**: Wednesday, April 2, 2026

---

## 📚 Additional Resources

### Relevant Papers

1. **FedAvg**: McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
2. **FedProx**: Li et al. "Federated Optimization in Heterogeneous Networks" (2020)
3. **FedAdam**: Reddi et al. "Adaptive Federated Optimization" (2021)

### Useful Links

- **Flower Framework**: https://flower.dev/ (Production FL framework)
- **NVIDIA FLARE**: https://nvidia.github.io/NVFlare/ (Enterprise FL platform)
- **TensorFlow Federated**: https://www.tensorflow.org/federated

---

## 🔐 Privacy & Security Notes

**What This Project Demonstrates:**

✅ **Data Privacy**: Raw patient data never leaves client hospitals

✅ **Compliance Ready**: Framework supports HIPAA-compliant workflows

✅ **Differential Privacy**: Can be added via gradient perturbation (future work)

✅ **Secure Aggregation**: Only aggregated model updates are shared

**Not Implemented (Production Requirements):**
- Encrypted communication channels
- Secure multi-party computation
- Homomorphic encryption
- Client authentication

---

## 📝 License

[Add your license here]

---

## 🆘 Support

For questions or issues:
1. Check this README
2. Review generated reports in `reports/`
3. Contact team members
4. Check `ecg-fl-experiment/` for image-based examples

---

**Last Updated**: March 30, 2026

**Project Status**: ✅ Fully Functional | 📊 Results: 90.78% Accuracy | 🎓 Ready for Presentation
