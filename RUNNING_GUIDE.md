# Running Guide for BTP Project

## Team Member: Sushmitha
**Dataset Assigned**: NIH Chest X-ray  
**Deadline**: Wednesday (April 2, 2026)
**Requirements**: Compare 3 FL techniques with complete metrics

---

## ⚠️ IMPORTANT NOTE

The current `fl_simulation.py` framework **ONLY works with CSV files**. 

**Your Options for NIH Chest X-ray Dataset:**

### Option 1: Use Pre-extracted Features (RECOMMENDED FOR QUICK RESULTS)
Find or create a CSV with extracted features from NIH Chest X-ray images:
- Features could be: ResNet/VGG embeddings, radiomics features, etc.
- Format: `patient_id, feature1, feature2, ..., featureN, disease_label`

### Option 2: Use the ECG-FL Code as Reference
Check `ecg-fl-experiment/fl_ecg_federated_learning.ipynb` - this has:
- Image/signal data handling
- CNN-LSTM architecture for X-rays
- Already tested on NIH Chest X-ray

### Option 3: Use Synthetic Dataset for Now (FASTEST)
Run with the existing CSV dataset to get all comparison results ready, then adapt for images later.

---

## Quick Run Commands

### For Windows PowerShell:

**Test Run (5 clients, 10 rounds):**
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

**Generate Report:**
```powershell
python generate_report.py
```

---

## What You Need for Wednesday

✅ **1. Implementation**: Run FL simulation with all 3 strategies
- FedAvg (baseline)
- FedProx (handles non-IID better)  
- FedAdam (faster convergence)

✅ **2. Results to Report**:
- **Accuracy** (overall classification accuracy)
- **F1-Score** (macro and per-class)
- **Precision & Recall**
- **Specificity & Sensitivity** (for medical context)
- **AUC-ROC** (if binary/multi-class)
- **Loss curves** (convergence plots)
- **RMSE/MAE** (if regression)
- **Training time per round**

✅ **3. Comparison Analysis**:
- Table comparing all 3 strategies
- Plots showing convergence
- Discussion of tradeoffs:
  - Accuracy vs Communication rounds
  - Stability under non-IID data
  - Training speed

✅ **4. Documentation**:
- Run logs saved in `results/`
- Auto-generated report in `reports/`
- Clear explanation of which dataset you used and why

---

## Expected Output Structure

After running, you'll have:
```
results/
  └── simulation_results_sushmitha_YYYYMMDD_HHMMSS.json

reports/
  └── simulation_report_sushmitha_YYYYMMDD_HHMMSS.md
      ├── Strategy comparison tables
      ├── Convergence plots
      └── Performance metrics
```

---

## Troubleshooting

**Problem**: "Dataset not CSV"
- **Solution**: Extract features to CSV or use sample dataset

**Problem**: "Module not found"
- **Solution**: `pip install -r requirements.txt`

**Problem**: "Low accuracy"
- **Solution**: Increase `--num_rounds` or adjust learning rate in code

---

## Parameter Meanings

- `--num_clients`: Number of hospitals/clients (3-5 typical)
- `--num_rounds`: Communication rounds (10-20 for convergence)
- `--distribution`: `iid` (balanced) or `non_iid` (realistic, imbalanced)
- `--task_type`: `classification` or `regression`
- `--local_epochs`: Training epochs per client per round (default: 5)
- `--batch_size`: Batch size for training (default: 32)

---

## Tips for Wednesday Presentation

1. **Show the comparison table** with all metrics
2. **Explain why FedProx works better** for non-IID (adds proximal term)
3. **Explain why FedAdam converges faster** (adaptive momentum)
4. **Highlight privacy preservation**: Only model weights shared, not data
5. **Discuss clinical relevance**: F1, Sensitivity, Specificity matter more than just accuracy

---

## Next Steps After Wednesday

- [ ] Integrate NIH Chest X-ray image data properly
- [ ] Add FHIR integration
- [ ] Implement Digital Twin interface
- [ ] Test with FedYogi optimizer
- [ ] Deploy on edge devices
