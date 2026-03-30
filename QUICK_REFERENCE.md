# Quick Reference Card - Sushmitha's BTP

## Current Status
- ✅ Requirements.txt updated
- ✅ README updated with running instructions
- ✅ Simulation RUNNING with parameters:
  - Dataset: Synthetic Patient Monitoring
  - Clients: 5 (non-IID distribution)
  - Rounds: 10
  - Username: sushmitha

---

## What's Happening Now

The simulation is testing:
1. **Phase 1 - Model Comparison** (using FedAvg):
   - LogisticRegression
   - SimpleMLP
   - DeepMLP

2. **Phase 2 - Strategy Comparison** (using DeepMLP):
   - FedAvg
   - FedProx
   - FedAdam

---

## Your NIH Chest X-ray Dataset Issue

⚠️ **PROBLEM**: Current code only works with CSV, but NIH Chest X-ray is images

**SOLUTIONS**:

### A. Quick Fix (for Wednesday demo):
Use the Synthetic dataset (currently running) to show all 3 FL strategies comparison. This gives you all required metrics!

### B. Proper Fix (for final submission):
1. Check `ecg-fl-experiment/fl_ecg_federated_learning.ipynb`
2. It already has CNN-LSTM code for X-ray images
3. Adapt that code to match the new fl_simulation.py framework

### C. Alternative:
Find a CSV with pre-extracted X-ray features:
- Search Kaggle for "NIH Chest X-ray features CSV"
- Or extract features using ResNet/VGG and save as CSV

---

## Commands You Need

### Run Simulation:
```powershell
python fl_simulation.py --data_path "data/Synthetic_patient-HealthCare-Monitoring_dataset.csv" --target_col "Predicted Disease" --username "sushmitha" --task_type classification --distribution non_iid --num_clients 5 --num_rounds 10
```

### Generate Report:
```powershell
python generate_report.py
```

### Check Results:
```powershell
Get-ChildItem results/simulation_results_sushmitha_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

### View Latest Report:
```powershell
Get-ChildItem reports/simulation_report_sushmitha_*.md | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

---

## What to Tell Your Team

**For Madhav**: 
"I'm running the FL framework with the Synthetic dataset since the current code only supports CSV. All 3 strategies (FedAvg, FedProx, FedAdam) are being tested with the same client setup (5 clients, non-IID). Results will include all metrics: accuracy, F1, precision, recall, AUC, loss curves."

**For Wednesday Demo**:
- You'll have complete comparison results
- All requested metrics (F1, accuracy, sensitivity, specificity, etc.)
- Convergence plots
- Auto-generated report
- Can discuss adapting for NIH X-ray images as "future work" or show the old ECG-FL notebook

---

## Files Being Generated

1. `results/simulation_results_sushmitha_YYYYMMDD_HHMMSS.json`
   - Raw metrics for all strategies
   - Loss/accuracy curves
   - Training times

2. `reports/simulation_report_sushmitha_YYYYMMDD_HHMMSS.md`
   - Comparison tables
   - Plots
   - Analysis text

---

## Expected Completion Time

- Model training: ~5-15 minutes (depending on your CPU/GPU)
- Report generation: ~30 seconds
- Total: ~15-20 minutes for complete results

---

## Metrics You'll Get

✅ Accuracy
✅ F1-Score (macro & micro)
✅ Precision
✅ Recall (Sensitivity)
✅ Specificity
✅ AUC-ROC
✅ Loss curves
✅ Training time per round
✅ Convergence plots

All required for Wednesday! 🎉
