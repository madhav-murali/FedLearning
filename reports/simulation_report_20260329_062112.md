# Federated Learning Simulation Report

**Date**: 2026-03-29 06:42:12
**Task Type**: Classification
**Target**: `Predicted Disease`
**Clients**: 3 (Distribution: iid)

## 1. Strategy Comparison Summary (The 'Ups and Downs')
Based on the captured metrics, here is an analysis of the federated learning algorithms:

- **FedAvg**: The standard baseline. Usually provides steady convergence but can be volatile if client data is highly skewed (non-IID).
- **FedProx**: Introduces a proximal term to constrain local updates. **Ups:** Offers superior stability and prevents the global model from diverging in highly heterogeneous (non-IID) health datasets. **Downs:** Can converge slightly slower than FedAvg initially.
- **FedAdam**: Uses adaptive server-side momentum. **Ups:** Can provide much faster initial convergence and reach a better optimum. **Downs:** Highly sensitive to the learning rate and hyperparameter tuning; can overshoot if not configured perfectly.

## 2. Final Results Table

| Algorithm | Final Acc | Final F1 | Final AUC | Sensitivity | Specificity | Final Loss |
|-----------|-----------|----------|-----------|-------------|-------------|------------|
| **FedAvg** | 93.55% | 0.9354 | 0.9951 | 0.9356 | 0.0000 | 0.1679 |
| **FedProx** | 93.51% | 0.9350 | 0.9950 | 0.9352 | 0.0000 | 0.1707 |
| **FedAdam** | 78.30% | 0.7683 | 0.9765 | 0.7844 | 0.0000 | 1.3948 |

## 3. Visualizations

![Strategy Comparison](/home/madhav/home/btp/results/strategy_comparison_20260329_062112.png)

![Model Comparison](/home/madhav/home/btp/results/model_comparison_20260329_062112.png)

