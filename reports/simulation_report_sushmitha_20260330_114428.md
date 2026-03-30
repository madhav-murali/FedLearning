# Federated Learning Simulation Report for **Sushmitha**

**Date**: 2026-03-30 11:53:26
**Task Type**: Classification
**Target**: `Predicted Disease`
**Clients**: 5 (Distribution: non_iid)

## 1. Strategy Comparison Summary (The 'Ups and Downs')
Based on the captured metrics, here is an analysis of the federated learning algorithms:

- **FedAvg**: The standard baseline. Usually provides steady convergence but can be volatile if client data is highly skewed (non-IID).
- **FedProx**: Introduces a proximal term to constrain local updates. **Ups:** Offers superior stability and prevents the global model from diverging in highly heterogeneous (non-IID) health datasets. **Downs:** Can converge slightly slower than FedAvg initially.
- **FedAdam**: Uses adaptive server-side momentum. **Ups:** Can provide much faster initial convergence and reach a better optimum. **Downs:** Highly sensitive to the learning rate and hyperparameter tuning; can overshoot if not configured perfectly.

## 2. Final Results Table

| Algorithm | Final Acc | Final F1 | Final AUC | Sensitivity | Specificity | Final Loss |
|-----------|-----------|----------|-----------|-------------|-------------|------------|
| **FedAvg** | 47.05% | 0.3817 | 0.8203 | 0.4686 | 0.0000 | 1.3345 |
| **FedProx** | 48.85% | 0.4036 | 0.8010 | 0.4865 | 0.0000 | 1.3777 |
| **FedAdam** | 20.75% | 0.0816 | 0.7253 | 0.2048 | 0.0000 | 1.5909 |

## 3. Visualizations

![Strategy Comparison](C:\Users\sushmitha\Downloads\Projects\Projects\BTP\Fedlearning\reports\strategy_comparison_sushmitha_20260330_114428.png)

![Model Comparison](C:\Users\sushmitha\Downloads\Projects\Projects\BTP\Fedlearning\reports\model_comparison_sushmitha_20260330_114428.png)

