# Federated Learning Simulation Report
**Date**: 2026-02-05 09:13:55
**Source Data**: `results/simulation_results_20260205_091345.json`

## 1. Introduction
This report compares different model architectures (**Logistic Regression**, **Simple MLP**, **Deep MLP**) and FL algorithms (**FedAvg**, **FedProx**, **FedAdam**).

## 2. Models & Strategies
- **Models Evaluated**: Logistic Regression, Simple MLP, Deep MLP.
- **Strategies Evaluated**: FedAvg, FedProx, FedAdam.

## 3. Results Comparison

### Model Performance (Fixed Strategy: FedAvg)
| Model | Final Accuracy | Final Loss |
|-------|----------------|------------|
| LogisticRegression | 92.53% | 0.2132 |
| SimpleMLP | 94.09% | 0.1498 |
| DeepMLP | 94.05% | 0.1494 |

### Algorithm Performance (Fixed Model: DeepMLP)
| Algorithm | Final Accuracy | Final Loss |
|-----------|----------------|------------|
| FedAvg | 94.05% | 0.1494 |
| FedProx | 94.07% | 0.1521 |
| FedAdam | 84.73% | 0.7598 |


## 4. Visualizations

### Model Comparison
![Model Comparison](results/model_accuracy_plot_20260205_091345.png)

### Strategy Comparison
![Strategy Comparison](results/strategy_accuracy_plot_20260205_091345.png)

