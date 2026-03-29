# HealthCare Federated Learning Simulation

This project simulates a Federated Learning (FL) environment to train Machine Learning models on distributed healthcare data associated with patient monitoring. It allows for the comparison of different FL strategies and model architectures while ensuring data privacy (data remains on local clients).

## Overview

The simulation demonstrates how multiple hospitals (clients) can collaboratively train a global model without sharing their private patient datasets. It implements and compares three popular FL aggregation strategies:

*   **FedAvg (Federated Averaging):** The baseline standard for FL.
*   **FedProx:** robust to system heterogeneity.
*   **FedAdam:** An adaptive optimization strategy.

## Key Features

*   **Bring-Your-Own Dataset (BYOD):** Works out-of-the-box with any tabular `.csv` dataset. The preprocessor handles identifying and scaling numerical/categorical columns automatically without writing any code.
*   **Healthcare Evaluation Metrics:** Computes essential clinical numbers beyond accuracy (Sensitivity, Specificity, AUC-ROC, F1, RMSE).
*   **Multi-Model Testing:** Automatically tests linear models and fully connected networks side-by-side (`LogisticRegression`, `SimpleMLP`, `DeepMLP`).
*   **Clinical IID/Non-IID Distribution:** Allows testing across completely identical client hospitals or heavily skewed, distinct distributions with the `--distribution` flag.
*   **Automated Analytical Reporting:** Auto-plots convergence graphs and generates written summaries comparing algorithmic performance natively.

## Project Structure

*   `fl_simulation.py`: The primary command-line runner bridging the simulation servers & hospital clients.
*   `fl_data_preprocessing.py`: Automated pipeline for scaling, encoding, and partitioning any tabular dataset.
*   `generate_report.py`: Analytics suite parsing `.json` dumps to plot and summarize data points.
*   `fl_model.py`: Definitions of PyTorch dense neural network structures.
*   `fl_strategies.py`: Handles FedAvg, FedProx, and FedAdam server-side mathematical aggregation structures.

## Usage: 3 Easy Steps

Follow these steps to plug in your own dataset without editing any code!

### Step 1: Prepare Your Data
Make sure your data is in a clean `.csv` file in the main folder. You just need to know the name of the column you are trying to predict (e.g., `"Diagnosis"`, `"Blood Pressure"`, `"Readmitted"`).

### Step 2: Run the Simulation
Use the command line to tell the simulation what file to use and what your target is. It takes care of the rest!

**For Categorical targets (Classification):**
```bash
python fl_simulation.py \
    --data_path "YOUR_DATASET_NAME.csv" \
    --target_col "YOUR_TARGET_COLUMN" \
    --task_type classification \
    --num_clients 5 \
    --num_rounds 15
```

**For Continuous targets (Regression):**
```bash
python fl_simulation.py \
    --data_path "YOUR_DATASET_NAME.csv" \
    --target_col "YOUR_CONTINUOUS_TARGET" \
    --task_type regression \
    --num_clients 5 \
    --num_rounds 15
```

> **Advanced Simulation settings:** Add `--distribution non_iid` to simulate highly unbalanced/heterogeneous data among hospitals, or `--local_epochs N` to increase the amount of local training time per client.

### Step 3: View the Results
When your simulation completes, tell the automated reporting tool to parse the results. It will generate charts and a text review.

```bash
python generate_report.py
```

Look into the `results/` folder to see your plots, and read the `simulation_report_*.md` file generated in the main folder to see the "Ups and Downs" summary of the different algorithms tested against your specific dataset!
