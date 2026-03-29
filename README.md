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

*   `data/`: **Put your datasets here!** Isolated directory for raw input `.csv` files.
*   `src/`: Core simulation logic.
    *   `fl_data_preprocessing.py`: Automated pipeline for scaling, encoding, and partitioning any tabular dataset.
    *   `fl_model.py`: Definitions of PyTorch dense neural network structures.
    *   `fl_strategies.py`: Handles FedAvg, FedProx, and FedAdam server-side mathematical aggregation structures.
*   `fl_simulation.py`: The primary command-line runner bridging the simulation servers & hospital clients.
*   `generate_report.py`: Analytics suite parsing `.json` dumps to plot and summarize data points.

## Usage: 3 Easy Steps

Follow these steps to plug in your own dataset without editing any code!

### Step 1: Prepare Your Data
Drop your clean `.csv` file directly into the `data/` folder. You just need to know the name of the column you are trying to predict (e.g., `"Diagnosis"`, `"Blood Pressure"`, `"Readmitted"`).

### Step 2: Run the Simulation!
We've provided an easy one-click script that does the heavy lifting, running the simulation and auto-generating reports tagged with your name!

Use `./run_experiment.sh` and tell it your name, your file, and what you are trying to predict.

**Example for Disease Prediction (Classification):**
```bash
./run_experiment.sh \
    --username your_name \
    --dataset "YOUR_DATASET_NAME.csv" \
    --target "YOUR_TARGET_COLUMN" \
    --task classification \
    --clients 5 \
    --rounds 15
```

**Example for Continuous Outputs (Regression):**
```bash
./run_experiment.sh \
    --username your_name \
    --dataset "YOUR_DATASET_NAME.csv" \
    --target "YOUR_CONTINUOUS_TARGET" \
    --task regression \
    --clients 5 \
    --rounds 15
```

> **Advanced Simulation settings:** By default it runs identical hospital partitions. To change this, you currently need to modify `run_experiment.sh` itself to pass `--distribution non_iid` directly to the python execution command inside it.

### Step 3: View the Results!
The script automatically builds your reports without any extra commands!
Just check the `reports/` folder. All your visual graphs and textual reviews will be stored there as `reports/simulation_report_your_name_timestamp.md`.
