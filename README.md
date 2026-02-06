# HealthCare Federated Learning Simulation

This project simulates a Federated Learning (FL) environment to train Machine Learning models on distributed healthcare data associated with patient monitoring. It allows for the comparison of different FL strategies and model architectures while ensuring data privacy (data remains on local clients).

## Overview

The simulation demonstrates how multiple hospitals (clients) can collaboratively train a global model without sharing their private patient datasets. It implements and compares three popular FL aggregation strategies:

*   **FedAvg (Federated Averaging):** The baseline standard for FL.
*   **FedProx:** robust to system heterogeneity.
*   **FedAdam:** An adaptive optimization strategy.

## Key Features

*   **Multi-Model Support:** Train `LogisticRegression`, `SimpleMLP`, or `DeepMLP` models.
*   **Configurable Environment:** Easily adjust the number of clients, communication rounds, local epochs, and batch sizes in `fl_simulation.py`.
*   **Synthetic Healthcare Data:** Uses a synthetic patient health monitoring dataset.
*   **Performance Metrics:** Tracks Accuracy and Loss across communication rounds.
*   **Automated Reporting:** Generates simulation reports and visualizations.

## Project Structure

*   `fl_simulation.py`: The main entry point for running the simulation. Manages the server, clients, and training loop.
*   `fl_model.py`: Definitions of the PyTorch neural network models.
*   `fl_strategies.py`: Implementation of FL aggregation strategies (FedAvg, etc.).
*   `fl_data_preprocessing.py`: Loading and splitting of the dataset for clients.
*   `generate_report.py`: Tool to generate Markdown reports from simulation results.
*   `requirements.txt`: Python dependencies.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Simulation

To start the federated learning simulation, run the `fl_simulation.py` script:

```bash
python fl_simulation.py
```

This will:
1.  Load and preprocess the data.
2.  Run simulations for defined models and strategies.
3.  Save the results in the `results/` directory as a JSON file.
4.  Print progress and final metrics to the console.

### Generating Reports

After running a simulation, you can generate a readable report:

```bash
python generate_report.py
```

Check the generated `simulation_report.md` (or similar) for a summary of the results.

## Results & Analysis

Detailed analysis and comparison of the strategies can be found in `simulation_report_2.md`.
Current benchmarks indicate:
*   **FedAvg** generally provides a strong baseline accuracy.
*   **FedProx** offers stability in heterogeneous settings.
*   **FedAdam** may require tuning but offers adaptive convergence properties.
