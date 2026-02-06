import json
import matplotlib.pyplot as plt
import os
import glob
import datetime

RESULTS_DIR = 'results'

def generate_report():
    # Find latest results file
    list_of_files = glob.glob(os.path.join(RESULTS_DIR, 'simulation_results_*.json'))
    if not list_of_files:
        print("No results file found. Run fl_simulation.py first.")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    timestamp = latest_file.split('simulation_results_')[-1].replace('.json', '')
    
    print(f"Generating report for: {latest_file}")

    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    keys = list(results.keys())
    model_experiments = [k for k in keys if '_FedAvg' in k]
    strategy_experiments = [k for k in keys if 'DeepMLP_' in k]
    
    rounds = range(1, len(results[keys[0]]['accuracy']) + 1)
    
    # Define Plot filenames with timestamp
    model_plot_file = f'results/model_accuracy_plot_{timestamp}.png'
    strategy_plot_file = f'results/strategy_accuracy_plot_{timestamp}.png'
    report_file = f'simulation_report_{timestamp}.md'
    
    # 1. Plot Model Comparison (FedAvg)
    plt.figure(figsize=(10, 6))
    for key in model_experiments:
        label = key.replace('_FedAvg', '')
        plt.plot(rounds, results[key]['accuracy'], label=label)
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Model Comparison (Strategy: FedAvg) - {timestamp}')
    plt.legend()
    plt.grid(True)
    plt.savefig(model_plot_file)
    plt.close() # Close to prevent overlap
    
    # 2. Plot Strategy Comparison (DeepMLP)
    plt.figure(figsize=(10, 6))
    for key in strategy_experiments:
        label = key.replace('DeepMLP_', '')
        plt.plot(rounds, results[key]['accuracy'], label=label)
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Strategy Comparison (Model: DeepMLP) - {timestamp}')
    plt.legend()
    plt.grid(True)
    plt.savefig(strategy_plot_file)
    plt.close()
    
    # 3. Create Markdown Report
    with open(report_file, 'w') as f:
        f.write(f"# Federated Learning Simulation Report\n")
        f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Source Data**: `{latest_file}`\n\n")
        
        f.write("## 1. Introduction\n")
        f.write("This report compares different model architectures (**Logistic Regression**, **Simple MLP**, **Deep MLP**) and FL algorithms (**FedAvg**, **FedProx**, **FedAdam**).\n\n")
        
        f.write("## 2. Models & Strategies\n")
        f.write("- **Models Evaluated**: Logistic Regression, Simple MLP, Deep MLP.\n")
        f.write("- **Strategies Evaluated**: FedAvg, FedProx, FedAdam.\n\n")
        
        f.write("## 3. Results Comparison\n\n")
        f.write("### Model Performance (Fixed Strategy: FedAvg)\n")
        f.write("| Model | Final Accuracy | Final Loss |\n")
        f.write("|-------|----------------|------------|\n")
        for key in model_experiments:
            name = key.replace('_FedAvg', '')
            acc = results[key]['accuracy'][-1]
            loss = results[key]['loss'][-1]
            f.write(f"| {name} | {acc:.2f}% | {loss:.4f} |\n")
            
        f.write("\n")
        f.write("### Algorithm Performance (Fixed Model: DeepMLP)\n")
        f.write("| Algorithm | Final Accuracy | Final Loss |\n")
        f.write("|-----------|----------------|------------|\n")
        for key in strategy_experiments:
            name = key.replace('DeepMLP_', '')
            acc = results[key]['accuracy'][-1]
            loss = results[key]['loss'][-1]
            f.write(f"| {name} | {acc:.2f}% | {loss:.4f} |\n")

        f.write("\n\n")
        f.write("## 4. Visualizations\n\n")
        f.write("### Model Comparison\n")
        f.write(f"![Model Comparison]({model_plot_file})\n\n")
        
        f.write("### Strategy Comparison\n")
        f.write(f"![Strategy Comparison]({strategy_plot_file})\n\n")
        
    print(f"Report generated: {report_file}")

if __name__ == "__main__":
    generate_report()
