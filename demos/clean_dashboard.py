"""
Clean Results Dashboard - Professional visualization
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_latest_results():
    """Load the most recent simulation results"""
    results_dir = Path('results')
    json_files = list(results_dir.glob('simulation_results_sushmitha_*.json'))
    
    if not json_files:
        print("No results found!")
        return None
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    return data, latest_file.name

def create_clean_dashboard(data):
    """Create a clean, professional dashboard with comprehensive metrics"""
    
    # Create figure with 3 rows: 2x2 grid on top, wide chart on bottom
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Top left
    ax2 = fig.add_subplot(gs[0, 1])  # Top right
    ax3 = fig.add_subplot(gs[1, 0])  # Middle left
    ax4 = fig.add_subplot(gs[1, 1])  # Middle right
    ax5 = fig.add_subplot(gs[2, :])  # Bottom - spans both columns
    
    fig.suptitle('Federated Learning Results - Final Report', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    experiments = data['experiments']
    colors = ['#27ae60', '#3498db', '#e67e22']  # Green, Blue, Orange
    
    # Get top 3 experiments
    sorted_exp = sorted(
        [(name, metrics) for name, metrics in experiments.items() if 'accuracy' in metrics],
        key=lambda x: x[1]['accuracy'][-1],
        reverse=True
    )[:3]
    
    # Chart 1: Accuracy Over Rounds
    ax1.set_title('Accuracy Improvement', fontsize=11, fontweight='bold', pad=8)
    
    for idx, (name, metrics) in enumerate(sorted_exp):
        rounds = range(1, len(metrics['accuracy']) + 1)
        label = name.split('_')[0]  # Just model name
        ax1.plot(rounds, metrics['accuracy'], marker='o', 
                linewidth=2, markersize=5, label=label, 
                color=colors[idx], alpha=0.9)
    
    ax1.set_xlabel('Round', fontsize=9)
    ax1.set_ylabel('Accuracy (%)', fontsize=9)
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax1.legend(loc='lower right', fontsize=8)
    ax1.tick_params(labelsize=8)
    ax1.set_ylim([0, 100])
    ax1.set_xlim([0.5, 10.5])
    
    # Chart 2: Final Accuracy Bars
    ax2.set_title('Final Performance', fontsize=11, fontweight='bold', pad=8)
    
    names = [n.split('_')[0] for n, _ in sorted_exp]
    accs = [m['accuracy'][-1] for _, m in sorted_exp]
    
    bars = ax2.barh(names, accs, color=colors)
    ax2.set_xlim([0, 100])
    ax2.grid(True, alpha=0.25, axis='x', linestyle='--', linewidth=0.5)
    ax2.invert_yaxis()  # Highest on top
    ax2.tick_params(labelsize=8)
    
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax2.text(acc + 1.5, i, f'{acc:.1f}%', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    # Chart 3: Loss Reduction
    ax3.set_title('Loss Reduction', fontsize=11, fontweight='bold', pad=8)
    
    for idx, (name, metrics) in enumerate(sorted_exp):
        if 'loss' in metrics:
            rounds = range(1, len(metrics['loss']) + 1)
            label = name.split('_')[0]
            ax3.plot(rounds, metrics['loss'], marker='o', linewidth=2, 
                    markersize=5, label=label, 
                    color=colors[idx], alpha=0.9)
    
    ax3.set_xlabel('Round', fontsize=9)
    ax3.set_ylabel('Loss', fontsize=9)
    ax3.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.tick_params(labelsize=8)
    ax3.set_xlim([0.5, 10.5])
    
    # Chart 4: Best Model Metrics
    ax4.set_title('Best Model: All Metrics', fontsize=11, fontweight='bold', pad=8)
    
    best_name, best_metrics = sorted_exp[0]
    
    metric_labels = []
    values = []
    
    if 'accuracy' in best_metrics:
        metric_labels.append('Accuracy')
        values.append(best_metrics['accuracy'][-1])
    
    for metric, label in [('f1', 'F1'), ('precision', 'Precision'), 
                           ('recall', 'Recall'), ('auc', 'AUC')]:
        if metric in best_metrics:
            metric_labels.append(label)
            values.append(best_metrics[metric][-1] * 100)
    
    bars = ax4.bar(metric_labels, values, color='#2ecc71', edgecolor='black', linewidth=0.8)
    ax4.set_ylim([0, 105])
    ax4.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.5)
    ax4.tick_params(axis='x', rotation=0, labelsize=8)
    ax4.tick_params(axis='y', labelsize=8)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{val:.0f}', ha='center', va='bottom', 
                fontsize=8, fontweight='bold')
    
    # Add model name as subtitle
    model_label = best_name.replace('_', ' ')
    ax4.text(0.5, -0.18, f'Model: {model_label}', 
            transform=ax4.transAxes, ha='center', fontsize=8, style='italic')
    
    # Chart 5: Comprehensive Metrics Comparison (Bottom - spans full width)
    ax5.set_title('Comprehensive Performance Metrics (Final Round)', 
                  fontsize=11, fontweight='bold', pad=10)
    
    metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC']
    x_pos = np.arange(len(metric_names))
    width = 0.25
    
    for idx, (name, metrics) in enumerate(sorted_exp):
        values = []
        
        # Get final values (convert to percentage scale)
        if 'accuracy' in metrics:
            values.append(metrics['accuracy'][-1])
        else:
            values.append(0)
        
        for metric in ['f1', 'precision', 'recall', 'auc']:
            if metric in metrics:
                values.append(metrics[metric][-1] * 100)
            else:
                values.append(0)
        
        offset = width * (idx - 1)  # Center the bars
        label = name.split('_')[0]
        strategy = name.split('_')[1] if '_' in name else ''
        full_label = f"{label} ({strategy})" if strategy else label
        
        bars = ax5.bar(x_pos + offset, values, width, 
                      label=full_label, color=colors[idx], alpha=0.8)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if height > 5:  # Only show if visible
                ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{int(height)}', ha='center', va='bottom', 
                        fontsize=8, fontweight='bold')
    
    ax5.set_ylabel('Score (%)', fontsize=10)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(metric_names, fontsize=10)
    ax5.set_ylim([0, 105])
    ax5.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.5)
    ax5.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax5.tick_params(labelsize=9)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig

def print_summary(data):
    """Print summary"""
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING RESULTS SUMMARY")
    print("="*70)
    
    metadata = data['metadata']
    experiments = data['experiments']
    
    print(f"\nExperiment: {metadata['username']}")
    print(f"Clients: {metadata['num_clients']} ({metadata['distribution']} distribution)")
    print(f"Rounds: {metadata['num_rounds']}")
    
    print(f"\nTop 3 Models (by Accuracy):")
    print("-" * 70)
    
    sorted_experiments = sorted(
        [(name, m) for name, m in experiments.items() if 'accuracy' in m],
        key=lambda x: x[1]['accuracy'][-1],
        reverse=True
    )[:3]
    
    for rank, (name, metrics) in enumerate(sorted_experiments, 1):
        acc = metrics['accuracy'][-1]
        f1 = metrics.get('f1', [0])[-1]
        print(f"{rank}. {name:35s} -> {acc:5.2f}% (F1: {f1:.4f})")
    
    print("="*70)

def main():
    print("\nCLEAN RESULTS DASHBOARD")
    print("Generating professional visualization...\n")
    
    result = load_latest_results()
    if result is None:
        return
    
    data, filename = result
    print(f"Loaded: {filename}\n")
    
    print_summary(data)
    
    print("\nCreating dashboard...")
    fig = create_clean_dashboard(data)
    
    output_path = Path('reports') / f'clean_dashboard_{filename.replace(".json", ".png")}'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    print("\nOpening interactive window...")
    print("(Close the window when done)\n")
    
    plt.show()

if __name__ == "__main__":
    main()
