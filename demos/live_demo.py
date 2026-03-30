"""
Live Demo - Real-time Federated Learning Visualization
Shows training progress and results as they happen
"""
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
from src.fl_model import LogisticRegression
from src.fl_data_preprocessing import load_and_preprocess_data
from src.fl_strategies import aggregate_fedavg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Enable interactive plotting
plt.ion()

def evaluate_quick(model, test_loader):
    """Quick evaluation for live demo"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_client_quick(model, train_loader, lr=0.01):
    """Quick training for demo"""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    return model.state_dict()

def live_federated_learning_demo(num_clients=3, num_rounds=10):
    """
    Live demo showing federated learning in action
    """
    print("="*70)
    print("🚀 LIVE FEDERATED LEARNING DEMO")
    print("="*70)
    print("\n📊 Loading dataset...")
    
    # Load data
    client_data, (X_test, y_test), num_classes, _ = load_and_preprocess_data(
        'data/Synthetic_patient-HealthCare-Monitoring_dataset.csv',
        'Predicted Disease',
        task_type='classification',
        num_clients=num_clients,
        distribution='non_iid'
    )
    
    print(f"✅ Loaded {len(client_data)} clients with non-IID data")
    
    # Create dataloaders
    client_loaders = []
    for X, y in client_data:
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )
        client_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True))
    
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    input_dim = client_data[0][0].shape[1]
    
    # Initialize global model
    global_model = LogisticRegression(input_dim, num_classes)
    
    # Store metrics for plotting
    round_accuracies = []
    client_accuracies = [[] for _ in range(num_clients)]
    
    # Setup live plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('🔴 LIVE: Federated Learning in Action', fontsize=16, fontweight='bold')
    
    print("\n🎬 Starting Federated Learning...")
    print("="*70)
    
    for round_idx in range(num_rounds):
        round_start = time.time()
        
        print(f"\n📍 Round {round_idx + 1}/{num_rounds}")
        print("-" * 70)
        
        # Get global weights
        global_weights = global_model.state_dict()
        client_updates = []
        
        # Simulate each client training
        for client_idx, train_loader in enumerate(client_loaders):
            print(f"  🏥 Hospital {client_idx + 1} training...", end=" ")
            
            # Local model
            local_model = LogisticRegression(input_dim, num_classes)
            local_model.load_state_dict(copy.deepcopy(global_weights))
            
            # Train locally
            w_local = train_client_quick(local_model, train_loader)
            client_updates.append(w_local)
            
            # Evaluate this client
            local_acc = evaluate_quick(local_model, test_loader)
            client_accuracies[client_idx].append(local_acc)
            
            print(f"✓ Accuracy: {local_acc:.2f}%")
        
        # Aggregate at server
        print(f"  🔄 Server aggregating models...", end=" ")
        new_state_dict = aggregate_fedavg(client_updates)
        global_model.load_state_dict(new_state_dict)
        
        # Evaluate global model
        global_acc = evaluate_quick(global_model, test_loader)
        round_accuracies.append(global_acc)
        
        round_time = time.time() - round_start
        
        print(f"✓")
        print(f"  🌐 Global Model Accuracy: {global_acc:.2f}%")
        print(f"  ⏱️  Round Time: {round_time:.2f}s")
        
        # Update plots
        ax1.clear()
        ax1.plot(range(1, len(round_accuracies) + 1), round_accuracies, 
                 'b-o', linewidth=2, markersize=8, label='Global Model')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Global Model Performance', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, 100])
        
        ax2.clear()
        for client_idx in range(num_clients):
            if client_accuracies[client_idx]:
                ax2.plot(range(1, len(client_accuracies[client_idx]) + 1), 
                        client_accuracies[client_idx], 
                        marker='o', label=f'Hospital {client_idx + 1}', linewidth=2)
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Individual Hospital Performance', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([0, 100])
        
        plt.tight_layout()
        plt.pause(0.1)
        plt.draw()
    
    # Final summary
    print("\n" + "="*70)
    print("✅ FEDERATED LEARNING COMPLETE!")
    print("="*70)
    print(f"\n📊 FINAL RESULTS:")
    print(f"  • Starting Accuracy: {round_accuracies[0]:.2f}%")
    print(f"  • Final Accuracy: {round_accuracies[-1]:.2f}%")
    print(f"  • Improvement: +{round_accuracies[-1] - round_accuracies[0]:.2f}%")
    print(f"  • Best Client: Hospital {np.argmax([acc[-1] for acc in client_accuracies]) + 1}")
    print(f"  • Worst Client: Hospital {np.argmin([acc[-1] for acc in client_accuracies]) + 1}")
    
    print("\n💡 Key Insight:")
    print("   All hospitals improved their models WITHOUT sharing patient data!")
    print("   This is the power of Federated Learning! 🎉")
    print("="*70)
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    print("\n" + "🎬 " * 20)
    print("\nWelcome to the LIVE Federated Learning Demo!")
    print("\nThis will show you:")
    print("  ✓ Real-time training across multiple hospitals")
    print("  ✓ Live accuracy improvements")
    print("  ✓ Visual comparison of all clients")
    print("\nPerfect for presentations and demonstrations!")
    print("\n" + "🎬 " * 20 + "\n")
    
    input("Press ENTER to start the demo...")
    
    live_federated_learning_demo(num_clients=3, num_rounds=10)
