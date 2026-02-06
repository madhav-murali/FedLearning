import torch
import torch.nn as nn
import torch.optim as optim
import copy
import json
import os
import time
from fl_model import SimpleMLP, LogisticRegression, DeepMLP
from fl_data_preprocessing import load_and_preprocess_data
from fl_strategies import aggregate_fedavg, proximal_term, FedAdamServer

# --- Configuration ---
NUM_CLIENTS = 5
NUM_ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01
PROXIMAL_MU = 0.01 # For FedProx
DATA_PATH = 'Synthetic_patient-HealthCare-Monitoring_dataset.csv'
RESULTS_DIR = 'results'

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return loss_sum / len(test_loader), 100 * correct / total

def train_client(model, train_loader, epochs, lr, strategy='FedAvg', global_model=None):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if strategy == 'FedProx' and global_model is not None:
                loss += proximal_term(model, global_model, PROXIMAL_MU)
                
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

def get_model(model_name, input_dim, output_dim):
    if model_name == 'LogisticRegression':
        return LogisticRegression(input_dim, output_dim)
    elif model_name == 'SimpleMLP':
        return SimpleMLP(input_dim, output_dim)
    elif model_name == 'DeepMLP':
        return DeepMLP(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_simulation(strategy_name, model_name, client_loaders, test_loader, input_dim, output_dim):
    print(f"--- Starting Simulation: Strategy={strategy_name}, Model={model_name} ---")
    
    # Initialize Global Model
    global_model = get_model(model_name, input_dim, output_dim)
    
    # FedAdam State
    fed_adam_server = None
    if strategy_name == 'FedAdam':
        fed_adam_server = FedAdamServer(global_model, lr=0.01) 
    
    metrics = {'accuracy': [], 'loss': [], 'times': []}
    
    start_time_sim = time.time()
    
    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        
        # Distribute Global Weights
        global_weights = global_model.state_dict()
        client_updates = []
        
        for client_idx, train_loader in enumerate(client_loaders):
            local_model = get_model(model_name, input_dim, output_dim)
            local_model.load_state_dict(copy.deepcopy(global_weights))
            
            # Pass global_model copy for FedProx calculation if needed
            global_model_ref = copy.deepcopy(global_model) if strategy_name == 'FedProx' else None
            
            w_local = train_client(local_model, train_loader, LOCAL_EPOCHS, LEARNING_RATE, strategy_name, global_model_ref)
            client_updates.append(w_local)
        
        # Aggregation
        if strategy_name == 'FedAdam':
            new_state_dict = fed_adam_server.step(global_model, client_updates)
        else:
            new_state_dict = aggregate_fedavg(client_updates)
            global_model.load_state_dict(new_state_dict)
        
        # Evaluation
        val_loss, val_acc = evaluate(global_model, test_loader)
        
        metrics['loss'].append(val_loss)
        metrics['accuracy'].append(val_acc)
        metrics['times'].append(time.time() - round_start)
        
        print(f"[{model_name}][{strategy_name}] Round {round_idx+1}/{NUM_ROUNDS} | Acc: {val_acc:.2f}% | Loss: {val_loss:.4f}")

    total_time = time.time() - start_time_sim
    print(f"Finished {strategy_name} with {model_name} in {total_time:.2f}s")
    
    return metrics

def main():
    # Load Data
    client_data, (X_test, y_test), num_classes, _ = load_and_preprocess_data(DATA_PATH, NUM_CLIENTS)
    
    if client_data is None:
        print("Dataset load failed.")
        return

    # Convert to Loaders
    client_loaders = []
    for X, y in client_data:
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        client_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True))
    
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    input_dim = client_data[0][0].shape[1]
    
    strategies = ['FedAvg', 'FedProx', 'FedAdam']
    models = ['LogisticRegression', 'SimpleMLP', 'DeepMLP']
    results = {}
    
    # 1. Compare Models using FedAvg (Baseline strategy)
    print("\n=== Phase 1: Model Comparison (using FedAvg) ===")
    for model_name in models:
        key = f"{model_name}_FedAvg"
        results[key] = run_simulation('FedAvg', model_name, client_loaders, test_loader, input_dim, num_classes)

    # 2. Compare Strategies using the Best Model (Let's assume DeepMLP is usually best, or run all combinations)
    # For comprehensiveness, let's run strategies on DeepMLP specifically as it's the most capable.
    print("\n=== Phase 2: Strategy Comparison (using DeepMLP) ===")
    target_model = 'DeepMLP'
    for strategy in strategies:
        if strategy == 'FedAvg': continue # Already run above
        key = f"{target_model}_{strategy}"
        results[key] = run_simulation(strategy, target_model, client_loaders, test_loader, input_dim, num_classes)
        
    # Save Results
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'simulation_results_{timestamp}.json'
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f)
    
    print(f"Results saved to {filepath}")

if __name__ == "__main__":
    main()
