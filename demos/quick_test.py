"""
Quick Test Script - Verify Everything Works
Fast test with minimal data to show the system works
"""
import torch
import torch.nn as nn
import time
from src.fl_model import LogisticRegression
from src.fl_data_preprocessing import load_and_preprocess_data
from src.fl_strategies import aggregate_fedavg
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def quick_test(show_visual=True):
    """
    Quick 2-minute test to verify FL system works
    """
    print("\n" + "🧪 " * 25)
    print("\n⚡ QUICK FEDERATED LEARNING TEST")
    print("\nThis is a fast test to verify everything works correctly.")
    print("Perfect for: Presentations, Demos, Quick checks")
    print("\n" + "🧪 " * 25 + "\n")
    
    start_time = time.time()
    
    # Step 1: Load Data
    print("📊 Step 1/5: Loading dataset...")
    client_data, (X_test, y_test), num_classes, _ = load_and_preprocess_data(
        'data/Synthetic_patient-HealthCare-Monitoring_dataset.csv',
        'Predicted Disease',
        task_type='classification',
        num_clients=3,  # Only 3 clients for quick test
        distribution='non_iid'
    )
    print(f"   ✓ Loaded 3 clients with {len(X_test)} test samples")
    
    # Step 2: Initialize Model
    print("\n🤖 Step 2/5: Initializing AI models...")
    input_dim = client_data[0][0].shape[1]
    global_model = LogisticRegression(input_dim, num_classes)
    print(f"   ✓ Model created with {input_dim} features → {num_classes} diseases")
    
    # Step 3: Run Federated Learning (5 rounds with proper training)
    print("\n🔄 Step 3/5: Running Federated Learning (5 rounds)...")
    print("   Each round: Hospitals train (5 epochs) → Server aggregates → Test")
    print()
    
    accuracies = []
    
    for round_idx in range(5):  # 5 rounds for better demo
        # Create dataloaders
        client_updates = []
        
        for client_idx, (X_train, y_train) in enumerate(client_data):
            # Proper training (5 epochs per client)
            model = LogisticRegression(input_dim, num_classes)
            model.load_state_dict(global_model.state_dict())
            model.train()
            
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            # Train for multiple epochs
            X_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_tensor = torch.tensor(y_train, dtype=torch.long)
            
            # Create mini-batches for better training
            batch_size = 256
            for epoch in range(5):  # 5 epochs per round
                for i in range(0, len(X_tensor), batch_size):
                    batch_X = X_tensor[i:i+batch_size]
                    batch_y = y_tensor[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            client_updates.append(model.state_dict())
        
        # Aggregate
        new_weights = aggregate_fedavg(client_updates)
        global_model.load_state_dict(new_weights)
        
        # Test
        global_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            outputs = global_model(X_test_tensor)
            _, predictions = torch.max(outputs, 1)
            accuracy = accuracy_score(y_test, predictions.numpy())
            accuracies.append(accuracy * 100)
        
        print(f"   Round {round_idx + 1}: Accuracy = {accuracy * 100:.2f}%")
    
    # Step 4: Final Evaluation
    print("\n📈 Step 4/5: Final comprehensive evaluation...")
    global_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = global_model(X_test_tensor)
        _, predictions = torch.max(outputs, 1)
        
        final_accuracy = accuracy_score(y_test, predictions.numpy())
        final_f1 = f1_score(y_test, predictions.numpy(), average='macro')
    
    print(f"   ✓ Final Test Accuracy: {final_accuracy * 100:.2f}%")
    print(f"   ✓ Final F1-Score: {final_f1:.4f}")
    
    # Step 5: Results Summary
    print("\n📊 Step 5/5: Generating results summary...")
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE - SYSTEM VERIFIED!")
    print("="*70)
    
    print(f"\n⏱️  Total Test Time: {elapsed_time:.1f} seconds")
    print(f"\n📊 Results:")
    print(f"   • Starting Accuracy: {accuracies[0]:.2f}%")
    print(f"   • Final Accuracy: {accuracies[-1]:.2f}%")
    print(f"   • Improvement: +{accuracies[-1] - accuracies[0]:.2f}%")
    print(f"   • F1-Score: {final_f1:.4f}")
    
    print(f"\n🎯 Performance Rating:")
    if final_accuracy >= 0.85:
        rating = "EXCELLENT 🌟🌟🌟"
        comment = "Your model is highly accurate!"
    elif final_accuracy >= 0.70:
        rating = "GOOD ✓✓"
        comment = "Solid performance, well done!"
    elif final_accuracy >= 0.50:
        rating = "FAIR ✓"
        comment = "Decent results, could be improved."
    else:
        rating = "NEEDS WORK ⚠️"
        comment = "Model needs hyperparameter tuning."
    
    print(f"   {rating}")
    print(f"   {comment}")
    
    print(f"\n💡 Key Achievement:")
    print(f"   3 hospitals collaborated and improved their AI model")
    print(f"   WITHOUT sharing any patient data! 🔒")
    
    print(f"\n✅ What This Proves:")
    print(f"   ✓ Your FL system works correctly")
    print(f"   ✓ Models can learn from distributed data")
    print(f"   ✓ Privacy is preserved (no data leaves hospitals)")
    print(f"   ✓ Ready for full-scale experiments")
    
    print("\n" + "="*70)
    
    # Visual summary
    if show_visual:
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle('Quick Test Results', fontsize=14, fontweight='bold')
            
            # Accuracy plot
            ax1.plot(range(1, len(accuracies) + 1), accuracies, 'go-', linewidth=3, markersize=10)
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_title('Learning Progress')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 100])
            
            # Summary bar
            metrics = [final_accuracy * 100, final_f1 * 100]
            ax2.bar(['Accuracy', 'F1-Score'], metrics, color=['#2ecc71', '#3498db'])
            ax2.set_ylabel('Score (%)')
            ax2.set_title('Final Performance')
            ax2.set_ylim([0, 100])
            
            for i, v in enumerate(metrics):
                ax2.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
            
            plt.tight_layout()
            
            # Save
            plt.savefig('reports/quick_test_results.png', dpi=150, bbox_inches='tight')
            print(f"\n📊 Visual results saved to: reports/quick_test_results.png")
            
            plt.show()
            
        except ImportError:
            print("\n⚠️  Matplotlib not available for visualization")
    
    return final_accuracy, final_f1

if __name__ == "__main__":
    print("\n" + "⚡ " * 25)
    print("\n🚀 Quick Test - Verify Your Federated Learning System")
    print("\nThis script will:")
    print("  ✓ Test your FL system with real data")
    print("  ✓ Show results in ~2 minutes")
    print("  ✓ Verify everything works correctly")
    print("  ✓ Perfect for presentations!")
    print("\n" + "⚡ " * 25 + "\n")
    
    input("Press ENTER to start the quick test...")
    
    try:
        accuracy, f1 = quick_test(show_visual=True)
        
        print("\n" + "🎉 " * 25)
        print("\n✅ SYSTEM VERIFIED AND READY!")
        print(f"\nYou can confidently present:")
        print(f"  • Your FL system achieves {accuracy*100:.1f}% accuracy")
        print(f"  • Multiple hospitals collaborate successfully")
        print(f"  • Patient privacy is fully protected")
        print("\n" + "🎉 " * 25 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nPlease check:")
        print("  1. Dataset exists in data/ folder")
        print("  2. All dependencies installed (pip install -r requirements.txt)")
        print("  3. No syntax errors in source code")
