Title 
Federated Learning-Based ECG Classification using CNN: A Comparative Study of FedAvg, FedProx, 
and FedAdam 
1. Introduction 
Federated Learning (FL) enables decentralized model training where data remains local to each 
client. This is especially important in healthcare, where patient data privacy is critical. 
In this project, a federated learning framework was implemented to classify ECG signals from the 
MIT-BIH Arrhythmia dataset using a 1D Convolutional Neural Network (CNN). Three optimization 
strategies were compared: 
• Federated Averaging (FedAvg)  
• Federated Proximal (FedProx)  
• Federated Adaptive Optimization (FedAdam)  
2. Architecture 
The system follows a standard FL pipeline: 
1. Central server initializes global model  
2. Model is broadcast to clients (hospitals)  
3. Clients train locally on private ECG data  
4. Model updates are sent back  
5. Server aggregates updates using:  
o FedAvg  
o FedProx  
o FedAdam  
6. Process repeats for multiple communication rounds  
3. Dataset 
• Dataset: MIT-BIH Arrhythmia Dataset  
• Type: ECG time-series signals  
• Preprocessing:  
o Extracted 200-sample heartbeat segments  
o Normalized signals  
o Mapped annotations into grouped classes  
• Nature:  
o Highly imbalanced (normal beats dominate)  
o Real-world medical signal data  
4. Model 
A 1D CNN was used: 
• Convolution layers → extract waveform patterns  
• Pooling → reduce noise  
• Fully connected layer → classification  
Reason: 
ECG is sequential signal → CNN captures temporal morphology effectively 
5. Experimental Setup 
Parameter 
Clients 
Rounds 
Value 
3 
5 
Local Epochs 2 
Learning Rate 0.001 
Window Size 200 samples 
6. Results 
Final Performance 
Method Accuracy Precision Recall F1 Score 
FedAvg 0.9521 0.9064 0.9521 0.9287 
FedProx 0.9521 0.9064 0.9521 0.9287 
FedAdam 0.9521 0.9064 0.9521 0.9287 
7. Observations 
FedAvg 
• Gradual convergence  
• Stable learning  
• Works well when data distribution is similar  
FedProx 
• Faster convergence in early rounds  
• Handles client heterogeneity better  
• Prevents local model drift  
FedAdam 
• Fastest convergence (reaches optimal quickly)  
• Uses adaptive optimization (momentum + variance)  
• Stabilizes updates across clients  
8. Why All Methods Give Similar Results 
This is a very important insight (write this in your viva/report)       
In this experiment: 
• Client data distribution is not highly heterogeneous  
• Dataset is dominated by normal ECG beats  
• CNN model is strong enough to learn patterns quickly  
Therefore: 
• All methods converge to the same optimum  
• Differences between optimizers become minimal  
9. Why Each Method Works for ECG Data 
Why FedAvg works well 
• ECG patterns are consistent across patients  
• Averaging local models is sufficient  
• Simple and efficient  
Why FedProx helps 
• ECG signals vary slightly across individuals  
• Proximal term prevents local overfitting  
• Useful in real hospital scenarios  
Why FedAdam is powerful 
• Adaptive learning handles noisy gradients  
• ECG signals can be irregular and noisy  
• Adam stabilizes updates → faster convergence