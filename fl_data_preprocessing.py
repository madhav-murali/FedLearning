import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(filepath, num_clients=5):
    """
    Loads dataset, preprocesses features, encodes target, and partitions data for FL clients.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None, None, None

    # Select features and target
    feature_cols = [
        'Heart Rate (bpm)', 
        'SpO2 Level (%)', 
        'Systolic Blood Pressure (mmHg)', 
        'Diastolic Blood Pressure (mmHg)', 
        'Body Temperature (°C)'
    ]
    target_col = 'Predicted Disease'

    X = df[feature_cols].values
    y = df[target_col].values

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train/test (Server test set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Partition training data among clients (IID for simplicity, or non-IID if desired)
    # Here we do a simple random partition (IID-like)
    client_data = []
    indices = np.random.permutation(len(X_train))
    split_indices = np.array_split(indices, num_clients)

    for idxs in split_indices:
        X_client = X_train[idxs]
        y_client = y_train[idxs]
        client_data.append((X_client, y_client))

    print(f"Data Loaded: {len(X)} samples.")
    print(f"Features: {feature_cols}")
    print(f"Target Classes: {le.classes_}")
    print(f"Clients: {num_clients}, Test Set Size: {len(X_test)}")
    
    return client_data, (X_test, y_test), num_classes, feature_cols
