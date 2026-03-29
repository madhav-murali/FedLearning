import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(filepath, target_col, task_type='classification', num_clients=5, distribution='iid'):
    """
    Loads dataset, preprocesses features automatically (numeric/categorical), encodes target, 
    and partitions data for FL clients (IID or Non-IID skew).
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None, None, None

    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in the dataset.")
        return None, None, None, None

    # Separate features and target
    y = df[target_col].values
    X_df = df.drop(columns=[target_col])
    feature_cols = X_df.columns.tolist()

    # Identify numerical and categorical columns
    numeric_features = X_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Create preprocessing pipelines for numerical and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X = preprocessor.fit_transform(X_df)

    # Encode target
    num_classes = 1
    if task_type == 'classification':
        le = LabelEncoder()
        y = le.fit_transform(y)
        num_classes = len(le.classes_)
    else:
        # For regression, ensure floats
        y = y.astype(np.float32)

    # Split into train/test (Server test set)
    # Stratify only if classification
    stratify_col = y if task_type == 'classification' else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_col
    )

    # Partition training data among clients
    client_data = []
    
    if distribution == 'iid':
        # Simple random partition (IID-like)
        indices = np.random.permutation(len(X_train))
        split_indices = np.array_split(indices, num_clients)
        for idxs in split_indices:
            client_data.append((X_train[idxs], y_train[idxs]))
    
    elif distribution == 'non_iid' and task_type == 'classification':
        # Sort data by class to create a non-IID split easily (quantity skew)
        # This simulates different hospitals seeing specific patient demographics/diseases
        sort_idx = np.argsort(y_train)
        X_train_sorted = X_train[sort_idx]
        y_train_sorted = y_train[sort_idx]
        
        split_indices = np.array_split(np.arange(len(X_train)), num_clients)
        for idxs in split_indices:
            client_data.append((X_train_sorted[idxs], y_train_sorted[idxs]))
            
    else: # Fallback to IID for regression or unknown types
        indices = np.random.permutation(len(X_train))
        split_indices = np.array_split(indices, num_clients)
        for idxs in split_indices:
            client_data.append((X_train[idxs], y_train[idxs]))

    print(f"Data Loaded: {len(X)} samples.")
    print(f"Numeric Features: {len(numeric_features)} | Categorical Features: {len(categorical_features)}")
    print(f"Clients: {num_clients}, Distribution: {distribution.upper()}, Test Set Size: {len(X_test)}")
    if task_type == 'classification':
        print(f"Target Classes: {num_classes}")
    else:
        print(f"Target is Continuous (Regression).")
    
    return client_data, (X_test, y_test), num_classes, feature_cols
