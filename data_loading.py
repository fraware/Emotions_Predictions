import pandas as pd
from sklearn.preprocessing import LabelEncoder, train_test_split
import numpy as np

def load_and_process_data(file_path, train_size, random_state):
    """
    Load and process EEG data for emotion recognition.

    Parameters:
    file_path (str): Path to the CSV file containing EEG data.
    train_size (float): Proportion of the dataset to include in the train split.
    random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    tuple: Tuple containing (X_train, X_test, y_train, y_test) after processing.
    """
    try:
        # Load data
        data = pd.read_csv(file_path)

    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")

    # Check for 'label' column in the dataset
    if 'label' not in data.columns:
        raise ValueError("Missing 'label' column in the dataset")

    # Encode labels
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])

    # Split data
    y = data.pop('label')
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
    
    # Reshape data for deep learning models
    X_train = X_train.to_numpy().reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))

    # Convert labels to one-hot encoding
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    return X_train, X_test, y_train, y_test
