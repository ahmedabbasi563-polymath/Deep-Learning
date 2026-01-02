import os
import mat73
import numpy as np

def load_dataset(base_path):
    """
    Load URC Drone RF spectrogram dataset from .mat files
    """
    train_data = mat73.loadmat(os.path.join(base_path,
        "URC_drones_S_dB_flat_train_dataset_v1.mat"))
    test_data = mat73.loadmat(os.path.join(base_path,
        "URC_drones_S_dB_flat_test_dataset_v1.mat"))

    train_labels = mat73.loadmat(os.path.join(base_path,
        "URC_drones_y_labels_train_dataset_v1.mat"))
    test_labels = mat73.loadmat(os.path.join(base_path,
        "URC_drones_y_labels_test_dataset_v1.mat"))

    X_train = train_data["S_dB_flat"]
    X_test = test_data["S_dB_flat"]

    # Convert labels to zero-based indexing
    y_train = train_labels["y_label"] - 1
    y_test = test_labels["y_label"] - 1

    return X_train, X_test, y_train, y_test
