import os
import mat73

def load_urc_dataset(base_path: str):
    train_data = mat73.loadmat(os.path.join(base_path, "URC_drones_S_dB_flat_train_dataset_v1.mat"))
    test_data  = mat73.loadmat(os.path.join(base_path, "URC_drones_S_dB_flat_test_dataset_v1.mat"))
    train_lbls = mat73.loadmat(os.path.join(base_path, "URC_drones_y_labels_train_dataset_v1.mat"))
    test_lbls  = mat73.loadmat(os.path.join(base_path, "URC_drones_y_labels_test_dataset_v1.mat"))

    X_train = train_data["S_dB_flat"]
    X_test  = test_data["S_dB_flat"]

    # labels are 1..5 in dataset -> convert to 0..4
    y_train = train_lbls["y_label"] - 1
    y_test  = test_lbls["y_label"] - 1

    return X_train, X_test, y_train, y_test
