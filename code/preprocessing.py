import numpy as np
from sklearn.preprocessing import OneHotEncoder

def shuffle_data(X, y):
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

def normalize_spectrogram(sig_1d, height=256, width=922):
    sig_2d = sig_1d.reshape(height, width)
    col_mean = np.mean(sig_2d, axis=0)
    sig_centered = sig_2d - col_mean
    global_std = np.std(sig_2d)

    if global_std == 0:
        global_std = 1e-6

    return sig_centered / global_std

def preprocess_data(X):
    X_norm = np.array([normalize_spectrogram(sig) for sig in X])
    return X_norm[..., np.newaxis]

def encode_labels(y_train, y_test):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_oh = encoder.transform(y_test.reshape(-1, 1))
    return y_train_oh, y_test_oh, encoder
