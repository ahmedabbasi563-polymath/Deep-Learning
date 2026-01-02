import numpy as np
from sklearn.preprocessing import OneHotEncoder

H, W = 256, 922

def shuffle(X, y, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

def normalize_signature(sig_1d, height=H, width=W):
    s = sig_1d.reshape(height, width)
    col_mean = np.mean(s, axis=0)
    s_centered = s - col_mean
    std = np.std(s)
    if std == 0:
        std = 1e-6
    return s_centered / std

def to_2d_batch(X_flat):
    return np.array([normalize_signature(x) for x in X_flat], dtype=np.float32)

def to_gray_cnn_input(X_2d):
    return X_2d[..., np.newaxis]  # (N, 256, 922, 1)

def one_hot(y_train, y_test):
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    y_train_oh = enc.fit_transform(y_train.reshape(-1, 1))
    y_test_oh = enc.transform(y_test.reshape(-1, 1))
    return y_train_oh, y_test_oh, enc
