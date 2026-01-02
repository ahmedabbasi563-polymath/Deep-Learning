# UAV RF Signal Classification using LeNet CNN
# Author: Ahmed Abbasi

import os
import numpy as np
import mat73
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ==============================
# 1. Load Dataset
# ==============================

BASE_PATH = "/content/drive/MyDrive/Project Work/URC drones dataset v1"

train_data = mat73.loadmat(os.path.join(BASE_PATH, "URC_drones_S_dB_flat_train_dataset_v1.mat"))
test_data  = mat73.loadmat(os.path.join(BASE_PATH, "URC_drones_S_dB_flat_test_dataset_v1.mat"))
train_lbls = mat73.loadmat(os.path.join(BASE_PATH, "URC_drones_y_labels_train_dataset_v1.mat"))
test_lbls  = mat73.loadmat(os.path.join(BASE_PATH, "URC_drones_y_labels_test_dataset_v1.mat"))

X_train = train_data["S_dB_flat"]
X_test  = test_data["S_dB_flat"]
y_train = train_lbls["y_label"] - 1
y_test  = test_lbls["y_label"] - 1

# ==============================
# 2. Shuffle Dataset
# ==============================

def shuffle(X, y):
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

X_train, y_train = shuffle(X_train, y_train)

# ==============================
# 3. Normalize Spectrograms
# ==============================

def normalize_signature(sig_1d, height=256, width=922):
    sig_2d = sig_1d.reshape(height, width)
    col_mean = np.mean(sig_2d, axis=0)
    sig_centered = sig_2d - col_mean
    global_std = np.std(sig_2d)
    if global_std == 0:
        global_std = 1e-6
    return sig_centered / global_std

X_train_norm = np.array([normalize_signature(sig) for sig in X_train])
X_test_norm  = np.array([normalize_signature(sig) for sig in X_test])

X_train_norm = X_train_norm[..., np.newaxis]
X_test_norm  = X_test_norm[..., np.newaxis]

# ==============================
# 4. One-Hot Encode Labels
# ==============================

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_oh  = encoder.transform(y_test.reshape(-1, 1))

# ==============================
# 5. Build LeNet CNN (Grayscale)
# ==============================

model = models.Sequential([
    layers.InputLayer(input_shape=(256, 922, 1)),
    layers.Conv2D(6, (5, 5), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (5, 5), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation="relu"),
    layers.Dense(84, activation="relu"),
    layers.Dense(5, activation="softmax")
])

model.compile(
    optimizer=optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# 6. Train Model
# ==============================

history = model.fit(
    X_train_norm,
    y_train_oh,
    batch_size=128,
    epochs=20,
    validation_data=(X_test_norm, y_test_oh),
    verbose=1
)

# ==============================
# 7. Evaluate Model
# ==============================

test_loss, test_acc = model.evaluate(X_test_norm, y_test_oh, verbose=1)
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

# ==============================
# 8. Confusion Matrix
# ==============================

y_pred = model.predict(X_test_norm).argmax(axis=1)
y_true = y_test_oh.argmax(axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ==============================
# 9. SNR Robustness Analysis
# ==============================

snr_levels = [-30, -25, -20, -15, -10, -5, 0, 5]
samples_per_snr = 300

def snr_accuracy(model, X, y, snr_levels, samples_per_snr):
    results = {}
    for i, snr in enumerate(snr_levels):
        start = i * samples_per_snr
        end = (i + 1) * samples_per_snr
        X_snr = X[start:end]
        y_snr = y[start:end]

        preds = model.predict(X_snr).argmax(axis=1)
        true  = y_snr.argmax(axis=1)
        acc = np.mean(preds == true)
        results[snr] = acc * 100
        print(f"SNR {snr} dB: Accuracy = {acc * 100:.2f}%")
    return results

snr_results = snr_accuracy(model, X_test_norm, y_test_oh, snr_levels, samples_per_snr)

plt.figure(figsize=(8, 5))
plt.plot(list(snr_results.keys()), list(snr_results.values()), marker="o")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy vs SNR Level")
plt.grid(True)
plt.show()
