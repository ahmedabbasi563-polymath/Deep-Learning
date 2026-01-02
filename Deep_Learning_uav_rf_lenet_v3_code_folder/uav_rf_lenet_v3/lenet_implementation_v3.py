"""
LeNet Implementation v3 — UAV RF Spectrogram Classification
===========================================================

This script keeps the *original notebook-style flow* (single file) and is intended
to mirror the code structure shown in the PDF (LeNet Implementation v3).

It includes:
- Loading URC drones dataset v1 (.mat)
- Shuffling
- Spectrogram normalization (column mean subtraction + global std)
- One-hot encoding
- Model 1: Grayscale LeNet CNN (256x922x1)
- Evaluation: accuracy + confusion matrix
- SNR robustness sweep (-30 dB to +5 dB) assuming test set is ordered by SNR blocks
- Model 2: RGB (jet colormap) spectrogram pipeline (224x224x3) + LeNet-style CNN

NOTE:
- Update BASE_PATH below to point to your dataset folder.
"""

import os
import numpy as np
import mat73
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Optional dependency for resizing RGB images
from PIL import Image

# ----------------------------
# 0) Paths (EDIT THIS)
# ----------------------------
BASE_PATH = r"/content/drive/MyDrive/Project Work/URC drones dataset v1"
# If running locally on Windows, it might look like:
# BASE_PATH = r"D:\Project Work\URC drones dataset v1"

# ----------------------------
# 1) Load .mat files
# ----------------------------
train_data = mat73.loadmat(os.path.join(BASE_PATH, "URC_drones_S_dB_flat_train_dataset_v1.mat"))
test_data  = mat73.loadmat(os.path.join(BASE_PATH, "URC_drones_S_dB_flat_test_dataset_v1.mat"))
train_lbls = mat73.loadmat(os.path.join(BASE_PATH, "URC_drones_y_labels_train_dataset_v1.mat"))
test_lbls  = mat73.loadmat(os.path.join(BASE_PATH, "URC_drones_y_labels_test_dataset_v1.mat"))

X_train = train_data["S_dB_flat"]
X_test  = test_data["S_dB_flat"]

# labels are 1..5 in dataset -> convert to 0..4
y_train = train_lbls["y_label"] - 1
y_test  = test_lbls["y_label"] - 1

print("Train X:", X_train.shape, " Train y:", y_train.shape)
print("Test  X:", X_test.shape,  " Test  y:", y_test.shape)

# ----------------------------
# 2) Shuffle train set
# ----------------------------
def shuffle_in_unison(X, y, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

X_train, y_train = shuffle_in_unison(X_train, y_train, seed=42)

# ----------------------------
# 3) Normalize spectrograms
#    - reshape to (256, 922)
#    - subtract column mean
#    - divide by global std (with safety)
# ----------------------------
H, W = 256, 922

def normalize_signature(sig_1d, height=H, width=W):
    sig_2d = sig_1d.reshape(height, width)
    col_mean = np.mean(sig_2d, axis=0)
    sig_centered = sig_2d - col_mean
    global_std = np.std(sig_2d)
    if global_std == 0:
        global_std = 1e-6
    return sig_centered / global_std

X_train_norm_2d = np.array([normalize_signature(sig) for sig in X_train], dtype=np.float32)
X_test_norm_2d  = np.array([normalize_signature(sig) for sig in X_test], dtype=np.float32)

# For grayscale CNN: add channel dim -> (N, 256, 922, 1)
X_train_norm = X_train_norm_2d[..., np.newaxis]
X_test_norm  = X_test_norm_2d[..., np.newaxis]

print("X_train_norm:", X_train_norm.shape)
print("X_test_norm :", X_test_norm.shape)

# ----------------------------
# 4) One-hot encode labels
# ----------------------------
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_oh  = encoder.transform(y_test.reshape(-1, 1))

print("y_train_oh:", y_train_oh.shape)
print("y_test_oh :", y_test_oh.shape)

# ----------------------------
# 5) Model 1: Grayscale LeNet CNN
# ----------------------------
grayscale_model = models.Sequential([
    layers.InputLayer(input_shape=(H, W, 1)),
    layers.Conv2D(6, (5, 5), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (5, 5), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation="relu"),
    layers.Dense(84, activation="relu"),
    layers.Dense(5, activation="softmax")
])

grayscale_model.compile(
    optimizer=optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(grayscale_model.summary())

# ----------------------------
# 6) Train (Grayscale)
# ----------------------------
history_gray = grayscale_model.fit(
    X_train_norm,
    y_train_oh,
    batch_size=128,
    epochs=20,
    validation_data=(X_test_norm, y_test_oh),
    verbose=1
)

# ----------------------------
# 7) Evaluate (Grayscale)
# ----------------------------
test_loss, test_acc = grayscale_model.evaluate(X_test_norm, y_test_oh, verbose=1)
print(f"[Grayscale] Final Test Accuracy: {test_acc * 100:.2f}%")

# Confusion matrix
y_pred = grayscale_model.predict(X_test_norm).argmax(axis=1)
y_true = y_test_oh.argmax(axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Confusion Matrix (Grayscale CNN)")
plt.show()

# ----------------------------
# 8) SNR robustness analysis
# Assumption (as in notebook): the test set is ordered by SNR blocks.
# If your dataset isn't ordered this way, this will not be meaningful.
# ----------------------------
snr_levels = [-30, -25, -20, -15, -10, -5, 0, 5]
samples_per_snr = 300

def snr_accuracy(model, X, y_onehot, snr_levels, samples_per_snr):
    results = {}
    for i, snr in enumerate(snr_levels):
        start = i * samples_per_snr
        end = (i + 1) * samples_per_snr

        X_snr = X[start:end]
        y_snr = y_onehot[start:end]

        preds = model.predict(X_snr).argmax(axis=1)
        true  = y_snr.argmax(axis=1)

        acc = np.mean(preds == true) * 100
        results[snr] = acc
        print(f"SNR {snr:>3} dB: Accuracy = {acc:.2f}%")
    return results

snr_results = snr_accuracy(grayscale_model, X_test_norm, y_test_oh, snr_levels, samples_per_snr)

plt.figure()
plt.plot(list(snr_results.keys()), list(snr_results.values()), marker="o")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs SNR (Grayscale CNN)")
plt.grid(True)
plt.show()

# ============================================================
# Model 2: RGB (Jet colormap) Spectrogram + LeNet-style CNN
# ============================================================

def signature_to_rgb_image(sig_2d, out_size=(224, 224), cmap="jet"):
    """
    Convert a normalized 2D spectrogram (256x922) to jet-colored RGB image (224x224x3).
    """
    s = sig_2d.astype(np.float32)
    s_min, s_max = float(s.min()), float(s.max())
    denom = (s_max - s_min) if (s_max - s_min) != 0 else 1e-6
    s01 = (s - s_min) / denom  # [0,1]

    rgba = plt.get_cmap(cmap)(s01)          # (H,W,4) in [0,1]
    rgb = (rgba[..., :3] * 255).astype(np.uint8)

    img = Image.fromarray(rgb)
    img = img.resize(out_size, resample=Image.BILINEAR)

    return (np.asarray(img).astype(np.float32) / 255.0)

print("Converting spectrograms to RGB (jet)...")
X_train_rgb = np.array([signature_to_rgb_image(x) for x in X_train_norm_2d], dtype=np.float32)
X_test_rgb  = np.array([signature_to_rgb_image(x) for x in X_test_norm_2d], dtype=np.float32)

print("X_train_rgb:", X_train_rgb.shape)  # (N, 224,224,3)
print("X_test_rgb :", X_test_rgb.shape)

rgb_model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 3)),
    layers.Conv2D(6, (5, 5), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (5, 5), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation="relu"),
    layers.Dense(84, activation="relu"),
    layers.Dense(5, activation="softmax")
])

rgb_model.compile(
    optimizer=optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(rgb_model.summary())

history_rgb = rgb_model.fit(
    X_train_rgb,
    y_train_oh,
    batch_size=128,
    epochs=20,
    validation_data=(X_test_rgb, y_test_oh),
    verbose=1
)

test_loss_rgb, test_acc_rgb = rgb_model.evaluate(X_test_rgb, y_test_oh, verbose=1)
print(f"[RGB] Final Test Accuracy: {test_acc_rgb * 100:.2f}%")

# Confusion matrix (RGB)
y_pred_rgb = rgb_model.predict(X_test_rgb).argmax(axis=1)
cm_rgb = confusion_matrix(y_true, y_pred_rgb)
disp_rgb = ConfusionMatrixDisplay(cm_rgb)
disp_rgb.plot()
plt.title("Confusion Matrix (RGB CNN)")
plt.show()
