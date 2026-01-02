from data_loader import load_dataset
from preprocessing import shuffle_data, preprocess_data, encode_labels
from model import build_lenet
from evaluation import plot_confusion_matrix, snr_analysis

BASE_PATH = "/content/drive/MyDrive/Project Work/URC drones dataset v1"

# Load data
X_train, X_test, y_train, y_test = load_dataset(BASE_PATH)

# Shuffle & preprocess
X_train, y_train = shuffle_data(X_train, y_train)
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

y_train_oh, y_test_oh, encoder = encode_labels(y_train, y_test)

# Build & train model
model = build_lenet()
model.fit(
    X_train,
    y_train_oh,
    batch_size=128,
    epochs=20,
    validation_data=(X_test, y_test_oh),
    verbose=1
)

# Evaluation
plot_confusion_matrix(model, X_test, y_test_oh)

snr_levels = [-30, -25, -20, -15, -10, -5, 0, 5]
snr_analysis(model, X_test, y_test_oh, snr_levels, samples_per_snr=300)
