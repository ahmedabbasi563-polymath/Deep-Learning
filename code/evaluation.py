import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

def snr_analysis(model, X_test, y_test, snr_levels, samples_per_snr):
    results = {}

    for i, snr in enumerate(snr_levels):
        start = i * samples_per_snr
        end = (i + 1) * samples_per_snr

        X_snr = X_test[start:end]
        y_snr = y_test[start:end]

        preds = model.predict(X_snr).argmax(axis=1)
        true = y_snr.argmax(axis=1)

        acc = np.mean(preds == true) * 100
        results[snr] = acc
        print(f"SNR {snr} dB → Accuracy: {acc:.2f}%")

    plt.plot(results.keys(), results.values(), marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy vs SNR")
    plt.grid(True)
    plt.show()

    return results
