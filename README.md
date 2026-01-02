# UAV RF Signal Classification Using Deep Learning

## 📌 Project Overview

This project implements an **RF-based UAV (Unmanned Aerial Vehicle) classification system** using **spectrogram analysis and deep learning**. Instead of relying on visual sensors, the system analyzes **radio-frequency (RF) transmission patterns** emitted by UAVs to identify and classify different UAV types.

The work presents a **complete signal-processing and machine learning pipeline**, from loading raw RF spectrogram data to training and evaluating Convolutional Neural Networks (CNNs). The approach is relevant to **airspace security, surveillance, and RF signal intelligence** applications.

---

## 🎯 Objectives

* Classify UAVs using RF signal characteristics
* Convert high-dimensional RF data into learnable representations
* Evaluate model robustness under varying noise conditions
* Demonstrate practical RF + deep learning integration

---

## 🧠 System Pipeline (High-Level)

1. Load flattened RF spectrograms from MATLAB `.mat` files
2. Reshape and normalize spectrogram data
3. Encode class labels for supervised learning
4. Train CNN-based classifiers (LeNet-style)
5. Evaluate accuracy, robustness, and class separation

---

## 📊 Dataset Description

* **Dataset:** URC Drones Dataset (v1)
* **Data Type:** RF signal spectrograms
* **Source Format:** MATLAB `.mat` files
* **Spectrogram Size:** `256 × 922` (frequency × time)
* **Classes:** 5 UAV classes
* **Samples:**

  * Training: ~5,600
  * Testing: ~2,400

Each sample represents a **log-domain RF spectrogram** capturing UAV transmission behavior over time and frequency.

---

## 🔬 Signal Preprocessing

To ensure stable learning and preserve RF characteristics, each spectrogram undergoes:

* Reshaping from flattened vectors to 2D matrices
* Column-wise mean subtraction (frequency-domain centering)
* Global standard deviation normalization
* Safeguards against numerical instability

This normalization improves convergence while retaining UAV-specific RF signatures.

---

## 🧠 Model Architectures

### Model 1: Grayscale Spectrogram CNN

* Input shape: `(256 × 922 × 1)`
* Two convolutional layers with ReLU activation
* Max-pooling for spatial downsampling
* Fully connected layers for classification
* Softmax output for 5 UAV classes

### Model 2: RGB Spectrogram CNN

* Spectrograms converted to **jet-colored RGB images**
* Resized to `(224 × 224 × 3)`
* LeNet-style CNN architecture
* Used to study the effect of color-encoded features

---

## 📈 Evaluation & Results

* **Test Accuracy (Grayscale CNN):** ~71.8%
* **Evaluation Metrics:**

  * Overall classification accuracy
  * Confusion matrix (class-level performance)
  * SNR-based robustness analysis

### SNR Robustness

Model accuracy is evaluated across multiple **Signal-to-Noise Ratio (SNR)** levels ranging from **-30 dB to +5 dB**. Results show:

* Near-perfect accuracy at higher SNR levels
* Gradual degradation under extreme noise
* Strong resilience to RF interference

---

## 🛠️ Technologies & Tools

* **Languages:** Python, MATLAB
* **Libraries:** NumPy, SciPy, TensorFlow/Keras, Scikit-learn, Matplotlib
* **Techniques:** Spectrogram analysis, CNNs, RF normalization, SNR evaluation

---

## 🚀 Key Takeaways

* Demonstrates non-visual UAV detection using RF signals
* Integrates signal processing with deep learning
* Uses real-world, high-dimensional RF data
* Includes robustness testing beyond standard accuracy metrics

---

## 👤 Author

**Mentor : Mohammed Rashid**
**Ahmed Abbasi B.S. MIS**
**Ubaid-ur-Rahman B.S. CIT**

---

## 📄 License

This project is intended for academic and research purposes. Please cite appropriately if reused or extended.
