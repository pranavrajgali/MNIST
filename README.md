# MNIST Handwritten Digit Classifier

## Overview
This project trains a neural network to classify handwritten digits (0–9) using the **MNIST** dataset, which contains **60,000 training** and **10,000 test** grayscale images. Each image is **28×28** pixels.

---

## Features
- Loads and preprocesses MNIST by **normalizing pixel values to [0, 1]**
- **Flattens** 28×28 images into **784-dimensional vectors**
- Implements a **feedforward neural network** with the following architecture:
  - Dense (128) + ReLU
  - Dense (64) + Sigmoid
  - Dense (32) + Sigmoid
  - Output Dense (10) + Softmax
- Training setup:
  - **Optimizer:** Adam  
  - **Loss:** Sparse Categorical Cross-Entropy
- Trains for **10 epochs**, achieving ~**97.8%** test accuracy
- Evaluates performance using:
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-score)

---

## Requirements
- Python 3.x
- TensorFlow (Keras)
- NumPy
- Matplotlib
- scikit-learn
- seaborn

---

## Installation
Install dependencies with:

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
