# MNIST Handwritten Digit Classifier

A deep learning project that recognises handwritten digits (0-9) using two different neural network models, built from scratch as my first AI project.

---

## What This Project Does

This project trains an AI model to look at an image of a handwritten digit and correctly identify which number it is (0 through 9). It was built using the famous MNIST dataset and compares two different neural network approaches.

---

## Dataset

**MNIST (Modified National Institute of Standards and Technology)**

- Created by Yann LeCun in 1998
- 70,000 handwritten digit images total
- 60,000 images for training
- 10,000 images for testing
- Each image is 28×28 pixels in grayscale
- 10 classes: digits 0 through 9
- Loaded directly through TensorFlow (no download needed)

---

## Models Built

### Model 1 — MLP (Multi Layer Perceptron)
A basic neural network that looks at each pixel individually.

```
Architecture:
Input Layer   →  784 neurons  (one per pixel)
Hidden Layer 1 → 128 neurons  (ReLU activation)
Hidden Layer 2 →  64 neurons  (ReLU activation)
Output Layer  →  10 neurons   (one per digit)

Total parameters: ~109,000
```

**Result: 97.69% accuracy on 10,000 unseen test images**

---

### Model 2 — CNN (Convolutional Neural Network)
A smarter neural network that looks at groups of pixels together to find patterns like edges and curves.

```
Architecture:
Input             →  (28, 28, 1)
Conv2D (32 filters, 3×3 kernel)  →  finds simple patterns
MaxPooling2D (2×2)               →  shrinks image
Conv2D (64 filters, 3×3 kernel)  →  finds complex patterns
MaxPooling2D (2×2)               →  shrinks again
Flatten                          →  converts to 1D
Dense (64 neurons, ReLU)         →  fully connected
Dense (10 neurons, Softmax)      →  output layer

Total parameters: 121,930
```

**Result: 98.77% accuracy on 10,000 unseen test images**

---

## Results Comparison

| Model | Accuracy | Parameters | Epochs |
|-------|----------|------------|--------|
| MLP   | 97.69%   | ~109,000   | 20     |
| CNN   | 98.77%   | 121,930    | 5      |
| MNIST Benchmark (best ever) | 99.70% | — | — |

---

## How to Run

### Requirements
```
Python 3.11+
tensorflow
numpy
matplotlib
scikit-learn
```

### Install dependencies
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### Train the model (run once only)
```bash
python3 train_model.py
```

### Test the model
```bash
python3 testModel.py
```

### Test on your own handwritten digit
```bash
python3 test_digit.py
```

---



## Key Concepts Learned

- **What MNIST is** — 70,000 handwritten digit images created by Yann LeCun in 1998
- **X_train vs y_train** — images vs correct labels, the two things a model needs to learn
- **Normalisation** — scaling pixels from 0-255 to 0.0-1.0 so the model learns faster
- **Flattening** — converting 28×28 grid to 784 numbers for MLP
- **Conv2D** — filter windows that slide across image finding patterns (kernel)
- **MaxPooling** — shrinks feature maps keeping only strongest patterns
- **ReLU activation** — outputs 0 if negative, keeps value if positive
- **Softmax activation** — converts output scores to probabilities adding up to 100%
- **Loss function** — measures how wrong the model is (should decrease during training)
- **Backpropagation** — how the model fixes its mistakes by adjusting weights
- **Weights** — 121,930 numbers tuned during training that encode what digits look like
- **Benchmark** — MNIST best accuracy is 99.70% using advanced CNNs
- **Overfitting** — when model memorises training data but fails on new data
- **Random seed** — fixes randomness so results are reproducible




## Author
**Dipen Thapa**
Computer Science Student
First AI project — built from scratch learning every concept step by step
