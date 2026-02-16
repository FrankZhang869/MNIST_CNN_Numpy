# CNN From Scratch Using NumPy – MNIST Digit Classifier

This repository contains a full implementation of a **Convolutional Neural Network (CNN)** built entirely from scratch using only **NumPy**.  
No deep learning frameworks (PyTorch, TensorFlow, Keras, etc.) were used.

The model is trained and evaluated on the MNIST handwritten digit dataset.

---

## Overview

The purpose of this project was to:

- Build a CNN without relying on deep learning libraries
- Implement forward propagation manually
- Implement full backpropagation through convolutional and pooling layers
- Understand the computational mechanics behind modern CNN frameworks

This project prioritizes **educational clarity and mathematical transparency** over computational efficiency.

---

## Dataset

The model was trained on the MNIST dataset:

- 60,000 training images
- 10,000 test images
- 28×28 grayscale handwritten digits (0–9)

Each image is flattened or processed as a 2D matrix depending on the layer.

---

## 🏗 Architecture

The network includes:

- Convolution layer(s)
- ReLU activation
- Pooling layer(s)
- Fully connected (dense) layer(s)
- Softmax output layer
- Cross-entropy loss

All components were implemented manually, including:

- Convolution operations
- Backpropagation through convolution
- Gradient computation
- Weight updates
- Pooling forward/backward passes
- Softmax + cross-entropy derivatives

No automatic differentiation was used.

---

## Performance Considerations

Because convolutions are implemented directly using NumPy (without optimized C/CUDA kernels), the model is **computationally slow** compared to framework-based implementations.

In particular:

- Convolution operations are computed explicitly
- Backpropagation through convolution is especially expensive
- No GPU acceleration is used
- No highly optimized im2col or vectorized convolution tricks are implemented

### Testing Workaround

To make evaluation feasible, testing was performed on a **condensed subset of the MNIST test set (~100 samples)** instead of the full 10,000 images.

This allows:

- Faster experimentation
- Practical debugging
- Verification of correctness without excessive runtime

---

## Key Takeaways

- Deep understanding of convolution at the matrix level
- Manual derivation and implementation of CNN backpropagation
- Insight into why modern frameworks rely on highly optimized low-level operations
- Appreciation for the computational cost of naive convolution implementations
