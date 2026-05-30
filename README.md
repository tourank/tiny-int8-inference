# Tiny INT8 Inference

A small from-scratch C++ inference runtime for a PyTorch-trained MNIST MLP, with FP32 and INT8 inference paths.

The goal of this project is to understand what happens below a framework like PyTorch during deployment: tensor export, raw binary loading, manual matrix-vector kernels, activation functions, quantization, integer accumulation, and accuracy/performance tradeoffs.

## Overview

The model is a tiny MLP trained on MNIST:

```text
28x28 image
→ flatten to 784
→ Linear(784, 128)
→ ReLU
→ Linear(128, 10)
→ argmax
```

Training happens in Python/PyTorch. Inference happens in C++ without PyTorch or external ML libraries.

Big picture:

```text
PyTorch training
    ↓
Export raw weights/images
    ↓
C++ loads binary tensors
    ↓
Manual FP32 / INT8 inference
    ↓
Accuracy and timing benchmark
```

## Results

```text
Compiler: g++ -std=c++17 -O2
Machine: MacBook Air
Test set: 10,000 MNIST images
```

| Inference path                         | Accuracy |  Total time | ms / image | Images / sec |
| -------------------------------------- | -------: | ----------: | ---------: | -----------: |
| FP32 weights + FP32 matmul             |   0.9696 |  0.416392 s |  0.0416392 |     24,015.8 |
| INT8 weights dequantized + FP32 matmul |   0.9693 |  0.405726 s |  0.0405726 |     24,647.2 |
| INT8 fc1 + float fc2                   |   0.9693 | 0.0453072 s | 0.00453072 |      220,716 |
| INT8 fc1 + INT8 fc2                    |   0.9695 |  0.051736 s |  0.0051736 |      193,289 |

In other words, the INT8 paths preserve accuracy almost exactly. The hybrid INT8-fc1 path is fastest in this simple benchmark because `fc1` dominates the computation, while quantizing the hidden activation before the much smaller `fc2` layer adds overhead.

