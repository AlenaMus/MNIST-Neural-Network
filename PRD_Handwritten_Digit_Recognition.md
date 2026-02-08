# Product Requirements Document (PRD)
# Handwritten Digit Recognition Neural Network

**Version:** 1.0
**Date:** February 8, 2026
**Status:** Draft
**Owner:** AI Development Course - Lesson 36

---

## Executive Summary

This document outlines the requirements for building an educational neural network program that recognizes and classifies handwritten digits (0-9) using the MNIST dataset. The program serves as a learning tool for students to understand fundamental concepts of deep learning, including data preprocessing, neural network architecture, model training, and evaluation.

**Key Objective:** Create a well-documented, beginner-friendly Python implementation of a digit classification neural network that achieves >95% accuracy on the MNIST test set.

---

## Table of Contents

1. [Product Overview](#product-overview)
2. [Problem Statement](#problem-statement)
3. [Goals and Success Metrics](#goals-and-success-metrics)
4. [User Stories](#user-stories)
5. [Functional Requirements](#functional-requirements)
6. [Technical Specifications](#technical-specifications)
7. [Non-Functional Requirements](#non-functional-requirements)
8. [Implementation Guidelines](#implementation-guidelines)
9. [Acceptance Criteria](#acceptance-criteria)
10. [Launch Plan](#launch-plan)
11. [Future Enhancements](#future-enhancements)

---

## 1. Product Overview

### 1.1 Vision
Provide an accessible, educational implementation of a neural network that demonstrates core machine learning concepts through the practical application of handwritten digit recognition.

### 1.2 Target Audience

**Primary Persona: Learning Laura**
- **Role:** Student/Beginner in Machine Learning
- **Goals:** Understand neural network fundamentals through hands-on coding
- **Pain Points:** Complex ML tutorials lack clear explanations; hard to connect theory to practice
- **Needs:** Step-by-step code with educational comments explaining WHY, not just WHAT

**Secondary Persona: Teaching Tom**
- **Role:** Instructor/Educator
- **Goals:** Provide students with a clean, working example of neural networks
- **Needs:** Well-documented code that can be used as teaching material

### 1.3 Value Proposition

For students learning neural networks who need practical, understandable examples, this program is an educational implementation that provides clear, commented code explaining each step. Unlike complex production systems or overly simplified tutorials, our solution balances technical accuracy with educational clarity.

---

## 2. Problem Statement

### 2.1 User Problem
Beginners learning neural networks struggle to bridge the gap between theoretical concepts and practical implementation. Existing code examples often lack sufficient explanation or are too complex for first-time learners.

### 2.2 Business Problem
Create a reusable, maintainable educational asset that can serve as a foundation for teaching neural network concepts in an AI development course.

### 2.3 Why Now?
Neural networks and deep learning are fundamental skills in modern AI development. The MNIST dataset is an industry-standard benchmark that provides immediate, verifiable results, making it ideal for educational purposes.

---

## 3. Goals and Success Metrics

### 3.1 Product Goals

**Primary Goals:**
1. Build a functional neural network that classifies handwritten digits with >95% test accuracy
2. Provide comprehensive code documentation that explains each step educationally
3. Demonstrate best practices in data preprocessing, model building, and evaluation

**Secondary Goals:**
1. Serve as a template for future neural network projects
2. Enable easy modification and experimentation by learners
3. Execute in reasonable time (<1 minute training on GPU)

### 3.2 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Model Accuracy (Test)** | >95% | Model evaluation on test set |
| **Code Documentation Coverage** | 100% of major steps | Each functional block has explanatory comments |
| **Training Time** | <1 minute | Measured on GPU |
| **Code Clarity** | Understandable by beginners | Peer review by target audience |
| **Reproducibility** | 100% | Same results when run multiple times |

### 3.3 North Star Metric
**Learning Effectiveness:** Percentage of users who can successfully explain what each code section does after reviewing the implementation (Target: >90%)

---

## 4. User Stories

### 4.1 Core User Stories

**Story 1: Data Loading and Understanding**
```
As a student learning neural networks,
I want to load the MNIST dataset and understand its structure,
So that I know what data I'm working with and how it's organized.

Acceptance Criteria:
- Dataset loads successfully using Keras built-in loader
- Code comments explain dataset dimensions (60K training, 10K test)
- Code comments explain image format (28x28 grayscale)
- Code displays sample statistics (data shape, value ranges)
```

**Story 2: Data Preprocessing**
```
As a student learning neural networks,
I want to preprocess the data with clear explanations,
So that I understand why normalization, flattening, and encoding are necessary.

Acceptance Criteria:
- Pixel values normalized from [0-255] to [0-1] with explanation
- Images flattened from 28x28 to 784-element vectors with explanation
- Labels converted to one-hot encoding with explanation
- Code comments explain the "why" behind each preprocessing step
```

**Story 3: Model Building**
```
As a student learning neural networks,
I want to build a neural network architecture with documented design choices,
So that I understand how layers, neurons, and activation functions work together.

Acceptance Criteria:
- Input layer accepts 784-element vectors
- Hidden layers use ReLU activation (explained in comments)
- Output layer has 10 neurons with softmax activation (explained)
- Each layer's purpose is clearly documented
```

**Story 4: Model Training**
```
As a student learning neural networks,
I want to train the model with clear progress tracking,
So that I can see the learning process and understand training parameters.

Acceptance Criteria:
- Model compiled with optimizer, loss function, and metrics
- Training shows epoch-by-epoch progress
- Code explains choice of optimizer (Adam), loss (categorical crossentropy)
- Batch size and epochs are documented with rationale
```

**Story 5: Model Evaluation**
```
As a student learning neural networks,
I want to evaluate the model's performance on unseen data,
So that I understand how well the model generalizes.

Acceptance Criteria:
- Model evaluated on test set (not training set)
- Accuracy and loss reported clearly
- Code explains difference between training and test performance
- Results are clearly displayed and interpretable
```

---

## 5. Functional Requirements

### 5.0 Data Visualization (FR-0)

**FR-0.1: Preview Sample Images**
- **Requirement:** Display sample images (digits 6-9) from the dataset in a single figure
- **Method:** Use matplotlib to create a 2x2 or 1x4 grid subplot
- **Output:** Visual display of 4 images showing digits 6, 7, 8, and 9
- **Purpose:** Allow users to visually inspect the dataset before training
- **Documentation:** Comment explaining the visualization purpose and matplotlib usage

```python
# Example visualization layout:
# ┌─────────┬─────────┐
# │  Six    │  Seven  │
# │   6     │    7    │
# ├─────────┼─────────┤
# │  Eight  │  Nine   │
# │   8     │    9    │
# └─────────┴─────────┘
```

**FR-0.2: Image Display Configuration**
- **Requirement:** Configure proper display settings for grayscale images
- **Settings:**
  - Colormap: 'gray' for grayscale display
  - Title: Show the digit label for each image
  - Axis: Hide axis for cleaner visualization
- **Documentation:** Comment explaining display parameters

### 5.0.1 GPU Configuration (FR-0.1)

**FR-0.1: GPU Detection**
- **Requirement:** Verify GPU availability at program start
- **Method:** Use `tf.config.list_physical_devices('GPU')`
- **Output:** Print GPU device name if available, warning if not
- **Documentation:** Comment explaining GPU acceleration benefits

**FR-0.2: GPU Memory Management (Optional)**
- **Requirement:** Configure GPU memory growth to prevent OOM errors
- **Method:** Use `tf.config.experimental.set_memory_growth()`
- **Rationale:** Prevents TensorFlow from allocating all GPU memory at once
- **Documentation:** Comment explaining memory management strategy

### 5.1 Data Loading (FR-1)

**FR-1.1: MNIST Dataset Import**
- **Requirement:** Load MNIST dataset using `tensorflow.keras.datasets.mnist.load_data()`
- **Input:** None (automatic download)
- **Output:** Training set (X_train, y_train) and test set (X_test, y_test)
- **Validation:** Verify shapes - X_train: (60000, 28, 28), y_train: (60000,)

**FR-1.2: Dataset Documentation**
- **Requirement:** Display dataset information
- **Output:** Print statements showing:
  - Number of training samples
  - Number of test samples
  - Image dimensions
  - Sample pixel value range

### 5.2 Data Preprocessing (FR-2)

**FR-2.1: Normalization**
- **Requirement:** Scale pixel values from [0, 255] to [0, 1]
- **Method:** Divide all pixel values by 255.0
- **Rationale:** Neural networks train better with normalized inputs
- **Documentation:** Comment explaining why normalization improves training

**FR-2.2: Flattening**
- **Requirement:** Convert 28x28 2D images to 784-element 1D vectors
- **Method:** Reshape from (num_samples, 28, 28) to (num_samples, 784)
- **Rationale:** Fully connected networks require 1D input
- **Documentation:** Comment explaining the reshape operation

**FR-2.3: Label Encoding**
- **Requirement:** Convert integer labels (0-9) to one-hot encoded vectors
- **Method:** Use `tensorflow.keras.utils.to_categorical()`
- **Output:** 10-dimensional vectors (e.g., 3 → [0,0,0,1,0,0,0,0,0,0])
- **Documentation:** Comment explaining one-hot encoding purpose

### 5.3 Neural Network Architecture (FR-3)

**FR-3.1: Model Structure**
- **Requirement:** Sequential model with fully connected (Dense) layers
- **Architecture Specification:**
  ```
  Input Layer:  784 neurons (28×28 flattened)
  Hidden Layer 1: 128 neurons, ReLU activation
  Hidden Layer 2: 64 neurons, ReLU activation
  Output Layer: 10 neurons, Softmax activation
  ```

**FR-3.2: Layer Documentation**
- **Requirement:** Each layer must have comments explaining:
  - Number of neurons and why
  - Activation function choice
  - Layer's role in the network

**FR-3.3: Model Summary**
- **Requirement:** Display model architecture summary
- **Output:** `model.summary()` showing layers, parameters, and shape

### 5.4 Model Compilation (FR-4)

**FR-4.1: Loss Function - Categorical Crossentropy**
- **Requirement:** Use categorical crossentropy loss function
- **Rationale:** Standard and optimal choice for multi-class classification problems
- **Documentation:** Comprehensive comment explaining loss function theory

**Loss Function Explanation (Must be included in code comments):**

```
CATEGORICAL CROSSENTROPY LOSS FUNCTION
======================================

What is a Loss Function?
------------------------
A loss function measures how "wrong" our model's predictions are compared to
the actual labels. The goal of training is to MINIMIZE this loss value.

Why Categorical Crossentropy?
-----------------------------
1. MULTI-CLASS CLASSIFICATION: We have 10 classes (digits 0-9), not just 2
2. PROBABILITY OUTPUT: Works with softmax output (probabilities)
3. PENALIZES CONFIDENCE: Heavily penalizes confident wrong predictions
4. MATHEMATICAL FOUNDATION: Based on information theory (entropy)

Mathematical Formula:
--------------------
Loss = -Σ(y_true × log(y_pred))

Where:
- y_true = actual label (one-hot encoded, e.g., [0,0,0,1,0,0,0,0,0,0] for digit 3)
- y_pred = predicted probabilities from softmax
- log = natural logarithm
- Σ = sum over all 10 classes

Example Calculation:
-------------------
True label: 3 → one-hot: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Predicted:              [0.01, 0.02, 0.05, 0.85, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]

Loss = -(0×log(0.01) + 0×log(0.02) + 0×log(0.05) + 1×log(0.85) + ...)
     = -log(0.85)
     = 0.163

Good prediction (0.85 for correct class) = Low loss (0.163)

If prediction was wrong:
Predicted:              [0.85, 0.02, 0.05, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]
Loss = -log(0.01) = 4.605

Bad prediction (0.01 for correct class) = High loss (4.605)

Why Not Other Loss Functions?
-----------------------------
- MSE (Mean Squared Error): Works but converges slower for classification
- Binary Crossentropy: Only for 2-class problems
- Sparse Categorical Crossentropy: Alternative if labels aren't one-hot encoded
```

**FR-4.2: Optimizer - Adam**
- **Requirement:** Use Adam (Adaptive Moment Estimation) optimizer
- **Rationale:** Best general-purpose optimizer for deep learning
- **Documentation:** Comprehensive comment explaining optimizer choice

**Optimizer Explanation (Must be included in code comments):**

```
ADAM OPTIMIZER
==============

What is an Optimizer?
---------------------
An optimizer is the algorithm that adjusts the neural network's weights
during training to minimize the loss function. It determines HOW the
network learns from its mistakes.

Why Adam? (Adaptive Moment Estimation)
--------------------------------------
Adam combines the best properties of two other optimizers:
1. Momentum: Remembers past gradients to smooth out updates
2. RMSprop: Adapts learning rate for each parameter individually

Key Advantages of Adam:
-----------------------
1. ADAPTIVE LEARNING RATE: Automatically adjusts learning rate per parameter
   - Parameters that need big updates get bigger learning rates
   - Parameters that need small updates get smaller learning rates

2. MOMENTUM: Uses moving averages of past gradients
   - Helps escape local minima
   - Smooths out noisy gradients
   - Accelerates convergence in consistent directions

3. WORKS OUT OF THE BOX: Default hyperparameters work well for most problems
   - learning_rate = 0.001
   - beta_1 = 0.9 (momentum term)
   - beta_2 = 0.999 (scaling term)
   - epsilon = 1e-7 (prevents division by zero)

4. MEMORY EFFICIENT: Only stores 2 moving averages per parameter

5. FAST CONVERGENCE: Typically converges faster than SGD or RMSprop

Mathematical Intuition:
----------------------
Standard Gradient Descent: w = w - learning_rate × gradient
Adam:                      w = w - learning_rate × (momentum_adjusted_gradient / scale)

Where momentum and scale adapt based on the history of gradients.

Why Not Other Optimizers?
-------------------------
- SGD (Stochastic Gradient Descent): Works but slower, requires careful tuning
- RMSprop: Good but Adam usually performs equally or better
- Adagrad: Learning rate can become too small over time
- SGD with Momentum: Good but Adam adapts learning rate automatically

For MNIST with our architecture, Adam typically achieves:
- 95%+ accuracy in 5-10 epochs
- Stable training without oscillation
- No need for learning rate scheduling
```

**FR-4.3: Metrics**
- **Requirement:** Track accuracy metric
- **Output:** Display accuracy during training and evaluation

### 5.5 Model Training (FR-5)

**FR-5.1: Training Parameters**
- **Epochs:** 10 (minimum)
- **Batch Size:** 32-128 (configurable)
- **Validation:** Optional validation split or use test set

**FR-5.2: Training Progress**
- **Requirement:** Display epoch-by-epoch progress
- **Output:** Loss and accuracy for each epoch
- **Documentation:** Comments explaining training process

**FR-5.3: Training History**
- **Requirement:** Store training history for analysis
- **Output:** History object containing metrics over time

### 5.6 Model Evaluation (FR-6)

**FR-6.1: Test Set Evaluation**
- **Requirement:** Evaluate model on test set
- **Output:** Test loss and test accuracy
- **Documentation:** Comment explaining train/test difference

**FR-6.2: Results Display**
- **Requirement:** Print final results clearly
- **Format:**
  ```
  Test Loss: 0.XXXX
  Test Accuracy: XX.XX%
  ```

**FR-6.3: Learning Curve Visualization (Loss vs Epochs)**
- **Requirement:** Plot training history showing loss function over epochs
- **Method:** Use matplotlib to create line plots from training history
- **Output:** Graph with two lines - training loss and validation loss
- **Purpose:** Visualize model learning progress and detect overfitting/underfitting
- **Documentation:** Comments explaining how to interpret the graph

```
Learning Curve Graph Requirements:
----------------------------------
- X-axis: Epochs (1 to N)
- Y-axis: Loss value
- Lines:
  * Training Loss (blue solid line)
  * Validation Loss (orange dashed line)
- Title: "Model Loss Over Training Epochs"
- Legend: Clearly labeled lines
- Grid: Enable for easier reading

Example Expected Output:
                    Model Loss Over Training Epochs
    Loss │
    0.5  │ ●
         │   ●
    0.4  │     ●
         │       ●─────────────── Validation Loss
    0.3  │         ●
         │           ●
    0.2  │             ●
         │               ●
    0.1  │                 ●───── Training Loss
         │                   ● ● ●
    0.0  └─────────────────────────────────────
         1   2   3   4   5   6   7   8   9   10
                        Epochs

Interpretation Guide (include in comments):
- Both lines decreasing: Model is learning well
- Training loss << Validation loss: Overfitting (model memorizes training data)
- Both lines high and flat: Underfitting (model not learning enough)
- Lines converging: Good generalization
```

**FR-6.4: Accuracy Curve Visualization**
- **Requirement:** Plot training history showing accuracy over epochs
- **Method:** Similar to loss graph, but for accuracy metric
- **Output:** Graph with training accuracy and validation accuracy
- **Purpose:** Complementary view to loss graph

```
Accuracy Graph Requirements:
----------------------------
- X-axis: Epochs
- Y-axis: Accuracy (0.0 to 1.0 or 0% to 100%)
- Lines:
  * Training Accuracy (blue solid line)
  * Validation Accuracy (orange dashed line)
- Title: "Model Accuracy Over Training Epochs"
```

**FR-6.5: Confusion Matrix (10×10)**
- **Requirement:** Generate and display a 10×10 confusion matrix after testing
- **Method:** Use sklearn.metrics.confusion_matrix or manual calculation
- **Input:** Model predictions on test set (10,000 samples)
- **Output:** Heatmap visualization showing prediction distribution
- **Purpose:** Identify which digits the model confuses with each other
- **Documentation:** Comments explaining how to read the confusion matrix

```
Confusion Matrix Specification:
-------------------------------
Dimensions: 10 rows × 10 columns
- Rows: Actual/True labels (0-9)
- Columns: Predicted labels (0-9)
- Cells: Count of predictions for each (actual, predicted) pair
- Diagonal: Correct predictions (should be highest values)
- Off-diagonal: Misclassifications (errors)

Example 10×10 Confusion Matrix:
                        PREDICTED
              0    1    2    3    4    5    6    7    8    9
         ┌────────────────────────────────────────────────────┐
       0 │ 973    0    1    0    0    1    2    1    2    0   │
       1 │   0 1127    2    1    0    1    2    0    2    0   │
       2 │   2    2 1015    3    1    0    1    4    4    0   │
A      3 │   0    0    3  995    0    3    0    3    4    2   │
C      4 │   1    0    2    0  964    0    4    1    2    8   │
T      5 │   2    0    0    6    1  876    3    1    2    1   │
U      6 │   4    2    0    0    2    2  946    0    2    0   │
A      7 │   1    3    5    2    0    0    0 1010    2    5   │
L      8 │   2    0    3    2    3    3    2    2  954    3   │
       9 │   3    3    1    3    8    3    0    4    3  981   │
         └────────────────────────────────────────────────────┘

Reading the Matrix:
- Diagonal values (973, 1127, 1015, ...) = correct predictions
- Row sum = total actual samples for that digit
- Column sum = total predictions for that digit
- Cell (3,8) = 4 means: 4 times a "3" was predicted as "8"

Common Confusions to Look For:
- 4 ↔ 9 (similar top parts)
- 3 ↔ 8 (similar curves)
- 5 ↔ 6 (similar shapes)
- 7 ↔ 1 (similar strokes)
```

**FR-6.6: Confusion Matrix Visualization**
- **Requirement:** Display confusion matrix as a color-coded heatmap
- **Method:** Use seaborn heatmap or matplotlib imshow
- **Settings:**
  - Color map: Blues or viridis (darker = higher count)
  - Annotations: Show numbers in each cell
  - Labels: 0-9 on both axes
  - Title: "Confusion Matrix - MNIST Test Set"
- **Output:** Visual heatmap with clear color gradient

### 5.7 Custom Image Prediction (FR-7)

**FR-7.1: Load External Image**
- **Requirement:** Function to load a new, unknown handwritten digit image from file
- **Supported Formats:** PNG, JPG, JPEG, BMP
- **Input:** File path to the image
- **Output:** Preprocessed image ready for prediction
- **Documentation:** Comments explaining image loading and preprocessing steps

**FR-7.2: Image Preprocessing for Prediction**
- **Requirement:** Preprocess external image to match MNIST format
- **Steps:**
  1. Load image from file path
  2. Convert to grayscale (if RGB)
  3. Resize to 28×28 pixels
  4. Invert colors if needed (MNIST has white digits on black background)
  5. Normalize pixel values to [0, 1]
  6. Flatten to 784-element vector
  7. Reshape for model input (1, 784) - batch dimension
- **Documentation:** Each preprocessing step must be commented

```
Image Preprocessing Pipeline:
-----------------------------
Original Image (any size, any format)
        │
        ▼
    Load Image ──────────── Using PIL/Pillow or cv2
        │
        ▼
Convert to Grayscale ───── If image has color channels
        │
        ▼
    Resize to 28×28 ────── Match MNIST dimensions
        │
        ▼
  Invert Colors ────────── If white background (MNIST is black bg)
        │
        ▼
    Normalize ──────────── Divide by 255.0
        │
        ▼
    Flatten ────────────── Reshape to (784,)
        │
        ▼
  Add Batch Dimension ──── Reshape to (1, 784)
        │
        ▼
  Ready for Prediction
```

**FR-7.3: Prediction Function**
- **Requirement:** Function to predict digit from preprocessed image
- **Input:** Preprocessed image array (1, 784)
- **Output:**
  - Predicted digit (0-9)
  - Confidence score (probability)
  - All class probabilities (for visualization)
- **Method:** Use `model.predict()` on preprocessed image
- **Documentation:** Comments explaining prediction output interpretation

**FR-7.4: Prediction Visualization**
- **Requirement:** Display the input image alongside prediction results
- **Output:** Figure showing:
  - The input image (28×28)
  - Predicted digit label
  - Confidence percentage
  - Optional: Bar chart of all class probabilities
- **Purpose:** User-friendly display of prediction results

```
Example Prediction Output:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│    ┌────────────┐        Prediction Results              │
│    │            │        ─────────────────               │
│    │    ████    │        Predicted Digit: 7              │
│    │      ██    │        Confidence: 98.5%               │
│    │      ██    │                                        │
│    │      ██    │        Class Probabilities:            │
│    │            │        0: 0.1%  5: 0.2%                │
│    └────────────┘        1: 0.3%  6: 0.1%                │
│     Input Image          2: 0.2%  7: 98.5%  ← Predicted  │
│                          3: 0.1%  8: 0.3%                │
│                          4: 0.2%  9: 0.1%                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**FR-7.5: Error Handling for Custom Images**
- **Requirement:** Handle common errors gracefully
- **Error Cases:**
  - File not found
  - Unsupported image format
  - Corrupted image file
  - Image too small to resize
- **Output:** Clear error messages guiding user to fix the issue
- **Documentation:** Comments explaining error handling

**FR-7.6: Interactive Prediction Loop (Optional)**
- **Requirement:** Allow user to predict multiple images in sequence
- **Method:** Loop that prompts for file path, makes prediction, asks to continue
- **Exit Condition:** User enters 'q' or 'quit' to exit loop
- **Purpose:** Convenient testing of multiple custom images

```python
# Example usage flow:
# Enter image path (or 'q' to quit): my_digit.png
# Prediction: 7 (Confidence: 98.5%)
# Enter image path (or 'q' to quit): another_digit.jpg
# Prediction: 3 (Confidence: 95.2%)
# Enter image path (or 'q' to quit): q
# Exiting prediction mode.
```

### 5.8 Code Documentation (FR-8)

**FR-7.1: Section Headers**
- **Requirement:** Clear section markers for each major step
- **Format:**
  ```python
  # ============================================
  # SECTION: Data Loading
  # ============================================
  ```

**FR-7.2: Inline Comments**
- **Requirement:** Educational comments before each code block
- **Style:** Explain WHY and WHAT, not just WHAT
- **Audience:** Assume reader knows basic Python but not ML

**FR-7.3: Parameter Explanations**
- **Requirement:** Document all important parameters
- **Examples:** epochs, batch_size, neurons, learning_rate

---

## 6. Technical Specifications

### 6.1 Technology Stack

**Programming Language:**
- Python 3.8 or higher

**Required Libraries:**
```python
tensorflow >= 2.10.0
keras (included in TensorFlow)
numpy >= 1.21.0
matplotlib >= 3.4.0 (optional, for visualization)
```

**Development Environment:**
- Any Python IDE or text editor
- Jupyter Notebook (optional)
- Google Colab (optional)

**GPU Requirements:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.x or 12.x
- cuDNN library (compatible with CUDA version)
- TensorFlow-GPU (included in tensorflow>=2.10.0)
- Minimum GPU Memory: 2 GB (recommended: 4+ GB)

### 6.2 Dataset Specifications

**MNIST Dataset:**
- **Source:** `tensorflow.keras.datasets.mnist`
- **Training Set:** 60,000 images
- **Test Set:** 10,000 images
- **Image Format:** 28×28 grayscale (single channel)
- **Pixel Values:** Integers 0-255
- **Labels:** Integers 0-9 (10 classes)
- **File Size:** ~11 MB (automatic download)

### 6.3 Neural Network Architecture

**Model Type:** Sequential (Feedforward)

**Detailed Architecture:**
```
Layer 1 (Input):
  - Type: Flatten or Input + Dense
  - Units: 784 (28×28)
  - Activation: None (input layer)

Layer 2 (Hidden 1):
  - Type: Dense
  - Units: 128
  - Activation: ReLU
  - Purpose: Feature extraction

Layer 3 (Hidden 2):
  - Type: Dense
  - Units: 64
  - Activation: ReLU
  - Purpose: Higher-level feature combination

Layer 4 (Output):
  - Type: Dense
  - Units: 10
  - Activation: Softmax
  - Purpose: Classification probability distribution
```

**Total Parameters:** ~109,386 trainable parameters

### 6.3.1 Architecture Design Rationale

This section explains **WHY** we chose this specific neural network architecture. Understanding these design decisions is crucial for learning how to build neural networks for other problems.

#### Why Fully Connected (Dense) Network?

**Choice:** We use a fully connected (Dense) neural network instead of other architectures like CNN or RNN.

**Reasoning:**
- **Simplicity:** Fully connected networks are the most basic type of neural network, making them ideal for learning fundamentals
- **Sufficient for MNIST:** The MNIST dataset is relatively simple (clear digits, centered, uniform size), so a basic architecture achieves excellent results
- **Educational Value:** Understanding dense layers is prerequisite to understanding more complex architectures
- **Fast Training:** Dense networks train quickly, allowing rapid experimentation

#### Input Layer: 784 Neurons

**Choice:** Input layer accepts 784-element vectors.

**Reasoning:**
- **Mathematical Requirement:** Each input image is 28×28 pixels = 784 total pixel values
- **One Neuron Per Feature:** Each pixel is treated as an independent input feature
- **Flattening Necessity:** Dense layers require 1D input, so we reshape 2D images (28×28) to 1D vectors (784)
- **Information Preservation:** No information is lost during flattening; all pixel values are preserved

```
Original Image:     Flattened Vector:
[28 x 28 grid]  →   [784 values: pixel_0, pixel_1, ..., pixel_783]
```

#### Hidden Layer 1: 128 Neurons with ReLU

**Choice:** First hidden layer has 128 neurons with ReLU activation.

**Reasoning for 128 Neurons:**
- **Dimensionality Reduction:** We reduce from 784 inputs to 128, forcing the network to learn compressed representations
- **Sufficient Capacity:** 128 neurons provide enough capacity to learn complex patterns in digit shapes
- **Power of 2:** Using powers of 2 (128 = 2^7) is a convention that can optimize memory usage on modern hardware
- **Balance:** Not too few (underfitting) and not too many (overfitting, slow training)

**Reasoning for ReLU Activation:**
- **Non-linearity:** ReLU (Rectified Linear Unit) introduces non-linearity, allowing the network to learn complex patterns
- **Formula:** ReLU(x) = max(0, x) — outputs x if positive, 0 if negative
- **Computational Efficiency:** Extremely fast to compute compared to sigmoid or tanh
- **Gradient Flow:** Avoids the "vanishing gradient" problem that affects sigmoid/tanh in deep networks
- **Sparsity:** Outputs zero for negative inputs, creating sparse activations that can improve efficiency

```
ReLU Function:
        |      /
        |     /
        |    /
   0 ---|---/----
        |  0
   negative → 0, positive → same value
```

#### Hidden Layer 2: 64 Neurons with ReLU

**Choice:** Second hidden layer has 64 neurons with ReLU activation.

**Reasoning for 64 Neurons:**
- **Funnel Architecture:** We progressively reduce neurons (784 → 128 → 64 → 10), creating a "funnel" shape
- **Higher-Level Features:** This layer combines features from the first hidden layer into more abstract representations
- **Feature Hierarchy:** First layer might learn edges/lines; second layer learns combinations (curves, corners)
- **Compression:** Forces the network to identify the most important patterns for classification

**Why the Funnel Shape Works:**
```
784 inputs (raw pixels)
    ↓
128 neurons (low-level features: edges, strokes)
    ↓
64 neurons (high-level features: curves, loops, intersections)
    ↓
10 outputs (digit classes)
```

This progressive reduction mirrors how humans recognize digits: we don't look at individual pixels, but rather at shapes, curves, and patterns that combine to form recognizable digits.

#### Output Layer: 10 Neurons with Softmax

**Choice:** Output layer has exactly 10 neurons with Softmax activation.

**Reasoning for 10 Neurons:**
- **One Per Class:** We have 10 digit classes (0-9), so we need exactly 10 output neurons
- **Direct Mapping:** Each neuron corresponds to the probability of one digit class
- **Classification Output:** Neuron 0 → probability of digit "0", Neuron 1 → probability of digit "1", etc.

**Reasoning for Softmax Activation:**
- **Probability Distribution:** Softmax converts raw scores into probabilities that sum to 1.0
- **Mutual Exclusivity:** A digit can only be ONE class (3 cannot also be 7), so probabilities should compete
- **Interpretability:** Output values are directly interpretable as confidence levels
- **Formula:** Softmax(x_i) = exp(x_i) / Σexp(x_j)

```
Example Softmax Output:
Raw scores:    [2.1, 0.5, 8.2, 0.1, 0.3, 0.2, 0.4, 0.8, 0.1, 0.3]
After Softmax: [0.02, 0.01, 0.94, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
                                ↑
                    Predicted digit: 2 (94% confidence)
```

#### Why Not More/Fewer Layers?

**Why Not 1 Hidden Layer?**
- Single hidden layer can theoretically approximate any function, but may need many more neurons
- Two hidden layers allow hierarchical feature learning (low-level → high-level)
- Empirically, 2 hidden layers provide excellent accuracy on MNIST with fewer total parameters

**Why Not 3+ Hidden Layers?**
- Diminishing returns for this simple dataset
- Risk of overfitting on small datasets
- Longer training time without significant accuracy improvement
- For MNIST, 2 hidden layers typically achieve 97-98% accuracy; more layers rarely improve this

**Why Not More Neurons (e.g., 256, 512)?**
- More neurons = more parameters = higher risk of overfitting
- More neurons = longer training time
- 128-64 configuration is sufficient to capture digit patterns
- Can experiment with larger networks, but benefit is minimal for MNIST

**Why Not Fewer Neurons (e.g., 32, 16)?**
- Risk of underfitting: network may not have enough capacity to learn all digit patterns
- Some digits have subtle differences (3 vs 8, 4 vs 9) that require sufficient representational power
- 128-64 provides a good balance between capacity and efficiency

#### Alternative Architectures Considered

| Architecture | Parameters | Expected Accuracy | Trade-off |
|--------------|------------|-------------------|-----------|
| 784 → 64 → 10 | ~50K | ~96% | Simpler but less capacity |
| 784 → 128 → 64 → 10 | ~109K | ~97% | **Chosen: Good balance** |
| 784 → 256 → 128 → 10 | ~235K | ~97.5% | More parameters, minimal gain |
| 784 → 512 → 256 → 128 → 10 | ~530K | ~97.5% | Overkill for MNIST |

#### Summary: Architecture Design Principles

1. **Match Input to Data:** Input layer size = number of features (784 pixels)
2. **Match Output to Classes:** Output layer size = number of classes (10 digits)
3. **Use Funnel Shape:** Progressively reduce dimensions to force feature compression
4. **ReLU for Hidden Layers:** Fast, effective, avoids vanishing gradients
5. **Softmax for Classification:** Produces probability distribution over classes
6. **Start Simple:** Begin with minimal architecture, add complexity only if needed
7. **Powers of 2:** Common convention (32, 64, 128, 256) for potential hardware optimization

### 6.4 Training Configuration

**Optimizer:**
- **Type:** Adam
- **Learning Rate:** 0.001 (default)
- **Beta_1:** 0.9 (default)
- **Beta_2:** 0.999 (default)

**Loss Function:**
- **Type:** Categorical Crossentropy
- **Formula:** -Σ(y_true * log(y_pred))

**Metrics:**
- **Primary:** Accuracy
- **Optional:** Precision, Recall, F1-Score

**Training Parameters:**
- **Epochs:** 10-20
- **Batch Size:** 32, 64, or 128
- **Validation Split:** Optional 10-20% of training data
- **Shuffle:** True (shuffle training data each epoch)

### 6.5 Performance Requirements

**Accuracy Targets:**
- **Training Accuracy:** >98%
- **Test Accuracy:** >95%
- **Minimum Acceptable:** 93%

**Training Time:**
- **GPU (Primary):** <1 minute for 10 epochs
- **CPU (Fallback):** <5 minutes for 10 epochs

**Note:** This program is designed to run on GPU for optimal performance.

**Memory:**
- **Peak RAM Usage:** <2 GB
- **Model Size:** <1 MB (saved model)

### 6.6 File Structure

```
L-36-NumbersNeuralNetwork/
│
├── PRD_Handwritten_Digit_Recognition.md  (this document)
├── mnist_digit_recognition.py             (main implementation)
├── requirements.txt                        (dependencies)
├── README.md                              (quick start guide)
├── test_images/                           (folder for custom digit images)
│   ├── my_digit_1.png                     (example custom image)
│   ├── my_digit_2.jpg                     (example custom image)
│   └── ...
└── results/                               (generated outputs)
    ├── model_summary.txt                  (model architecture)
    ├── sample_digits_preview.png          (digits 6-9 visualization)
    ├── training_loss_curve.png            (loss vs epochs graph)
    ├── training_accuracy_curve.png        (accuracy vs epochs graph)
    ├── confusion_matrix.png               (10x10 confusion matrix heatmap)
    └── saved_model/                       (trained model for reuse)
```

---

## 7. Non-Functional Requirements

### 7.1 Usability (NFR-1)

**NFR-1.1: Code Readability**
- Code must follow PEP 8 style guidelines
- Variable names must be descriptive and meaningful
- Maximum line length: 100 characters
- Consistent indentation (4 spaces)

**NFR-1.2: Documentation Quality**
- Comments written in clear, simple English
- Avoid jargon without explanation
- Explain acronyms on first use (e.g., ReLU = Rectified Linear Unit)
- Comments targeted at beginner level

**NFR-1.3: Learning Curve**
- New learners should understand code flow within 30 minutes
- Each section should be independently understandable
- Logical progression from simple to complex

### 7.2 Reliability (NFR-2)

**NFR-2.1: Reproducibility**
- Set random seeds for reproducible results
- Same code should produce same results (±0.5% accuracy)
- Document any sources of randomness

**NFR-2.2: Error Handling**
- Graceful handling of missing dependencies
- Clear error messages if MNIST download fails
- Validation of data shapes at key steps

**NFR-2.3: Stability**
- No runtime errors under normal conditions
- Handles different hardware configurations (GPU primary, CPU fallback)
- GPU memory management prevents out-of-memory errors

### 7.3 Performance (NFR-3)

**NFR-3.1: Training Efficiency**
- Complete training in <1 minute on GPU (primary target)
- Support CPU fallback if GPU unavailable (<5 minutes)
- Optimize batch size for GPU memory

**NFR-3.2: Memory Efficiency**
- No memory leaks during training
- Efficient data loading (not loading all data at once unnecessarily)
- Clear memory after training if needed

### 7.4 Maintainability (NFR-4)

**NFR-4.1: Code Structure**
- Modular design with clear sections
- Easy to modify hyperparameters
- Simple to add additional layers or features

**NFR-4.2: Version Control**
- Code should be version control friendly
- No hardcoded absolute paths
- Configuration separated from logic (where appropriate)

### 7.5 Portability (NFR-5)

**NFR-5.1: Platform Independence**
- Runs on Windows, macOS, and Linux
- No OS-specific dependencies
- Compatible with Python 3.8+

**NFR-5.2: Environment Flexibility**
- Works in standard Python environments
- Compatible with Jupyter Notebooks
- Runnable on Google Colab

---

## 8. Implementation Guidelines

### 8.1 Development Phases

**Phase 1: Setup and Data Loading (Priority: High)**
- Install dependencies
- Import required libraries
- Load MNIST dataset
- Verify data shapes and contents
- **Deliverable:** Working data loading script with output verification

**Phase 2: Data Preprocessing (Priority: High)**
- Implement normalization
- Implement flattening/reshaping
- Implement one-hot encoding
- Verify preprocessing results
- **Deliverable:** Preprocessed data ready for model

**Phase 3: Model Architecture (Priority: High)**
- Define Sequential model
- Add input and hidden layers
- Add output layer
- Display model summary
- **Deliverable:** Compiled model ready for training

**Phase 4: Model Training (Priority: High)**
- Compile model with optimizer and loss
- Train model on training data
- Track training progress
- **Deliverable:** Trained model with history

**Phase 5: Model Evaluation (Priority: High)**
- Evaluate on test set
- Display results
- Verify accuracy target met
- **Deliverable:** Performance report

**Phase 6: Documentation and Polish (Priority: Medium)**
- Add comprehensive comments
- Create README
- Add usage examples
- **Deliverable:** Fully documented codebase

### 8.2 Code Structure Template

```python
# ============================================
# MNIST Handwritten Digit Recognition
# ============================================
# Description: [Brief project description]
# Author: [Your name]
# Date: [Date]

# ============================================
# SECTION 1: Import Required Libraries
# ============================================
# [Comments explaining each library's purpose]

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt           # For visualization (graphs, images)
import seaborn as sns                      # For confusion matrix heatmap
from sklearn.metrics import confusion_matrix  # For confusion matrix calculation
from PIL import Image                      # For loading custom images

# ============================================
# SECTION 1.1: GPU Configuration
# ============================================
# [Verify GPU availability and configure TensorFlow to use GPU]
# Print available GPUs to confirm GPU is detected

# ============================================
# SECTION 2: Load and Explore Dataset
# ============================================
# [Comments explaining MNIST dataset]

# Load the dataset
# [Explain what load_data() returns]

# Display dataset information
# [Explain the shapes and what they mean]

# ============================================
# SECTION 2.1: Preview Sample Images (Digits 6-9)
# ============================================
# [Explain why visualizing data is important]

# Find sample images for digits 6, 7, 8, 9
# [Explain how to search for specific labels]

# Display in 2x2 grid
# [Explain matplotlib subplot usage]

# ============================================
# SECTION 3: Data Preprocessing
# ============================================

# 3.1: Normalize pixel values
# [Explain why normalization is important]

# 3.2: Reshape images (flatten)
# [Explain why flattening is needed]

# 3.3: One-hot encode labels
# [Explain one-hot encoding concept]

# ============================================
# SECTION 4: Build Neural Network Model
# ============================================
# [Explain the overall architecture]

# Define the model architecture
# [Explain each layer choice]

# Display model summary
# [Explain what the summary shows]

# ============================================
# SECTION 5: Compile the Model
# ============================================
# [Explain compilation step]

# Compile with optimizer, loss, and metrics
# [Explain each parameter choice]

# ============================================
# SECTION 6: Train the Model
# ============================================
# [Explain training process]

# Train the model
# [Explain epochs, batch_size, and validation]

# ============================================
# SECTION 7: Evaluate the Model
# ============================================
# [Explain importance of test set evaluation]

# Evaluate on test data
# [Explain the metrics]

# Display final results
# [Show how to interpret results]

# ============================================
# SECTION 8: Visualize Training History
# ============================================
# [Explain what training history contains]

# 8.1: Plot Loss Curve
# [Explain loss vs epochs graph interpretation]

# 8.2: Plot Accuracy Curve
# [Explain accuracy vs epochs graph interpretation]

# ============================================
# SECTION 9: Confusion Matrix
# ============================================
# [Explain what a confusion matrix shows]

# Generate predictions on test set
# [Explain prediction process]

# Create confusion matrix
# [Explain matrix structure and interpretation]

# Visualize as heatmap
# [Explain color coding and reading the matrix]

# ============================================
# SECTION 10: Custom Image Prediction
# ============================================
# [Explain purpose of custom prediction]

# 10.1: Image preprocessing function
# [Explain each preprocessing step]

# 10.2: Prediction function
# [Explain how to use the trained model for new images]

# 10.3: Test with custom image
# [Demonstrate prediction on user-provided image]
```

### 8.3 Documentation Standards

**Comment Types:**

1. **Section Headers** (explain major sections):
```python
# ============================================
# SECTION: Data Preprocessing
# ============================================
# Prepare the data for neural network training by:
# 1. Normalizing pixel values to [0,1] range
# 2. Reshaping 2D images to 1D vectors
# 3. Converting labels to one-hot encoded format
```

2. **Explanatory Comments** (explain WHY):
```python
# Normalize pixel values from [0, 255] to [0, 1]
# Why? Neural networks train better with smaller, normalized values
# This prevents large pixel values from dominating the learning process
X_train = X_train / 255.0
```

3. **Inline Comments** (clarify complex lines):
```python
y_train = keras.utils.to_categorical(y_train, 10)  # Convert to one-hot (10 classes)
```

### 8.4 Best Practices

**Data Handling:**
- Verify data shapes after each transformation
- Use meaningful variable names (X_train_normalized vs X1)
- Print intermediate results to confirm correctness

**Model Building:**
- Start with simple architecture, add complexity if needed
- Use standard activation functions (ReLU, Softmax)
- Document reason for each architectural choice

**Training:**
- Start with fewer epochs for testing, increase for final run
- Monitor both loss and accuracy
- Save training history for analysis

**Evaluation:**
- Always evaluate on test set (never training set)
- Report multiple metrics if possible
- Compare results to baseline (random guessing = 10%)

### 8.5 Common Pitfalls to Avoid

**Pitfall 1: Forgetting to Normalize**
- Impact: Poor training performance, unstable learning
- Solution: Always normalize pixel values to [0,1] range

**Pitfall 2: Training on Test Data**
- Impact: Overfitting, inflated accuracy
- Solution: Strict separation of train/test data

**Pitfall 3: Wrong Input Shape**
- Impact: Model won't train, shape errors
- Solution: Verify shapes match network expectations (784 for flattened)

**Pitfall 4: Incorrect Label Encoding**
- Impact: Model won't learn, poor accuracy
- Solution: Use one-hot encoding with categorical crossentropy

**Pitfall 5: Insufficient Comments**
- Impact: Code not educational, defeats purpose
- Solution: Comment every significant step with WHY and WHAT

---

## 9. Acceptance Criteria

### 9.1 Functional Acceptance Criteria

**AC-0: Data Visualization**
- [ ] Sample images (digits 6-9) displayed in a 2x2 grid
- [ ] Images displayed with correct grayscale colormap
- [ ] Each image labeled with its digit value
- [ ] Matplotlib figure displays correctly
- [ ] Comments explain visualization purpose

**AC-0.1: GPU Configuration**
- [ ] TensorFlow detects available GPU(s)
- [ ] GPU device name printed at startup
- [ ] GPU memory growth configured (if applicable)
- [ ] Comments explain GPU usage benefits
- [ ] Fallback message if GPU not available

**AC-1: Data Loading**
- [ ] MNIST dataset loads successfully without errors
- [ ] Training set contains 60,000 samples
- [ ] Test set contains 10,000 samples
- [ ] Data shapes are verified and printed
- [ ] Comments explain dataset structure

**AC-2: Data Preprocessing**
- [ ] Pixel values normalized to [0, 1] range
- [ ] Images reshaped from (28, 28) to (784,)
- [ ] Labels converted to one-hot encoding (10 classes)
- [ ] Each preprocessing step has explanatory comments
- [ ] Shapes verified after each transformation

**AC-3: Model Architecture**
- [ ] Sequential model created successfully
- [ ] Input layer accepts 784-element vectors
- [ ] At least 2 hidden layers with ReLU activation
- [ ] Output layer has 10 neurons with Softmax
- [ ] Model summary displays correctly
- [ ] Each layer choice is documented

**AC-4: Model Compilation**
- [ ] Model compiled with Adam optimizer
- [ ] Categorical crossentropy loss function used
- [ ] Accuracy metric tracked
- [ ] Compilation parameters are documented
- [ ] Loss function explanation included (formula, why chosen, example calculation)
- [ ] Optimizer explanation included (why Adam, advantages, hyperparameters)

**AC-5: Model Training**
- [ ] Model trains for at least 10 epochs
- [ ] Training progress displayed (loss and accuracy per epoch)
- [ ] Training completes without errors
- [ ] Training time < 1 minute on GPU
- [ ] TensorFlow successfully utilizes GPU
- [ ] Comments explain training parameters

**AC-6: Model Evaluation**
- [ ] Model evaluated on test set (not training set)
- [ ] Test accuracy >= 95%
- [ ] Test loss and accuracy displayed clearly
- [ ] Results are interpretable and well-formatted

**AC-6.1: Training History Visualization**
- [ ] Loss curve graph displays correctly
- [ ] X-axis shows epochs, Y-axis shows loss value
- [ ] Training loss and validation loss plotted as separate lines
- [ ] Graph has title, legend, and grid
- [ ] Accuracy curve graph displays correctly
- [ ] Comments explain how to interpret the graphs (overfitting/underfitting)

**AC-6.2: Confusion Matrix**
- [ ] 10×10 confusion matrix generated from test predictions
- [ ] Matrix displayed as color-coded heatmap
- [ ] Cell values (counts) visible in each cell
- [ ] Axis labels show digits 0-9
- [ ] Title clearly identifies the matrix
- [ ] Comments explain how to read the confusion matrix
- [ ] Diagonal values significantly higher than off-diagonal (correct predictions)

**AC-7: Custom Image Prediction**
- [ ] Function loads external image from file path
- [ ] Supports PNG, JPG, JPEG formats
- [ ] Image converted to grayscale correctly
- [ ] Image resized to 28×28 pixels
- [ ] Colors inverted if necessary (white on black)
- [ ] Pixel values normalized to [0, 1]
- [ ] Image flattened and reshaped for model input
- [ ] Prediction returns digit and confidence score
- [ ] Prediction displayed with input image
- [ ] Error handling for file not found and invalid formats
- [ ] Comments explain each preprocessing step

**AC-7: Code Documentation**
- [ ] Every major section has header comments
- [ ] All important steps have explanatory comments
- [ ] Comments explain WHY, not just WHAT
- [ ] Technical terms are explained on first use
- [ ] Code is readable and follows PEP 8

### 9.2 Non-Functional Acceptance Criteria

**AC-8: Usability**
- [ ] Code is understandable by ML beginners
- [ ] Variable names are descriptive and meaningful
- [ ] Logical flow from top to bottom
- [ ] No unexplained "magic numbers"

**AC-9: Performance**
- [ ] Training completes in < 1 minute (GPU)
- [ ] GPU memory usage efficient (no OOM errors)
- [ ] Test accuracy > 95%

**AC-10: Reliability**
- [ ] Code runs without errors on fresh Python environment
- [ ] Results are reproducible (with seed)
- [ ] No warnings during execution

**AC-11: Portability**
- [ ] Runs on Windows, macOS, and Linux
- [ ] Compatible with Python 3.8+
- [ ] Works with TensorFlow 2.x
- [ ] No hardcoded absolute paths

### 9.3 Testing Checklist

**Pre-Launch Testing:**

1. **Environment Test:**
   - [ ] Test on fresh Python 3.8 environment
   - [ ] Test on fresh Python 3.10 environment
   - [ ] Test on Windows OS
   - [ ] Test on macOS (if available)
   - [ ] Test on Linux (if available)

2. **Functionality Test:**
   - [ ] Run entire script without modifications
   - [ ] Verify all sections execute in order
   - [ ] Confirm output at each stage
   - [ ] Verify final accuracy >= 95%

3. **Documentation Test:**
   - [ ] Have beginner read code and explain each section
   - [ ] Verify comments are clear and helpful
   - [ ] Check for typos or grammatical errors

4. **Performance Test:**
   - [ ] Measure training time on GPU (<1 minute target)
   - [ ] Monitor GPU memory usage during training
   - [ ] Test with different batch sizes (optimize for GPU)
   - [ ] Verify TensorFlow detects and uses GPU

5. **Code Quality Test:**
   - [ ] Run PEP 8 linter
   - [ ] Check for unused imports
   - [ ] Verify consistent naming conventions

---

## 10. Launch Plan

### 10.1 Pre-Launch (Week -1)

**Development:**
- Complete implementation following this PRD
- Internal testing on multiple environments
- Code review for clarity and correctness

**Documentation:**
- Finalize all code comments
- Create README with quick start guide
- Create requirements.txt

**Validation:**
- Run acceptance criteria checklist
- Test with sample users from target audience
- Fix any identified issues

### 10.2 Launch (Week 0)

**Delivery:**
- Submit completed code to course repository
- Provide README and requirements.txt
- Include sample output/results

**Communication:**
- Brief presentation explaining implementation
- Demo of running code
- Highlight key learning points

### 10.3 Post-Launch (Week +1)

**Monitoring:**
- Gather feedback from students
- Identify areas of confusion
- Note common questions

**Iteration:**
- Improve comments based on feedback
- Add clarifications where needed
- Update README with FAQ

### 10.4 Launch Checklist

**Code Deliverables:**
- [ ] mnist_digit_recognition.py (main implementation)
- [ ] requirements.txt (dependencies)
- [ ] README.md (quick start guide)
- [ ] PRD_Handwritten_Digit_Recognition.md (this document)

**Verification:**
- [ ] All acceptance criteria met
- [ ] Test accuracy >= 95%
- [ ] Code runs without errors
- [ ] Documentation complete

**Communication:**
- [ ] Notify instructor/students of availability
- [ ] Provide usage instructions
- [ ] Share expected output/results

---

## 11. Future Enhancements

### 11.1 Phase 2 Enhancements (After Initial Launch)

**Enhancement 1: Visualization**
- Plot training history (loss and accuracy curves)
- Display sample predictions with images
- Show confusion matrix
- **Impact:** Better understanding of model performance
- **Effort:** 2-3 hours

**Enhancement 2: Model Saving/Loading**
- Save trained model to disk
- Load pre-trained model for inference
- **Impact:** Reuse model without retraining
- **Effort:** 1-2 hours

**Enhancement 3: Prediction on Custom Images**
- Allow users to test on their own handwritten digits
- Image preprocessing for custom inputs
- **Impact:** More engaging and practical
- **Effort:** 3-4 hours

**Enhancement 4: Hyperparameter Experimentation**
- Easy configuration of epochs, batch size, learning rate
- Compare different architectures
- **Impact:** Encourage experimentation and learning
- **Effort:** 2 hours

### 11.2 Phase 3 Enhancements (Advanced Features)

**Enhancement 5: Convolutional Neural Network (CNN)**
- Upgrade to CNN for better accuracy (>98%)
- Explain advantages of CNNs for image data
- **Impact:** Introduce advanced architecture concepts
- **Effort:** 4-6 hours

**Enhancement 6: Data Augmentation**
- Add rotation, translation, zoom augmentation
- Explain regularization benefits
- **Impact:** Improve generalization, prevent overfitting
- **Effort:** 3-4 hours

**Enhancement 7: Transfer Learning**
- Use pre-trained models (e.g., VGG, ResNet adapted)
- Compare from-scratch vs transfer learning
- **Impact:** Introduce modern ML techniques
- **Effort:** 5-6 hours

**Enhancement 8: Interactive Dashboard**
- Streamlit or Gradio interface
- Real-time drawing and prediction
- **Impact:** More engaging demonstration
- **Effort:** 6-8 hours

### 11.3 Prioritization of Enhancements

**RICE Prioritization:**

| Enhancement | Reach | Impact | Confidence | Effort | RICE Score | Priority |
|-------------|-------|--------|------------|--------|------------|----------|
| Visualization | 100 | 3 | 0.9 | 2 | 135 | 1 |
| Model Saving | 100 | 2 | 1.0 | 1 | 200 | 2 |
| Custom Images | 80 | 3 | 0.8 | 3 | 64 | 4 |
| Hyperparameter Config | 90 | 2 | 1.0 | 2 | 90 | 3 |
| CNN Upgrade | 70 | 3 | 0.7 | 5 | 29.4 | 5 |

**Recommendation:** Prioritize Visualization and Model Saving for Phase 2.

---

## Appendices

### Appendix A: Glossary

**Neural Network:** A computational model inspired by biological neurons, consisting of layers of interconnected nodes that learn to recognize patterns.

**MNIST:** Modified National Institute of Standards and Technology database, a large collection of handwritten digits used for training and testing image processing systems.

**Activation Function:** A mathematical function that determines the output of a neural network node (e.g., ReLU, Softmax).

**ReLU (Rectified Linear Unit):** Activation function that outputs max(0, x), introducing non-linearity while being computationally efficient.

**Softmax:** Activation function that converts raw output scores into probabilities that sum to 1, used for multi-class classification.

**One-Hot Encoding:** Representation of categorical data as binary vectors (e.g., 3 → [0,0,0,1,0,0,0,0,0,0]).

**Epoch:** One complete pass through the entire training dataset.

**Batch Size:** Number of training samples processed before updating model weights.

**Loss Function:** Metric that measures how far model predictions are from actual values.

**Optimizer:** Algorithm that adjusts model weights to minimize loss (e.g., Adam, SGD).

**Categorical Crossentropy:** Loss function for multi-class classification problems.

### Appendix B: References

**TensorFlow Documentation:**
- https://www.tensorflow.org/api_docs/python/tf/keras
- https://www.tensorflow.org/datasets/catalog/mnist

**Educational Resources:**
- Deep Learning Specialization (Andrew Ng)
- Neural Networks and Deep Learning (Michael Nielsen)
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron)

**Benchmark Results:**
- MNIST Benchmark: http://yann.lecun.com/exdb/mnist/

### Appendix C: Sample Output

**Expected Console Output:**
```
============================================
MNIST Handwritten Digit Recognition
============================================

Checking GPU availability...
GPU detected: NVIDIA GeForce RTX 3080
Using GPU for training.

============================================
Loading MNIST dataset...
Training set: (60000, 28, 28) images
Test set: (10000, 28, 28) images

Displaying sample digits 6-9...
[Figure 1: 2x2 grid showing digits 6, 7, 8, 9]

============================================
Preprocessing data...
Normalizing pixel values (0-255 → 0-1)...
Reshaping images to vectors (28x28 → 784)...
One-hot encoding labels (0-9 → 10-dim vectors)...

============================================
Building neural network model...
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
dense (Dense)               (None, 128)               100480
dense_1 (Dense)             (None, 64)                8256
dense_2 (Dense)             (None, 10)                650
=================================================================
Total params: 109,386
Trainable params: 109,386

============================================
Compiling model...
- Optimizer: Adam (learning_rate=0.001)
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy

============================================
Training model...
Epoch 1/10
1875/1875 [==============================] - 2s - loss: 0.2645 - accuracy: 0.9239
Epoch 2/10
1875/1875 [==============================] - 1s - loss: 0.1162 - accuracy: 0.9651
...
Epoch 10/10
1875/1875 [==============================] - 1s - loss: 0.0234 - accuracy: 0.9924

============================================
Evaluating model on test data...
313/313 [==============================] - 0.5s

Test Loss: 0.0876
Test Accuracy: 97.42%

============================================
Generating visualizations...

[Figure 2: Loss Curve - Training and Validation Loss vs Epochs]
[Figure 3: Accuracy Curve - Training and Validation Accuracy vs Epochs]
[Figure 4: 10x10 Confusion Matrix Heatmap]

============================================
Custom Image Prediction
============================================
Enter image path (or 'q' to quit): test_digit.png

Preprocessing image...
- Loaded: test_digit.png
- Converted to grayscale
- Resized to 28x28
- Normalized pixel values

Prediction Results:
- Predicted Digit: 7
- Confidence: 98.5%

[Figure 5: Input image with prediction overlay]

Enter image path (or 'q' to quit): q
Exiting prediction mode.

============================================
Program completed successfully!
============================================
```

**Expected Visual Outputs:**

```
Figure 1: Sample Digits Preview
┌───────────────────────────────────────┐
│  ┌─────────┐    ┌─────────┐          │
│  │    ██   │    │  ████   │          │
│  │   ██    │    │     █   │          │
│  │  █████  │    │    ██   │          │
│  │  █   █  │    │   █     │          │
│  │  █████  │    │   █     │          │
│  │         │    │         │          │
│  └─ "6" ───┘    └─ "7" ───┘          │
│  ┌─────────┐    ┌─────────┐          │
│  │  ████   │    │  ████   │          │
│  │ █    █  │    │ █    █  │          │
│  │  ████   │    │  █████  │          │
│  │ █    █  │    │      █  │          │
│  │  ████   │    │  ████   │          │
│  │         │    │         │          │
│  └─ "8" ───┘    └─ "9" ───┘          │
└───────────────────────────────────────┘

Figure 2: Loss Curve
┌───────────────────────────────────────┐
│    Model Loss Over Training Epochs    │
│ Loss                                  │
│ 0.5 │●                                │
│     │ ●                               │
│ 0.4 │  ●                              │
│     │   ●------- Validation Loss      │
│ 0.3 │    ●                            │
│     │     ●                           │
│ 0.2 │      ●                          │
│     │       ●                         │
│ 0.1 │        ●---- Training Loss      │
│     │         ● ● ●                   │
│ 0.0 └─────────────────────────────────│
│     1  2  3  4  5  6  7  8  9  10     │
│                Epochs                 │
└───────────────────────────────────────┘

Figure 4: Confusion Matrix (10x10 Heatmap)
┌───────────────────────────────────────┐
│      Confusion Matrix - Test Set      │
│              Predicted                │
│        0  1  2  3  4  5  6  7  8  9   │
│      ┌──────────────────────────────┐ │
│    0 │██                            │ │
│    1 │   ██                         │ │
│ A  2 │      ██                      │ │
│ c  3 │         ██                   │ │
│ t  4 │            ██                │ │
│ u  5 │               ██             │ │
│ a  6 │                  ██          │ │
│ l  7 │                     ██       │ │
│    8 │                        ██    │ │
│    9 │                           ██ │ │
│      └──────────────────────────────┘ │
│   Dark = High count (correct)         │
│   Light = Low count (errors)          │
└───────────────────────────────────────┘
```

### Appendix D: Dependencies (requirements.txt)

```txt
tensorflow>=2.10.0      # Includes GPU support (requires CUDA/cuDNN)
numpy>=1.21.0
matplotlib>=3.4.0       # For visualization (graphs, confusion matrix heatmap)
seaborn>=0.11.0         # For enhanced confusion matrix heatmap
scikit-learn>=1.0.0     # For confusion_matrix function
Pillow>=9.0.0           # For loading and preprocessing custom images
```

**Library Purposes:**
- **tensorflow:** Core neural network framework (includes Keras)
- **numpy:** Numerical operations and array manipulation
- **matplotlib:** Plotting loss/accuracy curves, displaying images
- **seaborn:** Beautiful confusion matrix heatmap visualization
- **scikit-learn:** Confusion matrix calculation
- **Pillow (PIL):** Load and preprocess custom digit images

**GPU Setup Notes:**
- TensorFlow 2.10+ automatically detects and uses available GPUs
- Ensure CUDA and cuDNN are installed and compatible with TensorFlow version
- Verify GPU detection with: `print(tf.config.list_physical_devices('GPU'))`

### Appendix E: Minimum Viable Product (MVP) Scope

**MVP Includes:**
- GPU configuration and verification
- Data loading and exploration
- Sample images preview (digits 6-9)
- Data preprocessing (normalize, flatten, one-hot)
- Neural network (2 hidden layers: 128→64→10)
- Loss function with detailed explanation (Categorical Crossentropy)
- Optimizer with detailed explanation (Adam)
- Training and evaluation
- Training history visualization (loss and accuracy curves)
- Confusion matrix (10×10) with heatmap visualization
- Custom image prediction function
- Comprehensive documentation

**MVP Excludes:**
- Model saving/loading (Phase 2)
- Data augmentation (Phase 3)
- Advanced architectures like CNN (Phase 3)
- Interactive web interface (Phase 3)

**MVP Success Criteria:**
- Test accuracy >= 95%
- Complete documentation with explanations
- All visualizations display correctly
- Custom image prediction works
- Runs without errors on GPU
- Training time < 1 minute on GPU

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Feb 8, 2026 | AI Development Course | Initial PRD creation |

---

## Approval Sign-Off

**Product Manager:** ___________________________ Date: ___________

**Technical Lead:** ___________________________ Date: ___________

**Instructor:** ________________________________ Date: ___________

---

**End of Product Requirements Document**
