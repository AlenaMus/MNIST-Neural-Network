"""
============================================
MNIST Handwritten Digit Recognition Neural Network
============================================
Description: Educational implementation of a neural network that recognizes
             handwritten digits (0-9) using the MNIST dataset.
Purpose: Learn fundamental concepts of deep learning including data preprocessing,
         neural network architecture, training, and evaluation.
Target Accuracy: >95% on test set
Author: AI Development Course - Lesson 36
Date: February 8, 2026
============================================
"""

# ============================================
# CELL 1: Imports and GPU Configuration
# ============================================
# This cell imports all required libraries and configures GPU acceleration.
# GPU acceleration significantly speeds up neural network training (minutes → seconds).
# ============================================

# Core Deep Learning Framework
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Numerical Operations
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt          # For plotting graphs and displaying images
import seaborn as sns                    # For beautiful confusion matrix heatmaps

# Machine Learning Utilities
from sklearn.metrics import confusion_matrix  # For evaluating prediction errors

# Image Processing (for custom predictions)
from PIL import Image                    # For loading and preprocessing custom images

# GUI for Drawing and File Upload
import tkinter as tk
from tkinter import filedialog

# System Utilities
import os
import warnings
warnings.filterwarnings('ignore')        # Suppress warnings for cleaner output

print("=" * 60)
print("MNIST HANDWRITTEN DIGIT RECOGNITION")
print("=" * 60)
print()

# ============================================
# GPU CONFIGURATION AND VERIFICATION
# ============================================
# WHY GPU? Neural networks perform millions of matrix operations. GPUs are designed
# for parallel processing and can train models 10-100x faster than CPUs.
#
# What we're doing:
# 1. Check if GPU is available
# 2. Configure GPU memory growth (prevents out-of-memory errors)
# 3. Print GPU information
# ============================================

print("Checking GPU availability...")
print("-" * 60)

# Get list of all available GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Configure GPU memory growth
        # WHY? By default, TensorFlow allocates all GPU memory at once.
        # Memory growth allows TensorFlow to allocate memory as needed,
        # preventing crashes and allowing multiple programs to use the GPU.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Print GPU information
        print(f"✓ GPU DETECTED: {len(gpus)} GPU(s) available")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        print()
        print("GPU will be used for training (10-100x faster than CPU)")
        print("Expected training time: < 1 minute for 10 epochs")

    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        print("Falling back to CPU...")
else:
    print("⚠ NO GPU DETECTED - Using CPU")
    print("Training will be slower (~2-5 minutes for 10 epochs)")
    print()
    print("To enable GPU:")
    print("  1. Install NVIDIA GPU drivers")
    print("  2. Install CUDA Toolkit (11.x or 12.x)")
    print("  3. Install cuDNN library")
    print("  4. Reinstall: pip install tensorflow[and-cuda]")

print("=" * 60)
print()


# ============================================
# CELL 2: Load MNIST Dataset
# ============================================
# MNIST (Modified National Institute of Standards and Technology) is a classic
# dataset of 70,000 handwritten digit images (0-9).
#
# Dataset Structure:
# - Training set: 60,000 images (used to train the model)
# - Test set: 10,000 images (used to evaluate final performance)
# - Image format: 28x28 pixels, grayscale (1 channel)
# - Pixel values: 0-255 (0=black, 255=white)
# - Labels: 0-9 (which digit the image represents)
# ============================================

print("LOADING MNIST DATASET")
print("=" * 60)

# Load the MNIST dataset
# This function returns 4 arrays:
# - X_train: Training images (60000, 28, 28)
# - y_train: Training labels (60000,)
# - X_test: Test images (10000, 28, 28)
# - y_test: Test labels (10000,)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Display dataset information
print(f"✓ Dataset loaded successfully!")
print()
print("Dataset Statistics:")
print("-" * 60)
print(f"Training Images: {X_train.shape[0]:,} samples")
print(f"  Shape: {X_train.shape} (samples, height, width)")
print(f"  Pixel value range: {X_train.min()} to {X_train.max()}")
print()
print(f"Training Labels: {y_train.shape[0]:,} samples")
print(f"  Label range: {y_train.min()} to {y_train.max()} (digits 0-9)")
print()
print(f"Test Images: {X_test.shape[0]:,} samples")
print(f"  Shape: {X_test.shape}")
print()
print(f"Test Labels: {y_test.shape[0]:,} samples")
print()
print("Each image:")
print(f"  - Resolution: 28×28 pixels (784 total pixels)")
print(f"  - Color: Grayscale (single channel)")
print(f"  - Format: White digit on black background")
print("=" * 60)
print()


# ============================================
# CELL 3: Preview Sample Images (Digits 6-9)
# ============================================
# Visualizing the data helps us understand what we're working with.
# We'll display one example of each digit: 6, 7, 8, 9
#
# WHY visualize?
# - Verify data loaded correctly
# - Understand image quality and variation
# - See what the model needs to learn to distinguish
# ============================================

print("PREVIEWING SAMPLE DIGITS (6, 7, 8, 9)")
print("=" * 60)

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('Sample MNIST Digits: 6, 7, 8, 9', fontsize=16, fontweight='bold')

# Digits we want to display
target_digits = [6, 7, 8, 9]

# Find and display one sample of each digit
for idx, digit in enumerate(target_digits):
    # Find the first occurrence of this digit in training set
    # np.where returns indices where condition is True
    sample_index = np.where(y_train == digit)[0][0]

    # Get the image
    sample_image = X_train[sample_index]

    # Calculate subplot position (row, col)
    row = idx // 2  # 0, 0, 1, 1
    col = idx % 2   # 0, 1, 0, 1

    # Display the image
    axes[row, col].imshow(sample_image, cmap='gray')
    axes[row, col].set_title(f'Digit: {digit}', fontsize=14, fontweight='bold')
    axes[row, col].axis('off')  # Hide axis for cleaner display

plt.tight_layout()
plt.show()

print("✓ Sample images displayed successfully")
print("  Notice: Images are 28×28 pixels, grayscale")
print("  White pixels = digit, Black pixels = background")
print("=" * 60)
print()


# ============================================
# CELL 4: Data Preprocessing
# ============================================
# Raw data needs preprocessing before feeding to neural network.
# Three critical steps: Normalization, Flattening, One-Hot Encoding
# ============================================

print("DATA PREPROCESSING")
print("=" * 60)
print()

# ============================================
# STEP 4.1: NORMALIZATION
# ============================================
# WHY NORMALIZE?
# - Original pixel values: 0-255 (integers)
# - Neural networks learn better with smaller values (0-1 range)
# - Large values can cause:
#   1. Unstable gradients during training
#   2. Slow convergence
#   3. Numerical overflow/underflow
#
# HOW? Divide all pixel values by 255.0
# Result: Values now in range [0.0, 1.0]
# ============================================

print("Step 1: NORMALIZATION")
print("-" * 60)
print("Converting pixel values from [0-255] to [0.0-1.0]")
print()
print(f"Before normalization:")
print(f"  Min value: {X_train.min()}, Max value: {X_train.max()}")
print(f"  Data type: {X_train.dtype}")

# Normalize by dividing by 255.0
# Note: 255.0 (float) ensures result is float, not integer
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print()
print(f"After normalization:")
print(f"  Min value: {X_train.min():.4f}, Max value: {X_train.max():.4f}")
print(f"  Data type: {X_train.dtype}")
print("✓ Normalization complete")
print()

# ============================================
# STEP 4.2: FLATTENING
# ============================================
# WHY FLATTEN?
# - Current shape: (60000, 28, 28) - 2D images
# - Dense layers need 1D input: (60000, 784)
# - 28 × 28 = 784 pixels per image
#
# What happens to the image?
# - All rows are concatenated into one long vector
# - No information is lost, just reshaped
# - Example: [[1,2],[3,4]] → [1,2,3,4]
# ============================================

print("Step 2: FLATTENING")
print("-" * 60)
print("Converting 2D images (28×28) to 1D vectors (784)")
print()
print(f"Before flattening:")
print(f"  Training shape: {X_train.shape}")
print(f"  Test shape: {X_test.shape}")

# Reshape from (num_samples, 28, 28) to (num_samples, 784)
X_train = X_train.reshape(X_train.shape[0], 28 * 28)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)

print()
print(f"After flattening:")
print(f"  Training shape: {X_train.shape}")
print(f"  Test shape: {X_test.shape}")
print(f"  Each image is now a vector of {X_train.shape[1]} pixel values")
print("✓ Flattening complete")
print()

# ============================================
# STEP 4.3: ONE-HOT ENCODING
# ============================================
# WHY ONE-HOT ENCODE?
# - Current labels: Single integers (0, 1, 2, ..., 9)
# - Problem: Neural networks interpret these as ordered values
#   (model might think 9 is "larger" than 1)
# - Solution: One-hot encoding - each class gets its own dimension
#
# How it works:
# - Label 0 → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# - Label 1 → [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# - Label 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# - Label 9 → [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#
# Benefits:
# - Treats each digit as independent category
# - Works perfectly with categorical crossentropy loss
# - Output layer has 10 neurons (one per digit)
# ============================================

print("Step 3: ONE-HOT ENCODING")
print("-" * 60)
print("Converting integer labels to categorical vectors")
print()
print(f"Before one-hot encoding:")
print(f"  Training labels shape: {y_train.shape}")
print(f"  Sample labels: {y_train[:10]}")

# Convert labels to one-hot encoding
# to_categorical(labels, num_classes)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print()
print(f"After one-hot encoding:")
print(f"  Training labels shape: {y_train.shape}")
print(f"  Sample label (digit 5):")
print(f"    {y_train[0]}")
print(f"    ↑ Position with '1' indicates the digit class")
print("✓ One-hot encoding complete")
print()
print("=" * 60)
print("✓ ALL PREPROCESSING COMPLETE")
print("=" * 60)
print()


# ============================================
# CELL 5: Build Neural Network Architecture
# ============================================
# We'll build a fully connected (Dense) neural network with:
# - Input layer: 784 neurons (one per pixel)
# - Hidden layer 1: 128 neurons with ReLU activation
# - Hidden layer 2: 64 neurons with ReLU activation
# - Output layer: 10 neurons with Softmax activation
#
# WHY THIS ARCHITECTURE?
# - Fully Connected: Good for learning patterns in flattened images
# - Funnel Shape (784→128→64→10): Forces network to learn compressed features
# - ReLU: Fast, prevents vanishing gradients, works well for hidden layers
# - Softmax: Perfect for multi-class classification (outputs probabilities)
# ============================================

print("BUILDING NEURAL NETWORK ARCHITECTURE")
print("=" * 60)
print()

# Create Sequential model (layers stacked in sequence)
model = Sequential([

    # ============================================
    # HIDDEN LAYER 1: 128 neurons with ReLU
    # ============================================
    # Input: 784 features (flattened 28×28 image)
    # Output: 128 features
    # Parameters: 784 × 128 + 128 (bias) = 100,480
    #
    # WHY 128 neurons?
    # - Reduces dimensionality from 784 to 128
    # - Forces network to learn compressed representations
    # - Enough capacity to learn digit patterns
    # - Power of 2 (memory efficiency)
    #
    # WHY ReLU (Rectified Linear Unit)?
    # - Formula: ReLU(x) = max(0, x)
    # - Fast to compute (just max operation)
    # - Prevents vanishing gradient problem
    # - Introduces non-linearity (allows learning complex patterns)
    # - Outputs: positive values pass through, negative become 0
    Dense(128, activation='relu', input_shape=(784,), name='hidden_layer_1'),

    # ============================================
    # HIDDEN LAYER 2: 64 neurons with ReLU
    # ============================================
    # Input: 128 features from previous layer
    # Output: 64 features
    # Parameters: 128 × 64 + 64 (bias) = 8,256
    #
    # WHY 64 neurons?
    # - Further compression (128 → 64)
    # - Learns higher-level features from Layer 1
    # - Layer 1: Low-level features (edges, strokes)
    # - Layer 2: High-level features (curves, loops, digit shapes)
    # - Creates hierarchical feature learning
    Dense(64, activation='relu', name='hidden_layer_2'),

    # ============================================
    # OUTPUT LAYER: 10 neurons with Softmax
    # ============================================
    # Input: 64 features from previous layer
    # Output: 10 probabilities (one per digit 0-9)
    # Parameters: 64 × 10 + 10 (bias) = 650
    #
    # WHY 10 neurons?
    # - One neuron per class (digits 0-9)
    # - Each neuron outputs probability for that digit
    #
    # WHY Softmax?
    # - Converts raw scores to probabilities (sum = 1.0)
    # - Formula: softmax(x_i) = exp(x_i) / Σ(exp(x_j))
    # - Ensures mutual exclusivity (high prob for one class → low for others)
    # - Perfect for multi-class classification
    # - Example output: [0.01, 0.02, 0.03, 0.89, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    #                                         ↑ 89% confident it's digit 3
    Dense(10, activation='softmax', name='output_layer')
])

print("✓ Model architecture created")
print()
print("ARCHITECTURE SUMMARY:")
print("-" * 60)
print("Layer Structure (Funnel Design):")
print("  Input:    784 neurons  (28×28 flattened image)")
print("           ↓")
print("  Hidden 1: 128 neurons  (ReLU) - Feature extraction")
print("           ↓")
print("  Hidden 2:  64 neurons  (ReLU) - Feature combination")
print("           ↓")
print("  Output:    10 neurons  (Softmax) - Classification")
print()

# Display detailed model summary
print("DETAILED MODEL SUMMARY:")
print("-" * 60)
model.summary()

# Calculate total parameters
total_params = model.count_params()
print()
print(f"Total Parameters: {total_params:,}")
print("  These are the 'weights' that will be learned during training")
print()
print("=" * 60)
print()


# ============================================
# CELL 6: Compile Model
# ============================================
# Compilation configures the learning process by specifying:
# 1. OPTIMIZER: Algorithm that updates weights (HOW to learn)
# 2. LOSS FUNCTION: Metric to minimize (WHAT to optimize)
# 3. METRICS: Performance measures to track (HOW WELL is it learning)
# ============================================

print("COMPILING THE MODEL")
print("=" * 60)
print()

# ============================================
# OPTIMIZER: ADAM (Adaptive Moment Estimation)
# ============================================
# WHY ADAM?
# - BEST general-purpose optimizer for deep learning
# - Combines advantages of two other optimizers:
#   1. Momentum: Uses moving average of past gradients (smooths learning)
#   2. RMSprop: Adapts learning rate per parameter
#
# How Adam works:
# - Maintains two moving averages for each parameter:
#   1. Mean of gradients (momentum)
#   2. Variance of gradients (adaptive learning rate)
# - Automatically adjusts learning rate for each parameter
# - Fast convergence with minimal hyperparameter tuning
#
# Default Hyperparameters (work well for most problems):
# - learning_rate = 0.001 (how big each update step is)
# - beta_1 = 0.9 (momentum decay rate)
# - beta_2 = 0.999 (variance decay rate)
# - epsilon = 1e-7 (prevents division by zero)
#
# Advantages over other optimizers:
# - vs SGD: Faster convergence, adaptive learning rates
# - vs RMSprop: Adds momentum for better optimization
# - vs Adagrad: Learning rate doesn't decay to zero
#
# Expected performance on MNIST:
# - Reaches 95%+ accuracy in ~5 epochs
# - Stable training without oscillation
# - No need for learning rate scheduling
# ============================================

print("1. OPTIMIZER: Adam")
print("-" * 60)
print("Adam (Adaptive Moment Estimation) is the gold-standard optimizer")
print()
print("How it works:")
print("  - Standard SGD: weight = weight - learning_rate × gradient")
print("  - Adam: weight = weight - adaptive_lr × momentum_adjusted_gradient")
print()
print("Key features:")
print("  ✓ Adaptive learning rates per parameter")
print("  ✓ Momentum (remembers past gradients)")
print("  ✓ Fast convergence")
print("  ✓ Minimal tuning needed")
print()
print("Default parameters (using these):")
print("  - Learning rate: 0.001")
print("  - Beta_1 (momentum): 0.9")
print("  - Beta_2 (variance): 0.999")
print()

# ============================================
# LOSS FUNCTION: CATEGORICAL CROSSENTROPY
# ============================================
# WHY CATEGORICAL CROSSENTROPY?
# - Standard loss for multi-class classification (10 classes: digits 0-9)
# - Measures difference between predicted and true probability distributions
# - Works perfectly with softmax output and one-hot encoded labels
# - Heavily penalizes confident wrong predictions
#
# Mathematical Formula:
# Loss = -Σ(y_true × log(y_pred))
#
# Where:
# - y_true = true label (one-hot: [0,0,0,1,0,0,0,0,0,0] for digit 3)
# - y_pred = predicted probabilities from softmax layer
# - log = natural logarithm
# - Σ = sum over all 10 classes
#
# EXAMPLE CALCULATION:
# ---------------------
# Scenario 1: GOOD PREDICTION
# True label: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# Predicted:      [0.01, 0.02, 0.05, 0.85, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]
#                                    ↑ 85% confident it's a 3 (CORRECT!)
#
# Loss = -(0×log(0.01) + 0×log(0.02) + ... + 1×log(0.85) + ...)
#      = -log(0.85)
#      = 0.163  ← LOW LOSS (good prediction)
#
# Scenario 2: BAD PREDICTION
# True label: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# Predicted:      [0.85, 0.02, 0.05, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]
#                  ↑ 85% confident it's a 0 (WRONG!)
#                                    ↑ Only 1% for correct class
#
# Loss = -log(0.01)
#      = 4.605  ← HIGH LOSS (bad prediction)
#
# Notice: Loss increases exponentially as confidence in wrong answer increases!
#
# Why not other loss functions?
# - MSE (Mean Squared Error): Works but slower convergence for classification
# - Binary Crossentropy: Only for 2-class problems (we have 10 classes)
# - Sparse Categorical Crossentropy: Same as categorical but for integer labels
#   (we use one-hot, so categorical is correct)
# ============================================

print("2. LOSS FUNCTION: Categorical Crossentropy")
print("-" * 60)
print("What is a loss function?")
print("  A measure of how 'wrong' the model's predictions are")
print("  Training goal: MINIMIZE this loss")
print()
print("Formula: Loss = -Σ(y_true × log(y_pred))")
print()
print("Example calculation:")
print("  True label: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]")
print("  Good prediction: [0.01, 0.02, 0.05, 0.85, ...]")
print("                                      ↑ 85% for class 3")
print("  Loss = -log(0.85) = 0.163  ← LOW (good!)")
print()
print("  Bad prediction: [0.85, 0.02, 0.05, 0.01, ...]")
print("                   ↑ 85% for class 0 (wrong!)")
print("                                     ↑ 1% for class 3 (correct)")
print("  Loss = -log(0.01) = 4.605  ← HIGH (bad!)")
print()
print("Why this loss function?")
print("  ✓ Perfect for multi-class classification (10 digits)")
print("  ✓ Works with softmax output")
print("  ✓ Penalizes confident wrong predictions heavily")
print("  ✓ Based on information theory (entropy)")
print()

# ============================================
# METRICS: ACCURACY
# ============================================
# Accuracy = (Number of correct predictions) / (Total predictions)
# Easy to interpret: 95% accuracy = model is correct 95% of the time
# ============================================

print("3. METRICS: Accuracy")
print("-" * 60)
print("Accuracy = Correct Predictions / Total Predictions")
print("  Easy to understand: 95% = correct 19 out of 20 times")
print("  We'll track this during training and evaluation")
print()

# Compile the model with optimizer, loss, and metrics
model.compile(
    optimizer='adam',                    # Adam optimizer with default params
    loss='categorical_crossentropy',     # Loss function for multi-class
    metrics=['accuracy']                 # Track accuracy during training
)

print("=" * 60)
print("✓ MODEL COMPILED SUCCESSFULLY")
print("=" * 60)
print("Ready for training with:")
print("  Optimizer: Adam (learning_rate=0.001)")
print("  Loss: Categorical Crossentropy")
print("  Metric: Accuracy")
print("=" * 60)
print()


# ============================================
# CELL 7: Train the Model
# ============================================
# Training is the process where the model learns to recognize digits.
#
# How training works:
# 1. Show model a batch of images
# 2. Model makes predictions
# 3. Calculate loss (how wrong the predictions are)
# 4. Backpropagation: Calculate gradients
# 5. Optimizer updates weights to reduce loss
# 6. Repeat until model learns patterns
#
# Training Parameters:
# - EPOCHS: Number of complete passes through training data
# - BATCH_SIZE: Number of samples processed before updating weights
# - VALIDATION_SPLIT: Percentage of training data to use for validation
# ============================================

print("TRAINING THE MODEL")
print("=" * 60)
print()

# Training Parameters
EPOCHS = 10          # Number of complete passes through training data
BATCH_SIZE = 32      # Number of samples per weight update

print("Training Configuration:")
print("-" * 60)
print(f"Epochs: {EPOCHS}")
print("  → The model will see the entire training dataset 10 times")
print("  → Each epoch improves the model's understanding")
print()
print(f"Batch Size: {BATCH_SIZE}")
print("  → Process 32 images at a time before updating weights")
print("  → Smaller batch = more frequent updates, noisier gradients")
print("  → Larger batch = more stable gradients, less frequent updates")
print("  → 32 is a good balance for this dataset")
print()
print(f"Validation Split: 10%")
print("  → Use 10% of training data to monitor overfitting")
print("  → Validation data is NOT used for training, only evaluation")
print("  → Helps detect if model is memorizing vs. learning")
print()
print("Training Process:")
print("  1. Feed batch of images to model")
print("  2. Model makes predictions")
print("  3. Calculate loss (prediction error)")
print("  4. Backpropagation (calculate how to adjust weights)")
print("  5. Optimizer updates weights")
print("  6. Repeat for all batches (= 1 epoch)")
print("  7. Repeat for all epochs")
print()
print("=" * 60)
print("STARTING TRAINING...")
print("=" * 60)
print()

# Train the model
# history object stores metrics for each epoch (loss, accuracy, etc.)
history = model.fit(
    X_train,                    # Training images (flattened, normalized)
    y_train,                    # Training labels (one-hot encoded)
    epochs=EPOCHS,              # Number of complete passes through data
    batch_size=BATCH_SIZE,      # Samples per gradient update
    validation_split=0.1,       # Use 10% of training data for validation
    verbose=1                   # Display progress bar
)

print()
print("=" * 60)
print("✓ TRAINING COMPLETE!")
print("=" * 60)
print()
print("What just happened?")
print("  - Model saw 60,000 training images 10 times (10 epochs)")
print("  - Adjusted 109,386 parameters to minimize prediction error")
print("  - Loss decreased → Model learned patterns in digit images")
print("  - Accuracy increased → Model makes better predictions")
print()
print("Notice the trend:")
print("  - Early epochs: Big improvements (learning basic patterns)")
print("  - Later epochs: Small improvements (refining understanding)")
print("  - Validation accuracy close to training = good generalization")
print("=" * 60)
print()


# ============================================
# CELL 8: Evaluate Model
# ============================================
# After training, we evaluate the model on TEST DATA.
#
# WHY TEST DATA?
# - Test set contains images the model has NEVER seen
# - Measures real-world performance (generalization)
# - Training accuracy can be misleading (overfitting)
#
# OVERFITTING vs. GENERALIZATION:
# - Overfitting: Model memorizes training data, fails on new data
#   (High training accuracy, low test accuracy)
# - Good generalization: Model learns patterns, works on new data
#   (Training and test accuracy are similar)
# ============================================

print("EVALUATING MODEL ON TEST SET")
print("=" * 60)
print()
print("Why evaluate on test set?")
print("  - Test data was NEVER seen during training")
print("  - Measures real-world performance")
print("  - Detects overfitting (memorization vs. learning)")
print()
print("Evaluating on 10,000 test images...")
print("-" * 60)

# Evaluate model on test set
# Returns: [test_loss, test_accuracy]
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

print()
print("=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print()

# Interpret results
if test_accuracy >= 0.97:
    print("✓ EXCELLENT! Model performs very well (≥97%)")
elif test_accuracy >= 0.95:
    print("✓ GOOD! Model meets target accuracy (≥95%)")
elif test_accuracy >= 0.90:
    print("⚠ ACCEPTABLE but below target (90-95%)")
else:
    print("✗ POOR performance (<90%) - Model needs improvement")

print()
print("What does this mean?")
print(f"  - Out of 10,000 test images, model correctly classifies")
print(f"    approximately {int(test_accuracy * 10000):,} images")
print(f"  - Error rate: {(1 - test_accuracy) * 100:.2f}%")
print()

# Compare training vs test accuracy (check for overfitting)
final_train_accuracy = history.history['accuracy'][-1]
accuracy_gap = abs(final_train_accuracy - test_accuracy)

print("Overfitting Check:")
print(f"  Final Training Accuracy: {final_train_accuracy * 100:.2f}%")
print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"  Difference: {accuracy_gap * 100:.2f}%")

if accuracy_gap < 0.03:
    print("  ✓ Good generalization (difference < 3%)")
elif accuracy_gap < 0.05:
    print("  ⚠ Slight overfitting (difference 3-5%)")
else:
    print("  ✗ Overfitting detected (difference > 5%)")
    print("    Model memorized training data instead of learning patterns")

print("=" * 60)
print()


# ============================================
# CELL 9: Plot Training History (Loss Curve)
# ============================================
# Visualize how loss changed during training.
#
# WHAT IS A LOSS CURVE?
# - X-axis: Epochs (training iterations)
# - Y-axis: Loss value (lower = better)
# - Two lines: Training loss and Validation loss
#
# HOW TO INTERPRET:
# - Both decreasing: Model is learning ✓
# - Training << Validation: Overfitting (memorizing) ✗
# - Both flat: Underfitting (not learning enough) ✗
# - Lines converging: Good generalization ✓
# ============================================

print("VISUALIZING TRAINING HISTORY - LOSS CURVE")
print("=" * 60)

# Create figure for loss curve
plt.figure(figsize=(12, 5))

# Plot training and validation loss
plt.plot(history.history['loss'], marker='o', linestyle='-', linewidth=2,
         label='Training Loss', color='#2E86AB')
plt.plot(history.history['val_loss'], marker='s', linestyle='--', linewidth=2,
         label='Validation Loss', color='#A23B72')

# Customize plot
plt.title('Model Loss Over Training Epochs', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss (Categorical Crossentropy)', fontsize=12, fontweight='bold')
plt.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(range(EPOCHS))
plt.tight_layout()
plt.show()

print()
print("How to interpret this graph:")
print("-" * 60)
print("✓ Both lines decreasing = Model is learning")
print("  Loss starts high, drops quickly in early epochs")
print()
print("✓ Lines close together = Good generalization")
print("  Model performs similarly on training and validation data")
print()
print("✗ Training << Validation = Overfitting")
print("  Model memorizes training data, fails on new data")
print()
print("✗ Both lines flat/high = Underfitting")
print("  Model not learning enough, needs more capacity/training")
print("=" * 60)
print()


# ============================================
# CELL 10: Plot Training History (Accuracy Curve)
# ============================================
# Visualize how accuracy improved during training.
# Accuracy is easier to interpret than loss (higher = better).
# ============================================

print("VISUALIZING TRAINING HISTORY - ACCURACY CURVE")
print("=" * 60)

# Create figure for accuracy curve
plt.figure(figsize=(12, 5))

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], marker='o', linestyle='-', linewidth=2,
         label='Training Accuracy', color='#06A77D')
plt.plot(history.history['val_accuracy'], marker='s', linestyle='--', linewidth=2,
         label='Validation Accuracy', color='#D4526E')

# Customize plot
plt.title('Model Accuracy Over Training Epochs', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(range(EPOCHS))
plt.ylim([0.9, 1.0])  # Focus on 90-100% range for better visibility
plt.tight_layout()
plt.show()

print()
print("How to interpret this graph:")
print("-" * 60)
print("✓ Both lines increasing = Model is learning")
print("  Accuracy improves with each epoch")
print()
print("✓ Lines close together = Good generalization")
print("  Similar performance on training and validation data")
print()
print("✗ Training >> Validation = Overfitting")
print("  High training accuracy but lower validation accuracy")
print()
print("Notice:")
print("  - Rapid improvement in early epochs (learning basic patterns)")
print("  - Slower improvement in later epochs (fine-tuning)")
print("  - Diminishing returns after ~5-7 epochs")
print("=" * 60)
print()


# ============================================
# CELL 11: Confusion Matrix
# ============================================
# A confusion matrix shows which digits the model confuses with each other.
#
# STRUCTURE:
# - Rows: Actual/True labels (what the digit really is)
# - Columns: Predicted labels (what the model thinks it is)
# - Diagonal: Correct predictions (should be highest)
# - Off-diagonal: Misclassifications (errors)
#
# EXAMPLE:
# Cell[3,8] = 15 means: 15 times a "3" was incorrectly predicted as "8"
#
# WHY USEFUL?
# - Identifies which digits are hardest to classify
# - Reveals systematic errors (e.g., 4 confused with 9)
# - Helps understand model's strengths and weaknesses
# ============================================

print("GENERATING CONFUSION MATRIX")
print("=" * 60)
print()
print("What is a confusion matrix?")
print("  A 10×10 table showing actual vs. predicted labels")
print("  - Rows: True digit (0-9)")
print("  - Columns: Predicted digit (0-9)")
print("  - Diagonal: Correct predictions")
print("  - Off-diagonal: Errors (confusions)")
print()
print("Generating predictions on test set...")

# Generate predictions on test set
y_pred_probs = model.predict(X_test, verbose=0)

# Convert predictions from probabilities to class labels
# argmax returns index of highest probability
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Convert true labels from one-hot to class labels
y_true_classes = np.argmax(y_test, axis=1)

# Create confusion matrix
# confusion_matrix[i,j] = number of samples with true label i predicted as j
cm = confusion_matrix(y_true_classes, y_pred_classes)

print(f"✓ Generated predictions for {len(y_test):,} test images")
print()

# Visualize confusion matrix as heatmap
print("Displaying confusion matrix heatmap...")
print()

plt.figure(figsize=(12, 10))

# Create heatmap using seaborn
sns.heatmap(cm,
            annot=True,              # Show numbers in cells
            fmt='d',                 # Integer format
            cmap='Blues',            # Color scheme (light to dark blue)
            xticklabels=range(10),   # Labels 0-9 on x-axis
            yticklabels=range(10),   # Labels 0-9 on y-axis
            cbar_kws={'label': 'Number of Predictions'},
            linewidths=0.5,          # Grid lines between cells
            linecolor='gray')

plt.title('Confusion Matrix - MNIST Test Set (10,000 images)',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Digit', fontsize=13, fontweight='bold')
plt.ylabel('True Digit', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print()
print("How to read the confusion matrix:")
print("-" * 60)
print("Diagonal (top-left to bottom-right):")
print("  → Correct predictions (darker = more correct)")
print()
print("Off-diagonal (all other cells):")
print("  → Misclassifications (errors)")
print("  → Example: Cell[3,8] = how many '3's predicted as '8'")
print()

# Find most common confusion
max_confusion = 0
confused_pair = (0, 0)
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > max_confusion:
            max_confusion = cm[i, j]
            confused_pair = (i, j)

print("Most common confusion:")
print(f"  Digit {confused_pair[0]} misclassified as {confused_pair[1]}: "
      f"{max_confusion} times")
print()

# Calculate per-digit accuracy
print("Per-digit accuracy:")
print("-" * 60)
for digit in range(10):
    correct = cm[digit, digit]
    total = cm[digit].sum()
    digit_accuracy = correct / total * 100
    print(f"  Digit {digit}: {digit_accuracy:.2f}% ({correct}/{total} correct)")

print()
print("Common confusions to look for:")
print("  - 4 ↔ 9 (similar top parts)")
print("  - 3 ↔ 8 (similar curves)")
print("  - 5 ↔ 6 (similar shapes)")
print("  - 7 ↔ 1 (similar strokes)")
print("=" * 60)
print()


# ============================================
# CELL 12: Custom Image Prediction Functions
# ============================================
# These functions allow prediction on custom handwritten digit images.
# Useful for testing the model on your own drawings.
# ============================================

print("CUSTOM IMAGE PREDICTION FUNCTIONS")
print("=" * 60)
print()

def preprocess_image(image_path):
    """
    Preprocess a custom image to match MNIST format.

    Pipeline:
    1. Load image from file
    2. Convert to grayscale (if RGB)
    3. Resize to 28×28 pixels
    4. Invert colors if needed (MNIST = white on black)
    5. Normalize to [0, 1]
    6. Flatten to 784-element vector
    7. Reshape to (1, 784) for model input

    Args:
        image_path (str): Path to image file

    Returns:
        numpy.ndarray: Preprocessed image ready for prediction (1, 784)
    """
    try:
        # 1. Load image
        img = Image.open(image_path)

        # 2. Convert to grayscale
        img = img.convert('L')  # 'L' mode = grayscale (0-255)

        # 3. Resize to 28×28 (MNIST dimensions)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # 4. Convert to numpy array
        img_array = np.array(img)

        # 5. Invert colors if needed
        # MNIST format: white digit (255) on black background (0)
        # If your image has black digit on white background, uncomment:
        # img_array = 255 - img_array

        # 6. Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0

        # 7. Flatten to 784 elements
        img_array = img_array.reshape(1, 784)

        return img_array

    except FileNotFoundError:
        print(f"Error: File not found - {image_path}")
        return None
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def predict_digit(model, image_path):
    """
    Predict the digit in a custom image.

    Args:
        model: Trained Keras model
        image_path (str): Path to image file

    Returns:
        tuple: (predicted_digit, confidence, all_probabilities)
               or (None, None, None) if error
    """
    # Preprocess image
    img_array = preprocess_image(image_path)

    if img_array is None:
        return None, None, None

    # Get prediction
    predictions = model.predict(img_array, verbose=0)

    # Get predicted class (digit with highest probability)
    predicted_digit = np.argmax(predictions[0])

    # Get confidence (probability of predicted class)
    confidence = predictions[0][predicted_digit] * 100

    # Get all probabilities
    all_probabilities = predictions[0]

    return predicted_digit, confidence, all_probabilities


def display_prediction(image_path, digit, confidence, probabilities):
    """
    Display the input image with prediction results.

    Args:
        image_path (str): Path to image file
        digit (int): Predicted digit (0-9)
        confidence (float): Confidence percentage
        probabilities (numpy.ndarray): Probabilities for all classes
    """
    # Load and display original image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Original image
    img = Image.open(image_path).convert('L')
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Right: Probability bar chart
    ax2.bar(range(10), probabilities * 100, color='steelblue', alpha=0.7)
    ax2.bar(digit, probabilities[digit] * 100, color='crimson', alpha=0.9)
    ax2.set_xlabel('Digit', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Prediction: {digit} (Confidence: {confidence:.2f}%)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(range(10))
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Print results
    print()
    print("=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Predicted Digit: {digit}")
    print(f"Confidence: {confidence:.2f}%")
    print()
    print("All class probabilities:")
    for i, prob in enumerate(probabilities):
        marker = " ← PREDICTED" if i == digit else ""
        print(f"  Digit {i}: {prob * 100:5.2f}%{marker}")
    print("=" * 60)
    print()


print("✓ Custom prediction functions defined:")
print("  - preprocess_image(image_path)")
print("  - predict_digit(model, image_path)")
print("  - display_prediction(image_path, digit, confidence, probabilities)")
print()
print("=" * 60)
print()


# ============================================
# CELL 13: Interactive GUI - Draw or Upload Images
# ============================================
# A Tkinter GUI window with two ways to test the model:
#   1. DRAW a digit with your mouse on a canvas
#   2. UPLOAD an image file using a file browser dialog
#
# The GUI shows the prediction results with a probability chart.
# ============================================

def predict_from_array(img_array_2d):
    """
    Predict digit from a 28x28 numpy array and display results.

    Args:
        img_array_2d: numpy array of shape (28, 28)

    Returns:
        tuple: (predicted_digit, confidence)
    """
    # Normalize and flatten
    img_flat = img_array_2d.astype('float32')
    if img_flat.max() > 1:
        img_flat = img_flat / 255.0
    img_flat = img_flat.reshape(1, 784)

    # Predict
    predictions = model.predict(img_flat, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = predictions[0][predicted_digit] * 100

    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(img_array_2d, cmap='gray')
    ax1.set_title('Image (as model sees it)', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2.bar(range(10), predictions[0] * 100, color='steelblue', alpha=0.7)
    ax2.bar(predicted_digit, predictions[0][predicted_digit] * 100, color='crimson', alpha=0.9)
    ax2.set_xlabel('Digit', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Prediction: {predicted_digit} ({confidence:.1f}%)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(range(10))
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    print(f"\nPredicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")
    print()
    print("All class probabilities:")
    for i, prob in enumerate(predictions[0]):
        marker = " <-- PREDICTED" if i == predicted_digit else ""
        print(f"  Digit {i}: {prob * 100:5.2f}%{marker}")
    print()

    return predicted_digit, confidence


def open_drawing_gui():
    """
    Open a Tkinter GUI window with:
    - A drawing canvas (black background, white pen) to draw digits
    - A "Load Image" button to upload an image file via file browser
    - A "Predict" button to run the model on the drawn/loaded image
    - A "Clear" button to reset the canvas
    """
    print("INTERACTIVE GUI MODE")
    print("=" * 60)
    print("A drawing window will open.")
    print("  - Draw a digit with your mouse (white on black)")
    print("  - Or click 'Load Image' to browse for an image file")
    print("  - Click 'Predict' to see the model's prediction")
    print("  - Click 'Clear' to erase and try again")
    print("  - Close the window when done")
    print("=" * 60)
    print()

    # ============================================
    # Create the main window
    # ============================================
    root = tk.Tk()
    root.title("MNIST Digit Prediction - Draw or Upload")
    root.configure(bg='#2b2b2b')
    root.resizable(False, False)

    # Canvas size: 280x280 (10x MNIST resolution for easier drawing)
    CANVAS_SIZE = 280
    PEN_WIDTH = 18  # Thick pen for drawing digits

    # ============================================
    # Title label
    # ============================================
    title_label = tk.Label(
        root,
        text="Draw a digit (0-9) or Load an image",
        font=('Arial', 14, 'bold'),
        fg='white', bg='#2b2b2b',
        pady=10
    )
    title_label.pack()

    # ============================================
    # Drawing canvas (black background)
    # ============================================
    canvas = tk.Canvas(
        root,
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        bg='black',
        cursor='crosshair',
        highlightthickness=2,
        highlightbackground='#666'
    )
    canvas.pack(padx=20, pady=5)

    # ============================================
    # Status label (shows prediction result)
    # ============================================
    status_label = tk.Label(
        root,
        text="Draw a digit and click 'Predict'",
        font=('Arial', 12),
        fg='#aaaaaa', bg='#2b2b2b',
        pady=5
    )
    status_label.pack()

    # ============================================
    # Drawing state
    # ============================================
    drawing_state = {'last_x': None, 'last_y': None}

    def start_draw(event):
        drawing_state['last_x'] = event.x
        drawing_state['last_y'] = event.y

    def draw(event):
        if drawing_state['last_x'] is not None:
            canvas.create_line(
                drawing_state['last_x'], drawing_state['last_y'],
                event.x, event.y,
                fill='white',
                width=PEN_WIDTH,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND
            )
        drawing_state['last_x'] = event.x
        drawing_state['last_y'] = event.y

    def stop_draw(event):
        drawing_state['last_x'] = None
        drawing_state['last_y'] = None

    canvas.bind('<Button-1>', start_draw)
    canvas.bind('<B1-Motion>', draw)
    canvas.bind('<ButtonRelease-1>', stop_draw)

    # ============================================
    # Get image from canvas as numpy array
    # ============================================
    def get_canvas_image():
        """Capture the canvas content as a 28x28 numpy array."""
        # Save canvas to a temporary PostScript file, then convert
        import tempfile

        # Method: Save canvas as .ps, convert to image via PIL
        tmp_ps = os.path.join(tempfile.gettempdir(), '_mnist_canvas.ps')
        tmp_png = os.path.join(tempfile.gettempdir(), '_mnist_canvas.png')

        try:
            canvas.postscript(file=tmp_ps, colormode='gray')
            # Convert PostScript to PNG using PIL
            img = Image.open(tmp_ps)
            img = img.convert('L')
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(img)

            # Clean up temp files
            if os.path.exists(tmp_ps):
                os.remove(tmp_ps)

            return img_array

        except Exception:
            # Fallback: manually read canvas pixel data using winfo
            # Create a blank image and draw the lines manually
            img = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 0)
            from PIL import ImageDraw
            draw_ctx = ImageDraw.Draw(img)

            # Read all line items from canvas
            for item in canvas.find_all():
                coords = canvas.coords(item)
                if len(coords) == 4:
                    draw_ctx.line(coords, fill=255, width=PEN_WIDTH)

            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(img)

            # Clean up temp files
            if os.path.exists(tmp_ps):
                os.remove(tmp_ps)

            return img_array

    # ============================================
    # Button actions
    # ============================================
    def on_predict():
        """Get canvas image and run prediction."""
        img_array = get_canvas_image()
        status_label.config(text="Predicting...", fg='yellow')
        root.update()

        predicted, confidence = predict_from_array(img_array)
        status_label.config(
            text=f"Prediction: {predicted}  (Confidence: {confidence:.1f}%)",
            fg='#00ff00',
            font=('Arial', 14, 'bold')
        )

    def on_clear():
        """Clear the canvas."""
        canvas.delete('all')
        status_label.config(
            text="Draw a digit and click 'Predict'",
            fg='#aaaaaa',
            font=('Arial', 12)
        )

    def on_load_image():
        """Open file browser, load image, display on canvas, predict."""
        file_path = filedialog.askopenfilename(
            title="Select a handwritten digit image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return  # User cancelled

        try:
            # Load and preprocess
            img = Image.open(file_path).convert('L')
            img_resized = img.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized)

            # Auto-invert: if background is bright (white paper), invert colors
            # MNIST expects white digit on black background
            if img_array.mean() > 127:
                print("  (Auto-inverted colors: black-on-white -> white-on-black)")
                img_array = 255 - img_array

            # Display loaded image on canvas
            canvas.delete('all')
            # Scale up to canvas size for display
            img_display = Image.fromarray(img_array).resize(
                (CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.NEAREST
            )
            # Convert to PhotoImage for Tkinter
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(img_display)
            canvas.image = photo  # Keep reference to prevent garbage collection
            canvas.create_image(0, 0, anchor='nw', image=photo)

            status_label.config(text="Image loaded! Predicting...", fg='yellow')
            root.update()

            # Predict
            predicted, confidence = predict_from_array(img_array)
            status_label.config(
                text=f"Prediction: {predicted}  (Confidence: {confidence:.1f}%)",
                fg='#00ff00',
                font=('Arial', 14, 'bold')
            )

        except Exception as e:
            status_label.config(text=f"Error: {e}", fg='red')

    # ============================================
    # Buttons frame
    # ============================================
    btn_frame = tk.Frame(root, bg='#2b2b2b', pady=10)
    btn_frame.pack()

    predict_btn = tk.Button(
        btn_frame, text="Predict",
        command=on_predict,
        font=('Arial', 12, 'bold'),
        bg='#4CAF50', fg='white',
        activebackground='#45a049',
        width=10, height=1,
        cursor='hand2'
    )
    predict_btn.pack(side=tk.LEFT, padx=5)

    clear_btn = tk.Button(
        btn_frame, text="Clear",
        command=on_clear,
        font=('Arial', 12, 'bold'),
        bg='#f44336', fg='white',
        activebackground='#da190b',
        width=10, height=1,
        cursor='hand2'
    )
    clear_btn.pack(side=tk.LEFT, padx=5)

    load_btn = tk.Button(
        btn_frame, text="Load Image",
        command=on_load_image,
        font=('Arial', 12, 'bold'),
        bg='#2196F3', fg='white',
        activebackground='#0b7dda',
        width=10, height=1,
        cursor='hand2'
    )
    load_btn.pack(side=tk.LEFT, padx=5)

    # ============================================
    # Instructions label at bottom
    # ============================================
    info_label = tk.Label(
        root,
        text="Tip: Draw a large, centered digit for best results",
        font=('Arial', 10),
        fg='#777', bg='#2b2b2b',
        pady=5
    )
    info_label.pack()

    # ============================================
    # Center window on screen
    # ============================================
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_reqwidth() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_reqheight() // 2)
    root.geometry(f"+{x}+{y}")

    # Start the GUI event loop
    root.mainloop()

    print()
    print("GUI window closed.")
    print("=" * 60)
    print()


# ============================================
# Launch the GUI automatically
# ============================================
print("INTERACTIVE TESTING")
print("=" * 60)
print()
print("Opening drawing/upload GUI window...")
print()
open_drawing_gui()


# ============================================
# CELL 14: Main Execution Summary
# ============================================
# Summary of the entire program execution.
# ============================================

print()
print("=" * 60)
print("PROGRAM EXECUTION COMPLETE!")
print("=" * 60)
print()
print("Summary of what we accomplished:")
print("-" * 60)
print("✓ 1. GPU Configuration")
print("     - Detected and configured GPU for acceleration")
print()
print("✓ 2. Data Loading")
print("     - Loaded 60,000 training images")
print("     - Loaded 10,000 test images")
print()
print("✓ 3. Sample Preview")
print("     - Displayed sample digits (6, 7, 8, 9)")
print()
print("✓ 4. Data Preprocessing")
print("     - Normalized pixel values (0-255 → 0-1)")
print("     - Flattened images (28×28 → 784)")
print("     - One-hot encoded labels")
print()
print("✓ 5. Model Building")
print("     - Built neural network (784→128→64→10)")
print("     - 109,386 trainable parameters")
print()
print("✓ 6. Model Compilation")
print("     - Optimizer: Adam")
print("     - Loss: Categorical Crossentropy")
print("     - Metric: Accuracy")
print()
print("✓ 7. Model Training")
print(f"     - Trained for {EPOCHS} epochs")
print("     - Batch size: 32")
print()
print("✓ 8. Model Evaluation")
print(f"     - Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"     - Test Loss: {test_loss:.4f}")
print()
print("✓ 9. Visualization")
print("     - Loss curve over epochs")
print("     - Accuracy curve over epochs")
print()
print("✓ 10. Confusion Matrix")
print("     - Identified prediction patterns and errors")
print()
print("✓ 11. Custom Prediction")
print("     - Defined functions for custom image prediction")
print()
print("=" * 60)
print()
print("Key Learnings:")
print("-" * 60)
print("1. Neural networks learn by adjusting weights to minimize loss")
print("2. Preprocessing (normalization, encoding) is critical for training")
print("3. Architecture choices affect model capacity and performance")
print("4. Adam optimizer + categorical crossentropy = good for classification")
print("5. Test accuracy measures real-world generalization")
print("6. Confusion matrix reveals systematic prediction errors")
print()
print("Next Steps:")
print("-" * 60)
print("1. Experiment with different architectures (more/fewer layers)")
print("2. Try different activation functions (tanh, LeakyReLU)")
print("3. Add dropout layers to prevent overfitting")
print("4. Implement convolutional layers (CNN) for better accuracy")
print("5. Test on your own handwritten digit images")
print()
print("=" * 60)
print("Thank you for learning with this educational implementation!")
print("=" * 60)
print()
