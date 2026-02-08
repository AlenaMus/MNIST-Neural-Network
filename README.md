# MNIST Handwritten Digit Recognition Neural Network

Educational implementation of a neural network that recognizes handwritten digits (0-9) using the MNIST dataset.

## Project Overview

**Goal:** Build a neural network from scratch that achieves >95% accuracy on recognizing handwritten digits.

**Purpose:** Learn fundamental deep learning concepts including:
- Data preprocessing and normalization
- Neural network architecture design
- Model training and optimization
- Performance evaluation and visualization
- Custom image prediction

**Target Audience:** Students and beginners learning machine learning and deep learning.

## Features

- **GPU Acceleration:** Automatic GPU detection and configuration
- **Comprehensive Comments:** Educational explanations of every step
- **Data Visualization:** Sample image preview, training curves, confusion matrix
- **High Accuracy:** Achieves 97%+ test accuracy
- **Custom Predictions:** Predict on your own handwritten digit images
- **Google Colab Ready:** Structured as cells for easy Colab usage

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) NVIDIA GPU with CUDA support for faster training

### Installation

1. **Clone or download the project:**
   ```bash
   cd L-36-NumbersNeuralNetwork
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   ```

### GPU Setup (Optional but Recommended)

For 10-100x faster training, set up GPU acceleration:

1. **Install NVIDIA GPU drivers:** [Download here](https://www.nvidia.com/Download/index.aspx)
2. **Install CUDA Toolkit (11.x or 12.x):** [Download here](https://developer.nvidia.com/cuda-toolkit)
3. **Install cuDNN library:** [Download here](https://developer.nvidia.com/cudnn)
4. **Verify GPU detection:**
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

If you see GPU listed, you're ready for accelerated training!

### Running the Program

**Standard Python:**
```bash
python mnist_digit_recognition.py
```

**Jupyter Notebook:**
1. Open `mnist_digit_recognition.py` in Jupyter
2. Copy each CELL section into separate notebook cells
3. Run cells sequentially

**Google Colab:**
1. Upload `mnist_digit_recognition.py` to Google Colab
2. Copy CELL sections into individual Colab cells
3. Run (GPU is free in Colab!)

## Project Structure

```
L-36-NumbersNeuralNetwork/
│
├── mnist_digit_recognition.py      # Main implementation (14 cells)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── PRD_Handwritten_Digit_Recognition.md  # Product Requirements Document
├── tasks.json                       # Development task breakdown
│
├── test_images/                     # Your custom digit images (optional)
│   └── (place your .png/.jpg images here)
│
└── results/                         # Generated outputs (created during run)
    ├── sample_digits_preview.png    # Visualization of digits 6-9
    ├── training_loss_curve.png      # Loss vs epochs graph
    ├── training_accuracy_curve.png  # Accuracy vs epochs graph
    └── confusion_matrix.png         # 10×10 confusion matrix heatmap
```

## Program Flow

The program is structured as 14 cells, each with a specific purpose:

### Cell 1: Imports and GPU Configuration
- Import all required libraries
- Detect and configure GPU acceleration
- Print GPU status and information

### Cell 2: Load MNIST Dataset
- Download and load MNIST dataset (60K training, 10K test)
- Display dataset statistics and information

### Cell 3: Preview Sample Images (Digits 6-9)
- Visualize sample digits in a 2×2 grid
- Verify data loaded correctly

### Cell 4: Data Preprocessing
- **Normalization:** Scale pixels from [0-255] to [0-1]
- **Flattening:** Reshape 28×28 images to 784-element vectors
- **One-Hot Encoding:** Convert labels to categorical format

### Cell 5: Build Neural Network Architecture
- Input layer: 784 neurons (flattened image)
- Hidden layer 1: 128 neurons (ReLU activation)
- Hidden layer 2: 64 neurons (ReLU activation)
- Output layer: 10 neurons (Softmax activation)
- Display model summary (109,386 parameters)

### Cell 6: Compile Model
- **Optimizer:** Adam (adaptive learning rate)
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- Detailed explanations of each choice

### Cell 7: Train the Model
- Train for 10 epochs with batch size 32
- Monitor training and validation performance
- Display epoch-by-epoch progress

### Cell 8: Evaluate Model
- Evaluate on 10,000 test images
- Report test accuracy and loss
- Check for overfitting

### Cell 9: Plot Training History (Loss Curve)
- Visualize loss vs epochs
- Compare training and validation loss

### Cell 10: Plot Training History (Accuracy Curve)
- Visualize accuracy vs epochs
- Monitor learning progress

### Cell 11: Confusion Matrix
- Generate 10×10 confusion matrix
- Identify which digits get confused
- Display as color-coded heatmap

### Cell 12: Custom Image Prediction Functions
- `preprocess_image()`: Prepare custom image for prediction
- `predict_digit()`: Predict digit and confidence
- `display_prediction()`: Visualize results

### Cell 13: Interactive Prediction Loop
- Test model on multiple custom images
- User-friendly interactive mode

### Cell 14: Main Execution Summary
- Summary of all accomplishments
- Key learnings and next steps

## Model Architecture

```
Input Layer:    784 neurons  (28×28 flattened image)
                   ↓
Hidden Layer 1: 128 neurons  (ReLU activation)
                   ↓
Hidden Layer 2:  64 neurons  (ReLU activation)
                   ↓
Output Layer:    10 neurons  (Softmax activation)

Total Parameters: 109,386 trainable parameters
```

**Why this architecture?**
- **Fully Connected:** Simple, educational, effective for MNIST
- **Funnel Shape (784→128→64→10):** Progressive feature compression
- **ReLU Activation:** Fast, prevents vanishing gradients
- **Softmax Output:** Perfect for multi-class classification

## Performance Metrics

**Expected Results:**
- Training Accuracy: >98%
- Test Accuracy: >97%
- Training Time: <1 minute on GPU, ~2-5 minutes on CPU
- Loss: <0.1 (after 10 epochs)

**Acceptance Criteria:**
- Test accuracy >= 95% ✓
- Well-documented code ✓
- Clear visualizations ✓
- Reproducible results ✓

## Understanding the Loss Function

**Categorical Crossentropy** measures prediction error:

```
Formula: Loss = -Σ(y_true × log(y_pred))

Example 1: Good Prediction
True label: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Predicted:      [0.01, 0.02, 0.05, 0.85, ...]
                                   ↑ 85% confident it's a 3
Loss = -log(0.85) = 0.163  ← LOW (good!)

Example 2: Bad Prediction
True label: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Predicted:      [0.85, 0.02, 0.05, 0.01, ...]
                 ↑ 85% confident it's a 0 (wrong!)
                                   ↑ 1% for correct class
Loss = -log(0.01) = 4.605  ← HIGH (bad!)
```

**Key Insight:** Loss increases exponentially when model is confidently wrong!

## Understanding the Optimizer

**Adam (Adaptive Moment Estimation)** is the best general-purpose optimizer:

**How it works:**
- Maintains moving averages of gradients (momentum)
- Adapts learning rate for each parameter individually
- Combines benefits of SGD + Momentum + RMSprop

**Advantages:**
- Fast convergence
- Works out-of-the-box (no tuning needed)
- Handles sparse gradients well
- Memory efficient

**Default parameters (used):**
- Learning rate: 0.001
- Beta_1 (momentum): 0.9
- Beta_2 (variance): 0.999

## Custom Image Prediction

Test the model on your own handwritten digits:

### Prepare Your Image
1. Draw a digit (0-9) on paper or digitally
2. Take a photo or screenshot
3. Recommended format: White digit on black background (MNIST style)
4. Save as PNG or JPG

### Make Prediction

**Option 1: Single Image**
```python
digit, confidence, probabilities = predict_digit(model, 'test_images/my_digit.png')
display_prediction('test_images/my_digit.png', digit, confidence, probabilities)
```

**Option 2: Interactive Mode**
```python
interactive_prediction_mode()
# Follow prompts to test multiple images
```

### Tips for Best Results
- Clear, single digit (not multiple digits)
- Centered in image
- Good contrast (dark vs light)
- Similar to MNIST style (simple, handwritten)

## Troubleshooting

### GPU Not Detected
**Problem:** Program shows "NO GPU DETECTED"

**Solutions:**
1. Install NVIDIA GPU drivers
2. Install CUDA Toolkit (check TensorFlow compatibility)
3. Install cuDNN library
4. Verify installation: `nvidia-smi` (should show GPU info)
5. Reinstall TensorFlow: `pip install tensorflow[and-cuda]`

**Workaround:** Program runs on CPU (slower but works)

### Low Accuracy (<90%)
**Problem:** Test accuracy below expected

**Solutions:**
1. Train for more epochs (try 15-20)
2. Check data preprocessing (normalization, one-hot encoding)
3. Verify model architecture matches specification
4. Ensure validation split is used (prevents overfitting)

### Memory Errors
**Problem:** "Out of memory" errors

**Solutions:**
1. Reduce batch size (try 16 or 8)
2. Enable GPU memory growth (done automatically in code)
3. Close other programs using GPU memory
4. Use CPU if GPU memory insufficient

### Custom Prediction Fails
**Problem:** Custom image prediction returns wrong digit

**Solutions:**
1. Invert colors (uncomment line in `preprocess_image()`)
2. Ensure image is 28×28 or will be resized properly
3. Check image is grayscale (not RGB)
4. Use clear, centered digit similar to MNIST

## Learning Resources

### Concepts Explained in Code
- Neural network architecture
- Forward propagation
- Backpropagation (automatic via TensorFlow)
- Loss functions (categorical crossentropy)
- Optimizers (Adam)
- Activation functions (ReLU, Softmax)
- One-hot encoding
- Normalization
- Overfitting vs generalization
- Confusion matrices

### Next Steps
1. **Experiment with hyperparameters:**
   - Try different learning rates
   - Change number of neurons (32, 64, 256, 512)
   - Add/remove layers
   - Try different batch sizes

2. **Improve the model:**
   - Add Dropout layers (prevent overfitting)
   - Try different activation functions (tanh, LeakyReLU)
   - Implement early stopping
   - Add batch normalization

3. **Upgrade to CNN:**
   - Replace Dense layers with Conv2D layers
   - Add MaxPooling layers
   - Achieve 98-99%+ accuracy

4. **Try other datasets:**
   - Fashion MNIST (clothing items)
   - CIFAR-10 (color images)
   - Your own custom dataset

### Recommended Reading
- **Deep Learning Specialization** (Andrew Ng) - Coursera
- **Neural Networks and Deep Learning** (Michael Nielsen) - Free online book
- **Hands-On Machine Learning** (Aurélien Géron) - Comprehensive guide
- **TensorFlow Documentation** - Official tutorials

## Educational Value

This project teaches:
- ✓ End-to-end ML workflow (data → model → evaluation)
- ✓ Neural network fundamentals
- ✓ Best practices in code documentation
- ✓ Data preprocessing techniques
- ✓ Model evaluation methods
- ✓ Visualization of ML results
- ✓ GPU acceleration setup
- ✓ Real-world application (digit recognition)

## Author & License

**Author:** AI Development Course - Lesson 36

**Purpose:** Educational implementation for learning neural networks

**Date:** February 8, 2026

**License:** Free to use for educational purposes

## Acknowledgments

- **MNIST Dataset:** Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- **TensorFlow/Keras:** Google Brain Team
- **Inspiration:** Classic "Hello World" of deep learning

## Support & Questions

For questions or issues:
1. Review the comprehensive comments in `mnist_digit_recognition.py`
2. Check the troubleshooting section above
3. Refer to the PRD document for detailed requirements
4. Consult TensorFlow documentation for API details

---

**Happy Learning!** 🚀

Start your journey into deep learning with this hands-on, well-documented implementation of handwritten digit recognition.
