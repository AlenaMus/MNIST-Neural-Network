# MNIST Neural Network - Quick Reference Card

One-page reference for the MNIST handwritten digit recognition program.

## 30-Second Overview

**What:** Neural network that recognizes handwritten digits (0-9)
**Dataset:** MNIST - 70,000 images (60K training, 10K test)
**Accuracy:** 97%+ on test set
**Training Time:** <1 minute on GPU

## Installation (2 minutes)

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn Pillow
python mnist_digit_recognition.py
```

## Architecture at a Glance

```
Input:    784 neurons  ← Flattened 28×28 image
   ↓
Hidden 1: 128 neurons  (ReLU)
   ↓
Hidden 2:  64 neurons  (ReLU)
   ↓
Output:    10 neurons  (Softmax) ← Probabilities for digits 0-9

Parameters: 109,386
```

## The 14 Cells

| Cell | What It Does | Time | Key Output |
|------|--------------|------|------------|
| 1 | Import libraries, check GPU | 10s | GPU detected: Yes/No |
| 2 | Load MNIST dataset | 15s | 60,000 train, 10,000 test |
| 3 | Preview digits 6-9 | 2s | 2×2 image grid |
| 4 | Preprocess data | 1s | Normalize, flatten, one-hot |
| 5 | Build model | 1s | 784→128→64→10 |
| 6 | Compile model | <1s | Adam + Categorical Crossentropy |
| 7 | Train model | 45s | 10 epochs, 99% train accuracy |
| 8 | Evaluate model | 2s | 97%+ test accuracy |
| 9 | Loss curve | 1s | Graph: loss vs epochs |
| 10 | Accuracy curve | 1s | Graph: accuracy vs epochs |
| 11 | Confusion matrix | 3s | 10×10 heatmap |
| 12 | Prediction functions | <1s | Load custom images |
| 13 | Interactive mode | - | Test your images |
| 14 | Summary | <1s | Recap & next steps |

## Key Formulas

### Loss Function (Categorical Crossentropy)
```
Loss = -Σ(y_true × log(y_pred))

Good prediction: Loss = -log(0.85) = 0.163  (LOW)
Bad prediction:  Loss = -log(0.01) = 4.605  (HIGH)
```

### Accuracy
```
Accuracy = Correct Predictions / Total Predictions

Example: 9,742 correct out of 10,000 = 97.42%
```

## Data Preprocessing Checklist

- [ ] **Normalize:** Divide pixels by 255.0 → [0, 1] range
- [ ] **Flatten:** Reshape (28, 28) → (784,)
- [ ] **One-Hot:** Convert label 3 → [0,0,0,1,0,0,0,0,0,0]

## Training Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Epochs | 10 | Full passes through data |
| Batch Size | 32 | Samples per weight update |
| Optimizer | Adam | Best general-purpose optimizer |
| Loss | Categorical Crossentropy | For multi-class classification |
| Learning Rate | 0.001 | Adam default (auto-adjusted) |
| Validation Split | 10% | Monitor overfitting |

## Expected Performance

```
Epoch 1:  ~93% accuracy
Epoch 2:  ~96% accuracy
Epoch 5:  ~98% accuracy
Epoch 10: ~99% accuracy

Final Test Accuracy: 97-98%
```

## Interpreting Graphs

### Loss Curve
✓ **Good:** Both lines decreasing, close together
✗ **Bad:** Training << Validation (overfitting)

### Accuracy Curve
✓ **Good:** Both lines increasing, converging
✗ **Bad:** Training >> Validation (overfitting)

### Confusion Matrix
- **Diagonal (dark):** Correct predictions
- **Off-diagonal (light):** Errors
- **Look for:** Strong diagonal, weak off-diagonal

## Common Confusions

Digits often confused by the model:
- 4 ↔ 9 (similar top parts)
- 3 ↔ 8 (similar curves)
- 5 ↔ 6 (similar shapes)
- 7 ↔ 1 (similar strokes)

## Custom Image Prediction

```python
# 1. Prepare image (28×28, grayscale, white digit on black background)
# 2. Save as PNG/JPG in test_images/

# 3. Predict
digit, confidence, probs = predict_digit(model, 'test_images/my_digit.png')
display_prediction('test_images/my_digit.png', digit, confidence, probs)

# Output: Predicted: 7 (Confidence: 98.5%)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No GPU detected | Install CUDA/cuDNN or use CPU |
| Low accuracy (<90%) | Train more epochs, check preprocessing |
| Out of memory | Reduce batch size to 16 |
| Custom prediction wrong | Invert colors, check 28×28 resize |

## File Locations

```
mnist_digit_recognition.py  ← Main program
requirements.txt            ← Dependencies
README.md                  ← Full documentation
COLAB_GUIDE.md             ← Google Colab instructions

test_images/               ← Your custom images
results/                   ← Generated graphs/matrices
```

## Modification Ideas

### Easy
```python
EPOCHS = 20              # Train longer
BATCH_SIZE = 64          # Larger batches
```

### Medium
```python
Dense(256, ...),         # More neurons
Dense(128, ...),
Dense(64, ...),
```

### Advanced
```python
from tensorflow.keras.layers import Dropout

Dense(128, activation='relu'),
Dropout(0.2),            # Prevent overfitting
Dense(64, activation='relu'),
Dropout(0.2),
```

## Google Colab Quick Start

1. Upload `mnist_digit_recognition.py` to Colab
2. **Runtime → Change runtime type → GPU**
3. Copy each CELL into separate Colab cells
4. Run all cells (Runtime → Run all)
5. Watch it achieve 97%+ accuracy!

## Key Takeaways

✓ **Preprocessing matters:** Normalization is critical
✓ **Architecture choice:** Funnel shape works well
✓ **ReLU for hidden layers:** Prevents vanishing gradients
✓ **Softmax for output:** Perfect for classification
✓ **Adam optimizer:** Works out-of-the-box
✓ **Test on unseen data:** Measures real performance
✓ **Visualize training:** Spot overfitting early
✓ **Confusion matrix:** Understand errors

## Learning Path

1. **Understand code:** Read all comments in each cell
2. **Run successfully:** Achieve 95%+ accuracy
3. **Experiment:** Modify parameters, observe changes
4. **Test custom images:** Draw your own digits
5. **Try variations:** Different architectures
6. **Next level:** Implement CNN, try other datasets

## Resources

- **Code:** `mnist_digit_recognition.py` (1,100 lines, 70% comments)
- **Full docs:** `README.md`
- **Colab guide:** `COLAB_GUIDE.md`
- **Task mapping:** `IMPLEMENTATION_SUMMARY.md`
- **Requirements:** PRD_Handwritten_Digit_Recognition.md

## Success Metrics

- [x] Code runs without errors
- [x] Test accuracy >= 95% (target: 97%+)
- [x] Training time < 1 minute (GPU)
- [x] All visualizations display
- [x] Custom prediction works
- [x] Can explain each step

## Quick Commands

```bash
# Install
pip install -r requirements.txt

# Run
python mnist_digit_recognition.py

# Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Interactive mode (in script)
# Uncomment: interactive_prediction_mode()
```

## Keyboard Shortcuts (Colab)

- `Ctrl + Enter` - Run cell
- `Shift + Enter` - Run cell, move next
- `Ctrl + M B` - Insert cell below
- `Ctrl + /` - Comment line

## Neural Network Cheat Sheet

**Activation Functions:**
- ReLU: max(0, x) - Hidden layers
- Softmax: exp(x)/Σexp(x) - Output (classification)

**Layer Types:**
- Dense: Fully connected
- Dropout: Regularization
- Conv2D: Feature extraction (advanced)

**Optimizers:**
- Adam: Best general (use this)
- SGD: Simple but slower
- RMSprop: Good alternative

**Loss Functions:**
- Categorical Crossentropy: Multi-class (>2 classes)
- Binary Crossentropy: Binary (2 classes)
- MSE: Regression (continuous values)

---

**Print this page for quick reference while coding!**
