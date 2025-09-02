---
Exp - 4

### 🎯 Aim

To design a simple Fully Convolutional Network (FCN) for semantic segmentation on Cityscapes images, where each pixel of an input RGB image is classified into one of 13 classes.

---

### ⚙️ Procedure

1. **Data Preparation**

   * Each image is split: left half → RGB input, right half → grayscale mask.
   * Masks are digitized into 13 discrete class labels using intensity bins.
   * Images and masks are resized to $128 \times 128$.

2. **Model Architecture**

   * Encoder: stacked **Conv2D + MaxPooling** layers extract spatial features.
   * Bottleneck: deeper **Conv2D** captures high-level context.
   * Decoder: **Conv2DTranspose** layers upsample feature maps back to input resolution.
   * Output: final **softmax layer** gives per-pixel class probabilities.

3. **Training**

   * Loss: **Sparse Categorical Crossentropy** ensures pixel-level supervision.
   * Optimizer: **Adam** for adaptive learning.
   * Training performed for multiple epochs with validation monitoring.

4. **Prediction & Visualization**

   * Model predicts per-pixel class labels for unseen validation images.
   * Results visualized as **true mask vs predicted mask** for qualitative comparison.

---

### 📊 Result

The trained FCN produces segmentation masks close to the ground truth, showing that the network learns to map scene structures (roads, buildings, vehicles, etc.) into correct class regions. Accuracy improves with more training epochs and larger datasets.

---

