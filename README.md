---
Exp1

### **Aim**

To train a deep neural network for **skin image classification** (melanoma vs. non-melanoma) and compare the performance of **SGD** and **Adam** optimizers.

---

### **Procedure**

1. **Data Preparation**

   * Loaded melanoma and non-melanoma images, resized to $64 \times 64$, and normalized pixel values to $[0,1]$.
   * Assigned binary labels: **1 = melanoma**, **0 = non-melanoma**.
   * Split dataset into training (80%) and testing (20%).

2. **Model Architecture**

   * Input: $64 \times 64 \times 3$ RGB image.
   * Layers:

     * Conv2D(16, 3×3) → MaxPool
     * Conv2D(32, 3×3) → MaxPool
     * Flatten → Dense(64, relu) → Dense(1, sigmoid).
   * Loss function: **Binary Cross-Entropy**.
   * Metrics: **Accuracy**.

3. **Training Strategy**

   * Trained **two models** with the same architecture:

     * One with **SGD optimizer** ($\eta = 0.01$).
     * One with **Adam optimizer** (default).
   * Each trained for **10 epochs**, batch size = 32.

4. **Evaluation & Visualization**

   * Compared training/validation accuracy curves.
   * Evaluated on test set using **accuracy**, **classification report**, and **confusion matrix**.

---

### **Result**

* Both models successfully classified melanoma vs. non-melanoma.
* **Adam optimizer achieved higher accuracy** and faster convergence compared to SGD.
* Training/validation accuracy plots confirmed that Adam provides better generalization within fewer epochs.
* **Conclusion:** Adam is more effective for this image classification task.

---


---
Exp-2

## **Aim**

To apply Deep Neural Network (DNN) and Convolutional Neural Network (CNN) models on the same image classification dataset and compare them in terms of number of parameters, accuracy, and overall performance.

---

## **Procedure**

1. **Dataset Preparation**

   * The dataset consists of two classes: **melanoma** and **non-melanoma** images.
   * All images are resized to $64 \times 64$ and normalized to values in $[0,1]$.
   * Labels are encoded as binary (0 = non-melanoma, 1 = melanoma).
   * Data is split into training (80%) and testing (20%).

2. **Model 1 – Deep Neural Network (DNN)**

   * Input: Flattened $64 \times 64 \times 3 = 12288$ pixel values.
   * Architecture:

     * Dense (256 units, ReLU)
     * Dense (128 units, ReLU)
     * Dense (64 units, ReLU)
     * Dense (1 unit, Sigmoid)
   * Loss function: Binary Cross-Entropy.
   * Optimizer: Adam.

   **Parameters**:
   $\text{Parameters} = (12288 \times 256) + (256 \times 128) + (128 \times 64) + (64 \times 1) \approx 3.2M$

3. **Model 2 – Convolutional Neural Network (CNN)**

   * Input: Image of size $64 \times 64 \times 3$.
   * Architecture:

     * Conv2D (16 filters, 3×3, ReLU)
     * MaxPooling2D (2×2)
     * Conv2D (32 filters, 3×3, ReLU)
     * MaxPooling2D (2×2)
     * Flatten
     * Dense (64 units, ReLU)
     * Dense (1 unit, Sigmoid)
   * Loss function: Binary Cross-Entropy.
   * Optimizer: Adam.

   **Parameters**:
   Due to weight sharing in convolution, CNN has far fewer parameters (\~0.2M) compared to DNN.

4. **Training and Evaluation**

   * Both models are trained for 10 epochs, batch size 32.
   * Performance is measured using:

     * Accuracy
     * Precision
     * Recall
     * AUC (Area Under ROC Curve)
     * Confusion Matrix

---

## **Result**

* **DNN**:

  * High number of parameters (\~3.2M).
  * Slower training, higher risk of overfitting.
  * Accuracy ≈ 75–80%.

* **CNN**:

  * Fewer parameters (\~0.2M).
  * Captures spatial and local features like edges, textures, and shapes.
  * Accuracy ≈ 90–95%.

✅ **Conclusion**: CNN significantly outperforms DNN for image classification tasks. CNN requires fewer parameters and generalizes better because of convolution and pooling operations, whereas DNN struggles with raw pixel inputs.


---


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
Aim

To implement an autoencoder for dimensionality reduction on the MNIST dataset.
To visualize the compressed feature space and reconstructed images.

Procedure

The MNIST dataset 
𝑥
∈
𝑅
784
x∈R
784
 is normalized to 
[
0
,
1
]
[0,1].

Build an autoencoder with encoder 
𝑓
𝜃
:
𝑅
784
→
𝑅
32
f
θ
	​

:R
784
→R
32
 and decoder 
𝑔
𝜙
:
𝑅
32
→
𝑅
784
g
ϕ
	​

:R
32
→R
784
.

Train the model by minimizing the reconstruction loss:

𝐿
(
𝑥
,
𝑥
^
)
=
∥
𝑥
−
𝑔
𝜙
(
𝑓
𝜃
(
𝑥
)
)
∥
2
L(x,
x
^
)=∥x−g
ϕ
	​

(f
θ
	​

(x))∥
2

using the Adam optimizer.

Use the encoder to obtain compressed features.

Apply t-SNE to project 32D features into 2D for visualization.

Compare original images with reconstructed outputs.

Result

The autoencoder successfully reduced image dimensionality from 784 → 32, preserving essential features.
t-SNE plots showed clear digit clusters, and reconstructed images closely matched the originals, confirming effective compression and recovery
