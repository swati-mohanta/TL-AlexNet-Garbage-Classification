# Transfer Learning using AlexNet for Garbage Classification

This project demonstrates the application of **Transfer Learning (TL)** using **AlexNet**, a pre-trained deep learning model, for **Garbage Classification**. The experiment focuses on fine-tuning different layers of AlexNet and comparing their performance on a custom dataset.

---

## Dataset

The dataset used in this project is the **[Garbage Classification Dataset](https://www.kaggle.com/datasets/alincijov/self-driving-cars)**.
It contains labeled images categorized into **recyclable** and **trash** classes.

* **Classes:**

  * Recyclable
  * Trash

* The dataset is divided into:

  * **70% Training data**
  * **15% Validation data**
  * **15% Testing data**

---

## Requirements

Install the required libraries before running the notebook:

```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

---

## Model Overview

**Base Model:** AlexNet (Pre-trained on ImageNet)
**Approach:** Transfer Learning by unfreezing specific fully connected (FC) layers.

Two experiments were conducted:

| Case       | Layers Trained     | Description                                     |
| ---------- | ------------------ | ----------------------------------------------- |
| **Case 1** | Last FC layer only | Fine-tuning the final classification layer      |
| **Case 2** | Last two FC layers | Fine-tuning the last two fully connected layers |

---

## Implementation Steps

1. **Load and preprocess the dataset**

   * Images resized to `227x227` (AlexNet input size)
   * Converted to PyTorch tensors

2. **Split dataset** into training, validation, and test sets

3. **Define DataLoaders** for efficient batching

4. **Transfer Learning Setup**

   * Load AlexNet pretrained on ImageNet
   * Freeze early layers
   * Replace and unfreeze desired FC layers

5. **Train and Evaluate Model**

   * Use **Adam optimizer** and **CrossEntropyLoss**
   * Evaluate using accuracy and classification report

6. **Visualize results**

   * Compare model performance between both cases

---

## Results

### 🔹 Case 1: Train Last FC Layer Only

```
Accuracy: 95.26%
Precision/Recall (Recyclable): 0.95 / 1.00
Precision/Recall (Trash): 0.75 / 0.15
```

### 🔹 Case 2: Train Last Two FC Layers

```
Accuracy: 91.05%
Precision/Recall (Recyclable): 0.97 / 0.93
Precision/Recall (Trash): 0.31 / 0.55
```

### Comparison Plot

A bar plot compares the accuracy between both cases:

```
Case 1: 95.26%
Case 2: 91.05%
```

---

## Sample Outputs

During training, sample images from the dataset are displayed with their corresponding labels for visual verification.

---

## Insights

* Freezing most of the layers and training only the last FC layer performs **better** for small datasets.
* Overfitting risk increases as more layers are unfrozen.
* AlexNet’s pretrained features are highly transferable for image classification tasks like garbage detection.

---

## Future Improvements

* Experiment with **data augmentation** to reduce overfitting
* Try **other CNN architectures** (ResNet, VGG16, MobileNet)
* Use **Grad-CAM** for visualizing model focus areas
