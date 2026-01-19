# Custom Image Classification using Transfer Learning

## 📌 Project Overview
This project implements a high-performance image classification system using **transfer learning** with a pre-trained **ResNet50** convolutional neural network.  
The goal is to demonstrate how leveraging pre-trained models significantly improves accuracy and training efficiency compared to training a CNN from scratch.

The task includes:
- Data preprocessing and augmentation
- Baseline CNN training
- Transfer learning with two-phase training
- Model evaluation and interpretability using Grad-CAM

---

## 📂 Dataset
- **Dataset:** Dogs vs Cats (Kaggle)
- **Classes:** Cat, Dog
- **Split Strategy:**
  - Training: 70%
  - Validation: 15%
  - Test: 15%

Images are resized to **224×224** to match ResNet50 input requirements.

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing & Augmentation
- Images are loaded from directory structure (`train/`, `val/`, `test/`)
- **Training data augmentation:**
  - Random rotations
  - Width/height shifts
  - Zoom
  - Horizontal flip
- **ResNet50-specific preprocessing** (`preprocess_input`) is applied for transfer learning

---

### 2️⃣ Baseline CNN (From Scratch)
A simple CNN was trained from scratch to establish a baseline performance.

**Architecture highlights:**
- 3 convolutional layers
- MaxPooling layers
- Fully connected dense layers with dropout

**Baseline Performance:**
- **Test Accuracy:** **81.09%**
- Training required significant time on CPU due to random weight initialization

This baseline helps quantify the benefit of transfer learning.

---

### 3️⃣ Transfer Learning with ResNet50
A pre-trained **ResNet50** model (ImageNet weights) was used.

#### 🔹 Phase 1: Feature Extraction
- ResNet50 convolutional base **frozen**
- Custom classification head trained
- Learning rate: `1e-4`

#### 🔹 Phase 2: Fine-Tuning
- Top layers of ResNet50 unfrozen
- Very small learning rate: `1e-5`
- Prevents catastrophic forgetting

**Final Transfer Learning Performance:**
- **Test Accuracy:** **99.07%**

This clearly outperforms the baseline CNN.

---


### 🔹 Why ResNet50?
ResNet50 was chosen due to its deep residual architecture, which effectively mitigates the vanishing gradient problem and provides strong feature representations pre-trained on ImageNet. This makes it highly suitable for transfer learning on smaller custom datasets.


## 📊 Model Evaluation

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1-score

A detailed classification report was generated on the test set.

### Confusion Matrix
The confusion matrix shows near-perfect classification with minimal misclassifications.

(See: `gradcam/confusion_matrix.png`)

---

## 🔍 Model Interpretability (Grad-CAM)
Grad-CAM was applied to visualize which regions of the image the model focuses on while making predictions.

- Heatmaps highlight relevant object regions (dog or cat)
- Demonstrates that the model learns meaningful visual features

(See: `gradcam/gradcam_sample.png`)

---

## 📈 Performance Comparison

| Model | Test Accuracy |
|-----|---------------|
| Baseline CNN | 81.09% |
| ResNet50 (Transfer Learning) | **99.07%** |

**Conclusion:** Transfer learning significantly improves both accuracy and training efficiency.

---

## 🛠️ Tech Stack

- Python 3.10
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies

pip install -r requirements.txt

### 2️⃣ Train Baseline CNN

python -m src.baseline_cnn

### 3️⃣ Train Transfer Learning Model

python -m src.transfer_model

### 4️⃣ Evaluate Model

python -m src.evaluate

### 5️⃣ Generate Grad-CAM

python -m src.gradcam

## 📁 Project Structure

Custom_Image_Classifier/

├── data/
├── models/
├── gradcam/
├── src/
│   ├── data_loader.py
│   ├── baseline_cnn.py
│   ├── transfer_model.py
│   ├── evaluate.py
│   └── gradcam.py
├── requirements.txt
├── README.md
└── split_dataset.py

## ✅ Key Learnings
Transfer learning drastically improves performance on limited data

Proper preprocessing is critical for pre-trained models

Fine-tuning with low learning rates prevents overfitting

Grad-CAM improves model interpretability and trust

## 🔁 Reproducibility
All experiments can be reproduced using the provided scripts and `requirements.txt`. The dataset is excluded from the repository and must be downloaded separately due to size constraints.

## 🧾 Conclusion
This project demonstrates an end-to-end transfer learning workflow, from data preprocessing to evaluation and interpretability.
The results clearly validate the effectiveness of transfer learning over training from scratch.

---

