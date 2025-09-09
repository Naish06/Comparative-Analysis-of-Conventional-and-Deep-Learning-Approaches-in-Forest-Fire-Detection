# Comparative-Analysis-of-Conventional-and-Deep-Learning-Approaches-in-Forest-Fire-Detection
## Forest Fire Detection using Machine Learning and Deep Learning

This repository contains an academic assignment on **forest fire image classification**.  
The project explores **traditional feature extraction + Random Forest** classifiers, a **novel hybrid approach**, and a **deep learning pipeline (ResNet50 + Random Forest)**.  
The aim is to compare models in terms of accuracy, robustness, error tendencies, and fairness metrics.

---

## Dataset
- Custom dataset with **6,248 images**, divided into:
  - **Fire** (forest fire scenes, with slight augmentation)
  - **NoFire** (forest/greenery scenes, with augmentation)
- The dataset was split into **train** and **test** sets.
- An additional **Gaussian noise variant** was created to evaluate robustness.

---

## Methods Implemented
1. **Traditional Feature Extraction + Random Forest**
   - HOG + RF
   - SIFT + RF
   - GLCM + RF
   - ORB + RF
2. **Novel Hybrid Feature Extraction + RF**
3. **Deep Learning Feature Extraction**
   - ResNet50 (pre-trained on ImageNet) + Random Forest

---

## Evaluation Metrics
- **Standard Metrics:** Accuracy, Precision, Recall, F1-score  
- **Advanced Metrics:** Confusion Matrix, ROC-AUC, Class-wise Error Analysis, Fairness Metrics (Error Rate Balance)  

---

## Running the Code on Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Mount Google Drive (dataset should be stored in `MyDrive/forestfire/`).
3. Upload the respective notebook from this repo.
4. Run all cells sequentially.

Example for mounting dataset:
```python
from google.colab import drive
drive.mount('/content/drive')
base_dir = '/content/drive/MyDrive/forestfire/'
