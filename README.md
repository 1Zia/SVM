# Breast Cancer Survival Prediction using SVM

## Project Overview

This project is developed as part of **Lab 08: Support Vector Machine**.
The objective of this project is to apply a **Support Vector Machine (SVM) classifier**
on a breast cancer dataset to predict patient survival status.
A **Streamlit frontend** is designed to interact with the trained SVM model.

---

## Dataset

- Source: Kaggle
- Link: https://www.kaggle.com/datasets/reihanenamdari/breast-cancer
- Type: Classification Dataset
- Target Variable: `Status` (Survival outcome)

The dataset contains both **numerical and categorical features**, such as age,
tumor size, cancer stage, hormone status, and survival months.

---

## Algorithm Used

**Support Vector Machine (SVM)**

- Kernel used: **Radial Basis Function (RBF)**
- SVM is used because it finds the **maximum margin hyperplane**
  and performs well on high-dimensional data.
- Feature scaling is applied using **StandardScaler**.

---

## Methodology

1. Dataset loaded from Kaggle
2. Target variable encoded
3. Categorical features converted using **One-Hot Encoding**
4. Features scaled using **StandardScaler**
5. SVM model trained using RBF kernel
6. Model and preprocessing objects saved using Pickle
7. Streamlit frontend created for prediction

---

## Files Description

- `train_model.py` → Trains SVM model and saves model files
- `app.py` → Streamlit frontend for prediction
- `data.csv` → Breast cancer dataset
- `svm_model.pkl` → Trained SVM model
- `scaler.pkl` → Feature scaler
- `features.pkl` → Feature names used during training
- `requirements.txt` → Required Python libraries

---

## How to Run the Project

### Step 1: Install Required Libraries

```bash
pip install -r requirements.txt
```
