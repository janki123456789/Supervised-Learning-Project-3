# 🚨 Risk Alert Classifier using Machine Learning

# 📌 Project Overview

The **Risk Alert Classifier** is a machine learning project designed to identify whether a customer belongs to:

- **0 → Low Risk**
- **1 → High Risk**

The project uses customer demographic information, transaction activity indicators, and credit behavior indicators to predict customer risk status.

The primary objective is to accurately identify **high-risk customers** while minimizing **False Negatives**, which is a critical requirement in risk prediction systems.

---

# 🎯 Objectives

- Build a baseline Logistic Regression model.
- Handle class imbalance using multiple resampling techniques.
- Implement Tree-Based Classification models.
- Perform Hyperparameter Tuning.
- Evaluate models using ROC-AUC and other classification metrics.
- Select the best model based on business requirements.

---

# 📂 Dataset

The dataset contains:

### 👤 Customer Demographic Information
- Age
- Gender
- Region
- Employment Type

### 💳 Transaction Activity Indicators
- Monthly Spend
- Monthly Transaction Count
- Failed Login Attempts
- Complaints
- Account Tenure

### 📈 Credit Behavior Indicators
- Credit Score
- Credit Utilization Ratio
- Missed Payments
- Average Late Payment Days
- Debt Balance

### 🎯 Target Variable

| Value | Risk Status |
|---------|-------------|
| 0 | Low Risk |
| 1 | High Risk |

---

# 🔗 Dataset Link

- [Risk_Alert_Classifier_Dataset.csv](Risk_Alert_Classifier_Dataset.csv)
---
# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn
- Jupyter Notebook

---

# 📋 Project Workflow

## Part A: Conceptual Understanding

- Logistic Regression
- Classification Metrics
- Type-I and Type-II Errors
- Precision, Recall, F1-Score
- ROC Curve and AUC
- Class Imbalance Problems

---

## Part B: Dataset Preparation

- Feature Selection
- Train-Test Split
- Missing Value Detection
- KNN Imputation

---

## Part C: Baseline Classification Model

### Logistic Regression

Evaluation Metrics:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-Score

---

## Part D: Handling Imbalanced Data

Implemented:

### 🔹 Under Sampling

Balances the classes by reducing majority class samples.

### 🔹 Over Sampling

Duplicates minority class samples.

### 🔹 SMOTE

Generates synthetic samples using nearest neighbors.

### 🔹 ADASYN

Creates adaptive synthetic data for difficult minority instances.

---

## Part E: Tree-Based Models

### 🌳 Decision Tree Classifier

- Training Accuracy
- Testing Accuracy
- Overfitting Analysis

### 🌲 Random Forest Classifier

- Improved Generalization
- Better Performance
- Reduced Overfitting

---

## Part F: Hyperparameter Tuning

### Randomized Search CV

Optimized:

- Decision Tree Hyperparameters
- Random Forest Hyperparameters

### Grid Search CV

Fine-tuned the best-performing model to obtain optimal performance.

---

## Part G: Model Evaluation

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC-ROC Score

---

# 📊 Models Implemented

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Randomized Search CV
- Grid Search CV

---

# 📈 Performance Comparison

Models were compared using:

- Recall for Minority Class
- F1-Score
- Accuracy
- AUC-ROC Score

Special emphasis was placed on minimizing **False Negatives**, as missing a High-Risk customer can have significant business consequences.

---

# 🏆 Best Model

After comparing multiple models and tuning techniques, the best model was selected based on:

- Highest Recall
- Better Generalization
- Strong AUC-ROC Performance
- Lower False Negatives

---

# 📁 Repository Structure

```bash
.
│
├── PR-3.ipynb
├── README.md
├── Risk_Alert_Classifier_Dataset.csv
└── requirements.txt

---

# 📌 Business Interpretation

### False Positive

A low-risk customer is incorrectly classified as high-risk.

### False Negative

A high-risk customer is incorrectly classified as low-risk.

Since false negatives are more dangerous in risk prediction systems, the final model was selected to minimize them.

---

# 📚 Libraries Used

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

---

# 👨‍💻 Author

Janki Dholariya


