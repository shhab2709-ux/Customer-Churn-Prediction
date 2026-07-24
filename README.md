# 📊 Customer Churn Prediction

## 📌 Project Overview

This project aims to predict whether a customer is likely to leave a telecom company (Customer Churn) using Machine Learning techniques.

By analyzing customer demographics, account information, and service usage, the model helps businesses identify customers at risk of leaving and take proactive retention actions.

---

## 🎯 Objectives

- Predict customer churn accurately.
- Perform complete data preprocessing.
- Handle missing values and outliers.
- Apply feature engineering.
- Handle class imbalance using SMOTE.
- Compare multiple machine learning algorithms.
- Tune model hyperparameters using Grid Search.
- Select the best-performing model based on evaluation metrics.

---

## 📂 Dataset

The dataset contains customer information including:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract
- Payment Method
- Monthly Charges
- Total Charges
- Churn (Target)

---

## 🛠 Data Preprocessing

The following preprocessing steps were performed:

- Data Cleaning
- Missing Value Handling
- Duplicate Removal
- Label Encoding
- One-Hot Encoding
- Feature Scaling
- Feature Engineering
- Outlier Detection
- Skewness Detection & Transformation
- Correlation Analysis
- Random Forest Feature Importance
- Handling Imbalanced Data using SMOTE

---

## 🤖 Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Gradient Boosting

---

## 🔍 Hyperparameter Tuning

Grid Search Cross Validation was used to optimize the model hyperparameters.

---

## 📈 Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve (if applicable)

---

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📁 Project Structure

```
Customer-Churn/
│
├── Customer_Churn.ipynb
├── requirements.txt
├── README.md
├── dataset.csv
└── model.pkl (optional)
```

---

## 🚀 How to Run

Clone the repository

```bash
git clone <repository-link>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the notebook

```bash
Open the project notebook (`Customer_Churn_Prediction.ipynb`) using **Jupyter Notebook** or **Google Colab**.
```

---

## 📊 Results

The project compared the performance of five machine learning models:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Gradient Boosting

Each model was evaluated using Accuracy, Precision, Recall, and F1-Score. The best-performing model was selected after comparing all evaluation metrics and hyperparameter tuning.

---

## 👥 Team Members

- Mohamed Sayed
- Salma Mohamed
- Shehab Ashraf
- Moaz Hesham
- Salma Ahmed

---

## 📜 License

This project was developed for educational purposes as part of a Machine Learning course.
