# 🎓 Student Performance Tracking System Using XGBoost

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

# 📖 Overview

The **Student Performance Tracking System** is a Machine Learning-based application designed to predict whether a student is likely to **Pass** or **Fail** based on academic, demographic, and behavioral factors.

Educational institutions generate large volumes of student data every year. However, extracting meaningful insights from this data remains a challenge. This project utilizes the **XGBoost Classification Algorithm**, one of the most powerful ensemble learning techniques, to analyze student information and generate performance predictions.

By identifying students at risk of academic failure early, educators can provide timely interventions, improve learning outcomes, and enhance overall student success.

---

# 🎯 Problem Statement

Educational institutions often rely on examinations and manual observation to evaluate student performance. These traditional approaches may fail to identify struggling students at an early stage.

The goal of this project is to build an intelligent prediction system capable of:

* Analyzing student-related factors.
* Predicting academic outcomes.
* Classifying students as Pass or Fail.
* Supporting data-driven educational decisions.
* Enabling early intervention strategies.

---

# 🚀 Project Objectives

The primary objectives of this project are:

* Predict student academic performance using Machine Learning.
* Identify students who may require additional support.
* Determine factors that influence academic success.
* Improve educational decision-making through predictive analytics.
* Assist institutions in monitoring student progress efficiently.

---

# 🛠 Technologies Used

## Programming Language

* Python 3.x

## Libraries

* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Joblib
* Matplotlib
* Seaborn

---

# 📂 Dataset Description

The dataset contains academic, demographic, and social information related to students.

### Features Include

* Age
* Gender
* Study Time
* Family Background
* School Support
* Attendance Records
* Previous Grades
* Social Activities
* Educational Factors

### Target Variable

The final grade (**G3**) is used to create the target variable.

| Final Grade (G3) | Performance |
| ---------------- | ----------- |
| G3 ≥ 10          | Pass (1)    |
| G3 < 10          | Fail (0)    |

This transforms the problem into a binary classification task.

---

# 🔄 Project Workflow

## 1. Data Collection

The dataset is loaded into a Pandas DataFrame for processing and analysis.

---

## 2. Data Exploration

The dataset is explored to understand:

* Number of records
* Available features
* Data types
* Missing values
* Dataset structure

---

## 3. Data Cleaning

To improve data quality:

### Numerical Features

Missing values are replaced using:

* Median Imputation

### Categorical Features

Missing values are replaced using:

* Mode Imputation

---

## 4. Feature Encoding

Machine Learning models require numerical input.

Categorical features such as:

* Gender
* Family Status
* School Support

are converted into numerical values using **Label Encoding**.

---

## 5. Target Variable Creation

A new column named **Performance** is created:

```python
Performance = 1 if G3 >= 10 else 0
```

Where:

* 1 = Pass
* 0 = Fail

---

## 6. Feature Selection

The final grade (G3) is excluded from the feature set because it directly determines the target variable.

Including G3 would result in data leakage.

---

## 7. Train-Test Split

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

This ensures reliable model evaluation on unseen data.

---

# 🤖 Machine Learning Model

## XGBoost Classifier

The project uses the **Extreme Gradient Boosting (XGBoost)** algorithm.

### Why XGBoost?

* High prediction accuracy
* Fast training speed
* Handles large datasets efficiently
* Built-in regularization
* Reduced overfitting
* Excellent feature importance analysis

---

# 🏋️ Model Training

The model learns patterns from historical student data and establishes relationships between student characteristics and academic outcomes.

```python
xgb.fit(X_train, y_train)
```

After training, the model can classify new students as Pass or Fail.

---

# 🔮 Prediction

Predictions are generated using:

```python
y_pred = xgb.predict(X_test)
```

The predicted values are compared with actual results to assess model performance.

---

# 📊 Model Evaluation

The following metrics are used:

## Accuracy Score

Measures overall prediction correctness.

```python
accuracy_score(y_test, y_pred)
```

---

## Classification Report

Provides:

* Precision
* Recall
* F1-Score
* Support

---

## Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

This helps evaluate classification effectiveness.

---

# 📈 Feature Importance Analysis

XGBoost provides feature importance scores automatically.

The analysis identifies factors that contribute most significantly to student success.

Examples include:

* Study Time
* Attendance
* Previous Grades
* School Support
* Family Background

Feature importance helps educators understand the key drivers of academic performance.

---

# 💾 Model Saving

The trained model is saved using Joblib.

```python
joblib.dump(
    xgb,
    "student_performance_model.pkl"
)
```

Benefits:

* No retraining required
* Faster deployment
* Easy integration into applications

---

# 🧪 Sample Prediction

The system can predict performance for a new student record.

### Output Example

```text
Predicted Performance : Pass
Probability of Passing : 96.4%
```

This demonstrates real-world applicability.

---

# 📁 Project Structure

Student-Performance-Tracking-System/
│
├── student_performance.py
├── student_data.csv
├── student_performance_model.pkl
├── README.md
│
├── outputs/
│ ├── confusion_matrix.png
│ ├── feature_importance.png
│ └── accuracy_report.png
│
└── requirements.txt

---

# ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Student-Performance-Tracking-System.git
```

Move into the project directory:

```bash
cd Student-Performance-Tracking-System
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

---

# ▶️ Run the Project

```bash
python student_performance.py
```

---

# 📌 Current Scope

The current version supports:

* Student Performance Prediction
* Pass/Fail Classification
* Data Cleaning and Preprocessing
* Feature Importance Analysis
* Model Evaluation
* Model Saving and Loading

---

# 🔮 Future Enhancements

### Multi-Class Classification

Students can be categorized as:

* Excellent
* Good
* Average
* Poor

### Real-Time Monitoring

Continuous performance tracking through educational platforms.

### Web Application

Deployment using:

* Flask
* Django
* Streamlit

### Interactive Dashboards

Visualization using:

* Power BI
* Tableau

### Personalized Recommendations

Generate:

* Study Plans
* Academic Guidance
* Extra Classes

based on predictions.

### Early Warning System

Automatically alert educators about at-risk students.

### Mobile Application

Dedicated applications for:

* Students
* Parents
* Teachers

### Cloud Deployment

Deploy on:

* AWS
* Microsoft Azure
* Google Cloud Platform

---

# 🎯 Expected Outcomes

Successful implementation can help institutions:

* Improve academic performance.
* Detect struggling students early.
* Support data-driven decision making.
* Optimize educational resources.
* Increase student success rates.

---

# 🏆 Conclusion

The Student Performance Tracking System demonstrates the practical application of Machine Learning in education. By leveraging the power of the XGBoost algorithm, the system accurately predicts whether students are likely to pass or fail and identifies the factors influencing academic performance.

The project serves as a strong foundation for future educational analytics platforms and highlights the growing role of Artificial Intelligence in transforming modern education.

---

## 👨‍💻 Author

**Biplob Kumar Dutta**

Machine Learning Project

Student Performance Tracking System Using XGBoost

2026
