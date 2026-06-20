# 🌍 Air Quality Index Prediction System Using Random Forest

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

# 📖 Introduction

Air pollution has become one of the most serious environmental and public health concerns worldwide. Poor air quality contributes to respiratory diseases, cardiovascular problems, and reduced quality of life. Monitoring and predicting air quality levels is therefore essential for governments, environmental agencies, healthcare organizations, and researchers.

This project presents an **Air Quality Index (AQI) Prediction System** powered by **Machine Learning**. The system uses the **Random Forest Regression Algorithm** to analyze environmental and geographical factors and predict air quality values accurately.

By utilizing historical air quality data, the model learns patterns between environmental indicators and AQI measurements, enabling accurate predictions and deeper understanding of factors affecting air quality.

---

# 🎯 Problem Statement

Air quality datasets contain large volumes of environmental measurements collected over time from various geographic regions. While this data is available, extracting meaningful insights and forecasting future air quality levels remains a challenge.

Traditional statistical methods often struggle to capture complex relationships between environmental variables and pollution levels.

The goal of this project is to develop a machine learning-based prediction system capable of:

* Analyzing environmental and geographical data.
* Predicting Air Quality Index values.
* Identifying the most influential factors affecting air quality.
* Supporting environmental monitoring and decision-making.
* Assisting researchers and policymakers with data-driven insights.

---

# 🚀 Project Objectives

The primary objectives of this project are:

* Predict Air Quality Index (AQI) values using machine learning.
* Analyze environmental factors affecting air quality.
* Improve air quality forecasting accuracy.
* Identify important predictors influencing AQI.
* Provide meaningful visualizations for analysis.
* Build a reusable predictive model for future deployment.

---

# 🛠 Technologies Used

## Programming Language

* Python 3.x

## Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib

---

# 📂 Dataset Description

The dataset contains environmental and geographical information associated with air quality measurements.

### Dataset Features

| Feature        | Description                        |
| -------------- | ---------------------------------- |
| Unique ID      | Unique record identifier           |
| Indicator ID   | Air quality indicator identifier   |
| Name           | Indicator name                     |
| Measure        | Measurement category               |
| Measure Info   | Additional measurement information |
| Geo Type Name  | Geographic classification          |
| Geo Join ID    | Geographic identifier              |
| Geo Place Name | Location name                      |
| Time Period    | Observation period                 |
| Start_Date     | Observation date                   |
| Data Value     | Air Quality Value                  |
| Message        | Additional information             |

### Target Variable

The model predicts:

| Target Column |
| ------------- |
| Data Value    |

This represents the Air Quality Index value to be estimated.

---

# 🔄 Project Workflow

## 1. Data Collection and Loading

The dataset is loaded into a Pandas DataFrame for preprocessing and analysis.

This allows the machine learning pipeline to access and manipulate environmental records efficiently.

---

## 2. Data Exploration

Before model development, the dataset is explored to understand:

* Dataset dimensions
* Available features
* Data types
* Missing values
* Overall structure

This step ensures data quality and suitability for machine learning.

---

## 3. Data Cleaning and Missing Value Handling

Real-world environmental datasets often contain incomplete information.

To improve data quality:

### Missing Values

The Message column contains missing values and is filled using a default value:

```python
df["Message"] = df["Message"].fillna("No Message")
```

### Duplicate Records

Duplicate observations are removed:

```python
df.drop_duplicates(inplace=True)
```

### Date Conversion

Date values are converted into proper datetime format:

```python
df["Start_Date"] = pd.to_datetime(
    df["Start_Date"],
    errors="coerce"
)
```

---

## 4. Feature Engineering

Additional date-based features are extracted from the observation date.

### Extracted Features

* Year
* Month
* Day
* Quarter
* DayOfWeek
* IsWeekend

Example:

```python
df["Year"] = df["Start_Date"].dt.year
df["Month"] = df["Start_Date"].dt.month
df["Day"] = df["Start_Date"].dt.day
df["Quarter"] = df["Start_Date"].dt.quarter
df["DayOfWeek"] = df["Start_Date"].dt.dayofweek
```

Weekend identification:

```python
df["IsWeekend"] = np.where(
    df["DayOfWeek"] >= 5,
    1,
    0
)
```

These engineered features help the model identify temporal trends in air quality.

---

## 5. Feature Encoding

Machine learning algorithms require numerical input.

Categorical features such as:

* Name
* Measure
* Geo Place Name
* Geo Type Name

are converted into numerical form using Label Encoding.

```python
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
```

---

## 6. Feature Selection

The target column is separated from input features.

```python
X = df.drop(columns=["Data Value"])
y = df["Data Value"]
```

### Dataset Summary

| Parameter                  | Value      |
| -------------------------- | ---------- |
| Total Records              | 18,862     |
| Features After Engineering | 16         |
| Target Variable            | Data Value |

---

## 7. Train-Test Split

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

This ensures unbiased model evaluation.

---

# 🤖 Machine Learning Model

## Random Forest Regressor

The project utilizes the Random Forest Regression algorithm.

Random Forest is an ensemble learning technique that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

### Model Configuration

```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
```

### Why Random Forest?

* Handles nonlinear relationships
* Robust against noise
* High prediction accuracy
* Reduced overfitting
* Supports feature importance analysis
* Performs well on large datasets

---

# 🏋️ Model Training

The model learns relationships between environmental variables and AQI values.

```python
model.fit(
    X_train,
    y_train
)
```

During training, multiple decision trees are built and combined to produce accurate predictions.

---

# 🔮 Prediction

The trained model generates AQI predictions on unseen data.

```python
y_pred = model.predict(X_test)
```

These predictions are compared with actual AQI values to evaluate model performance.

---

# 📊 Model Evaluation

Several regression metrics are used.

## Mean Absolute Error (MAE)

Measures average prediction error.

### Result

```text
1.9077
```

---

## Mean Squared Error (MSE)

Measures squared prediction error.

### Result

```text
30.0101
```

---

## Root Mean Squared Error (RMSE)

Measures overall prediction deviation.

### Result

```text
5.4782
```

---

## R² Score

Measures how well the model explains variance in AQI values.

### Result

```text
0.9484
```

### Interpretation

The model explains approximately **94.84% of the variability** in air quality values, indicating excellent predictive performance.

---

# 📈 Feature Importance Analysis

Random Forest provides feature importance scores that indicate the contribution of each feature.

### Top Influential Features

* Name
* Geo Join ID
* Unique ID
* Geo Place Name
* Indicator ID
* Measure Info
* Measure

Feature importance analysis helps identify the key factors influencing air quality measurements.

---

# 📉 Visualization Results

## Feature Importance Plot

Displays the contribution of the most influential features.

### Benefits

* Improves model interpretability.
* Identifies significant environmental factors.

---

## Actual vs Predicted Plot

Compares actual AQI values with predicted values.

### Observation

* Strong positive relationship observed.
* Most predictions closely match actual values.
* Indicates high model accuracy.

---

## Residual Distribution Plot

Analyzes prediction errors.

Residual Formula:

```python
Residual = Actual Value - Predicted Value
```

### Observation

* Most residuals are centered around zero.
* Prediction errors remain small.
* Indicates a well-trained model.

---

# 💾 Model Saving

The trained model is saved for future deployment.

```python
joblib.dump(
    model,
    "Air_Quality_RandomForest_Model.pkl"
)
```

### Benefits

* Reuse without retraining.
* Faster deployment.
* Easy integration into applications.

---

# 🧪 Sample Prediction

The model can predict AQI values for new environmental observations.

### Example Output

```text
Sample Predicted Data Value : 34.31
```

This demonstrates practical usage of the trained model.

---

# 📁 Project Structure

Air-Quality-Index-Prediction/
│
├── Air quality index.py
├── Air_Quality_and_Health_Impacts.csv
├── Air_Quality_RandomForest_Model.pkl
├── README.md
│
├── outputs/
│ ├── feature_importance.png
│ ├── actual_vs_predicted.png
│ └── residual_distribution.png
│
└── requirements.txt

---

# ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Air-Quality-Index-Prediction.git
```

Navigate to the project directory:

```bash
cd Air-Quality-Index-Prediction
```

Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

# ▶️ Run the Project

```bash
python "Air quality index.py"
```

---

# 📌 Current Scope

The current version focuses on:

* AQI Prediction
* Data Cleaning and Preprocessing
* Feature Engineering
* Feature Importance Analysis
* Visualization
* Model Evaluation
* Model Saving

---

# 🔮 Future Enhancements

### Advanced Machine Learning Models

* XGBoost Regressor
* LightGBM Regressor
* CatBoost Regressor

### Deep Learning Forecasting

* LSTM Networks
* Time-Series Prediction Models

### Web Application Development

Using:

* Flask
* Django
* Streamlit

### Interactive Dashboards

Using:

* Power BI
* Tableau

### Real-Time AQI Monitoring

Integration with live environmental sensors.

### Cloud Deployment

Deploy on:

* AWS
* Microsoft Azure
* Google Cloud Platform

---

# 🎯 Expected Outcomes

Successful implementation can help:

* Environmental Agencies
* Government Organizations
* Researchers
* Public Health Departments

Benefits include:

* Improved AQI forecasting
* Better environmental planning
* Early pollution warnings
* Data-driven policy decisions

---

# 🏆 Conclusion

The Air Quality Index Prediction System demonstrates how Machine Learning can be applied to environmental analytics. Using the Random Forest Regression algorithm, the model accurately predicts AQI values and identifies the most influential environmental factors affecting air quality.

With an R² score of approximately 94.84%, the system provides highly reliable predictions and serves as a strong foundation for future smart-city, environmental monitoring, and air quality forecasting applications.

---

## 👨‍💻 Author

**Biplob Kumar Dutta**
INTERN ID:- CTIS9255

NO OF WEEKS:- 8 WEEK

Machine Learning Project

Air Quality Index Prediction Using Random Forest


