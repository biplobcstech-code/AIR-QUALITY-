# =====================================================
# AIR QUALITY INDEX PREDICTION USING RANDOM FOREST
# =====================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

# =====================================================
# LOAD DATASET
# =====================================================

file_path = r"D:\DATASETS\Air_Quality_and_Health_Impacts.csv"

df = pd.read_csv(file_path)

# =====================================================
# DATASET INFORMATION
# =====================================================

print("="*60)
print("DATASET INFORMATION")
print("="*60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================================
# DATA CLEANING
# =====================================================

# Fill missing values

df["Message"] = df["Message"].fillna("No Message")

# Convert Date

df["Start_Date"] = pd.to_datetime(
    df["Start_Date"],
    errors="coerce"
)

# Remove duplicates

df.drop_duplicates(inplace=True)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

df["Year"] = df["Start_Date"].dt.year
df["Month"] = df["Start_Date"].dt.month
df["Day"] = df["Start_Date"].dt.day

df["Quarter"] = df["Start_Date"].dt.quarter

df["DayOfWeek"] = df["Start_Date"].dt.dayofweek

df["IsWeekend"] = np.where(
    df["DayOfWeek"] >= 5,
    1,
    0
)

# Drop original date

df.drop(
    "Start_Date",
    axis=1,
    inplace=True
)

# =====================================================
# ENCODE CATEGORICAL COLUMNS
# =====================================================

for col in df.select_dtypes(include="object").columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col].astype(str)
    )

# =====================================================
# TARGET COLUMN
# =====================================================

target_column = "Data Value"

# =====================================================
# FEATURES AND TARGET
# =====================================================

X = df.drop(
    columns=[target_column]
)

y = df[target_column]

print("\nFeature Shape:", X.shape)
print("Target Shape :", y.shape)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# =====================================================
# RANDOM FOREST MODEL
# =====================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

# =====================================================
# TRAIN MODEL
# =====================================================

print("\nTraining Model...")

model.fit(
    X_train,
    y_train
)

print("Model Training Completed")

# =====================================================
# PREDICTION
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# EVALUATION
# =====================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print("MAE  :", round(mae, 4))
print("MSE  :", round(mse, 4))
print("RMSE :", round(rmse, 4))
print("R2   :", round(r2, 4))

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features")

print(
    importance_df.head(10)
)

# =====================================================
# FEATURE IMPORTANCE PLOT
# =====================================================

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 10 Feature Importance"
)

plt.tight_layout()

plt.show()

# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

plt.xlabel("Actual Data Value")
plt.ylabel("Predicted Data Value")

plt.title(
    "Actual vs Predicted"
)

plt.grid(True)

plt.show()

# =====================================================
# RESIDUAL PLOT
# =====================================================

residuals = y_test - y_pred

plt.figure(figsize=(8,6))

sns.histplot(
    residuals,
    bins=30,
    kde=True
)

plt.title(
    "Residual Distribution"
)

plt.xlabel(
    "Prediction Error"
)

plt.show()

# =====================================================
# SAMPLE PREDICTION
# =====================================================

sample = X.iloc[[0]]

prediction = model.predict(sample)

print(
    "\nSample Predicted Data Value :",
    round(prediction[0], 2)
)

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    model,
    "Air_Quality_RandomForest_Model.pkl"
)

print(
    "\nModel Saved Successfully!"
)

# =====================================================
# END
# =====================================================

print(
    "\nAIR QUALITY PROJECT COMPLETED SUCCESSFULLY"
)
