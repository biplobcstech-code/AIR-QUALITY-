🌍 Air Quality Index Prediction Using Random Forest








📖 Project Description

Air pollution has become one of the most critical environmental issues affecting public health worldwide. Monitoring and predicting Air Quality Index (AQI) values helps governments, researchers, and environmental agencies make informed decisions to reduce pollution and protect public health.

This project implements a Random Forest Regression Model to predict air quality values using historical environmental and geographical data. The system performs data preprocessing, feature engineering, model training, evaluation, visualization, and model saving.

The trained model achieves an R² Score of 94.84%, demonstrating excellent predictive performance.

🎯 Objectives

The main objectives of this project are:

Predict Air Quality Index (AQI) values accurately.
Analyze important factors affecting air quality.
Identify the most influential environmental indicators.
Visualize model performance and prediction behavior.
Save the trained model for future deployment.

🛠️ Technologies Used
Technology	Purpose
Python	Programming Language
Pandas	Data Analysis
NumPy	Numerical Computation
Matplotlib	Data Visualization
Seaborn	Statistical Visualization
Scikit-Learn	Machine Learning
Joblib	Model Persistence

📂 Dataset Information

The dataset contains environmental and geographical measurements related to air quality.

Dataset Shape
(18862, 12)
Dataset Features
Column Name	Description
Unique ID	Unique record identifier
Indicator ID	Air quality indicator ID
Name	Indicator name
Measure	Measurement type
Measure Info	Additional measure information
Geo Type Name	Geographic area type
Geo Join ID	Geographic identifier
Geo Place Name	Geographic location
Time Period	Observation period
Start_Date	Date of observation
Data Value	AQI value (Target Variable)
Message	Additional notes
🔍 Exploratory Data Analysis

The dataset was inspected for:

Shape and dimensions
Missing values
Duplicate records
Data types
Feature distributions
Missing Values Found
Column	Missing Values
Message	18,862
Others	0

The Message column was filled using:

df["Message"] = df["Message"].fillna("No Message")
🧹 Data Preprocessing
1. Missing Value Handling
df["Message"] = df["Message"].fillna("No Message")
2. Date Conversion
df["Start_Date"] = pd.to_datetime(
    df["Start_Date"],
    errors="coerce"
)
3. Duplicate Removal
df.drop_duplicates(inplace=True)
⚙️ Feature Engineering

Several time-based features were extracted from the date column.

Extracted Features
df["Year"]
df["Month"]
df["Day"]
df["Quarter"]
df["DayOfWeek"]
Weekend Feature
df["IsWeekend"] = np.where(
    df["DayOfWeek"] >= 5,
    1,
    0
)
Remove Original Date Column
df.drop(
    "Start_Date",
    axis=1,
    inplace=True
)
🔤 Encoding Categorical Variables

Since machine learning models cannot process text directly, categorical features were converted into numerical values using Label Encoding.

for col in df.select_dtypes(include="object").columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col].astype(str)
    )
🎯 Target Variable
target_column = "Data Value"

The model predicts the Data Value column.

📊 Feature and Target Selection
X = df.drop(columns=["Data Value"])

y = df["Data Value"]
Shape After Processing
Feature Shape : (18862, 16)
Target Shape  : (18862,)
✂️ Train-Test Split

The dataset was divided into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
Split Summary
Dataset	Shape
Training Data	(15089, 16)
Testing Data	(3773, 16)
🤖 Machine Learning Model
Random Forest Regressor

Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their predictions.

Model Configuration
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
Why Random Forest?

✔ Handles nonlinear relationships

✔ Robust against noise

✔ Prevents overfitting

✔ High prediction accuracy

✔ Provides feature importance

✔ Works well on large datasets

🚀 Model Training
model.fit(
    X_train,
    y_train
)
Training Output
Training Model...
Model Training Completed
📈 Model Evaluation

The model was evaluated using four regression metrics.

Evaluation Results
Metric	Score
MAE	1.9077
MSE	30.0101
RMSE	5.4782
R² Score	0.9484
Mean Absolute Error (MAE)
1.9077

Average prediction error is approximately 1.9 AQI units.

Mean Squared Error (MSE)
30.0101

Measures the average squared difference between actual and predicted values.

Root Mean Squared Error (RMSE)
5.4782

Predictions deviate from actual values by approximately 5.48 AQI units.

R² Score
0.9484

The model explains 94.84% of the variance in AQI values.

Interpretation
R² Value	Model Quality
0.50	Moderate
0.70	Good
0.90+	Excellent

The obtained R² score indicates excellent model performance.

📊 Feature Importance Analysis

The Random Forest algorithm automatically calculates feature importance scores.

Top 10 Important Features
Rank	Feature	Importance
1	Name	0.314155
2	Geo Join ID	0.190590
3	Unique ID	0.189597
4	Geo Place Name	0.079977
5	Indicator ID	0.078581
6	Measure Info	0.056148
7	Measure	0.042192
8	Time Period	0.019511
9	Geo Type Name	0.007706
10	DayOfWeek	0.005404
Interpretation

The most influential factors affecting AQI prediction are:

Indicator Name
Geographic Information
Location Identifiers
Measurement Information
📉 Visualization Results
1. Feature Importance Plot
Purpose

Displays the contribution of each feature toward prediction.

Observation
Name has the highest influence.
Geographical features significantly affect AQI.
Date-related features contribute less.
2. Actual vs Predicted Plot
Purpose

Compares actual AQI values with predicted AQI values.

Observation
Most points lie close to the ideal prediction line.
Strong positive relationship between actual and predicted values.
Indicates high prediction accuracy.
3. Residual Distribution Plot
Purpose

Analyzes prediction errors.

Residual Formula:

Residual = Actual Value − Predicted Value
Observation
Residuals are centered around zero.
Most errors are very small.
Few outliers exist at extreme AQI values.
Conclusion

The model is unbiased and generalizes well on unseen data.

🔮 Sample Prediction
sample = X.iloc[[0]]

prediction = model.predict(sample)
Output
Sample Predicted Data Value : 34.31

This demonstrates how the model can predict AQI for new environmental records.

💾 Model Persistence

The trained model is saved using Joblib.

joblib.dump(
    model,
    "Air_Quality_RandomForest_Model.pkl"
)
Output
Model Saved Successfully!
📁 Project Structure
Air-Quality-Index-Prediction/
│
├── Air quality index.py
├── Air_Quality_and_Health_Impacts.csv
├── Air_Quality_RandomForest_Model.pkl
├── README.md
│
├── outputs/
│   ├── feature_importance.png
│   ├── actual_vs_predicted.png
│   └── residual_distribution.png
│
└── requirements.txt
▶️ Installation

Clone the repository:

git clone https://github.com/yourusername/Air-Quality-Index-Prediction.git

Move into the project folder:

cd Air-Quality-Index-Prediction

Install required packages:

pip install pandas numpy matplotlib seaborn scikit-learn joblib
▶️ Run the Project
python "Air quality index.py"
📌 Results Summary
Parameter	Result
Dataset Size	18,862 Records
Features After Engineering	16
Training Samples	15,089
Testing Samples	3,773
Algorithm	Random Forest Regression
Trees	300
MAE	1.9077
RMSE	5.4782
R² Score	94.84%
🔮 Future Improvements
XGBoost Regressor
LightGBM Regressor
CatBoost Regressor
Hyperparameter Optimization
Real-Time AQI Prediction API
Streamlit Dashboard
Flask Web Application
Air Quality Forecasting using LSTM
🎓 Conclusion

This project successfully develops a Machine Learning-based Air Quality Index Prediction System using the Random Forest Regression Algorithm. Through data preprocessing, feature engineering, model training, and evaluation, the system achieved an R² Score of 94.84%, demonstrating excellent predictive capability.

The results indicate that environmental and geographical attributes play a significant role in determining air quality conditions. The trained model can be further integrated into real-world monitoring systems, environmental dashboards, and smart-city applications for proactive air quality management.
