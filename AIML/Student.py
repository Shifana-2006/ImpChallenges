import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_excel(r"C:\Users\user\Downloads\student_performance 2.xlsx")
print("First 5 rows of the dataset:\n", df.head())

# Drop missing values
df = df.dropna()

# Define features and target
X = df[['Attendance (%)', 'Study Hours/Day', 'Participation (%)']]
y = df['Marks (%)']

# ✅ Use entire dataset for training and testing (good for small data demo)
X_train = X
y_train = y
X_test = X
y_test = y

# Initialize models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
svr_model = SVR(kernel='rbf')

# Train models
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
svr_model.fit(X_train, y_train)

# Choose the best model
model = rf_model  # You can switch to lr_model or svr_model if needed

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
print("\nModel Evaluation Metrics:")
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Predict for new student inputs
while True:
    print("\n--- Enter new student details to predict performance ---")
    attendance = float(input("Enter attendance (%): "))
    study_hours = float(input("Enter study hours per day: "))
    participation = float(input("Enter participation level (%): "))

    student_input = np.array([[attendance, study_hours, participation]])
    predicted_score = model.predict(student_input)[0]

    # Performance category
    if predicted_score < 50:
        grade = "Improvement needed"
    elif predicted_score < 70:
        grade = "Good"
    else:
        grade = "Excellent"

    print(f"\nPredicted Final Percentage: {predicted_score:.2f}%")
    print(f"Performance Category: {grade}")

    # Ask if you want to continue
    repeat = input("\nDo you want to predict another student? (yes/no): ").lower()
    if repeat != 'yes':
        print("Prediction ended.")
        break