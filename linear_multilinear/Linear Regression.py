import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("\n" + "="*80)
print("Loading Dataset and Preparing Data")
print("="*80)

try:
    df = pd.read_csv("realistic_body_data.csv")
    print("Dataset loaded successfully.\n")
except FileNotFoundError:
    print("Error: 'realistic_body_data.csv' not found. Please check the file path.")
    exit()

print("Checking for missing values:")
print(df.isnull().sum())

# Calculate means for numeric columns
x, y, z, a, b = [df[col].mean() for col in ["height", "weight", "waist", "hips", "chest"]]
print("\nColumn Means Calculated:")
print(f"Height: {x:.2f}")
print(f"Weight: {y:.2f}")
print(f"Waist : {z:.2f}")
print(f"Hips  : {a:.2f}")
print(f"Chest : {b:.2f}")

df.fillna({"height": x, "weight": y, "waist": z, "hips": a, "chest": b}, inplace=True)
print("\nMissing values filled successfully.")
print("Null check after imputation:")
print(df.isnull().sum())

# Encode gender
print("\nEncoding 'gender' column (One-Hot)...")
df_processed = pd.get_dummies(df, columns=['gender'], drop_first=True)
print("Gender column encoded.\n")

# ==============================================================
# SIMPLE LINEAR REGRESSION
# ==============================================================
print("\n" + "="*80)
print("SIMPLE LINEAR REGRESSION: Height → Weight")
print("="*80)

X_simple = df_processed[['height']]
y = df_processed['weight']

X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {y_test.shape[0]} samples")

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

print("\nModel Parameters:")
print(f"Intercept (b0): {model_simple.intercept_:.2f}")
print(f"Coefficient for Height (b1): {model_simple.coef_[0]:.2f}")
print(f"Formula: weight = {model_simple.coef_[0]:.2f} * height + {model_simple.intercept_:.2f}")

# Prediction
simple_pred = model_simple.predict(X_test)
r2_simple = r2_score(y_test, simple_pred)
rmse_simple = np.sqrt(mean_squared_error(y_test, simple_pred))

print("\nModel Performance (Simple):")
print(f"R² Score: {r2_simple:.4f}")
print(f"RMSE: {rmse_simple:.2f} kg")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.6, label='Actual Data (Test Set)', color='skyblue')
plt.plot(X_test, simple_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression: Height vs. Weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend()
plt.grid(True)
plt.show()

# ==============================================================
# MULTIPLE LINEAR REGRESSION
# ==============================================================
print("\n" + "="*80)
print("MULTIPLE LINEAR REGRESSION: All Features → Weight")
print("="*80)

feature_cols = [
    'age',
    'height',
    'waist',
    'chest',
    'hips',
    'gender_Male',
    'gender_Non-Binary'
]

X_multi = df_processed[feature_cols]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.3, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

print("\nModel Parameters:")
print(f"Intercept: {model_multi.intercept_:.2f}")
print("Coefficients:")
for i, col in enumerate(feature_cols):
    print(f"  {col:15s}: {model_multi.coef_[i]:.2f}")

# Prediction
y_pred_multi = model_multi.predict(X_test_m)
r2_multi = r2_score(y_test_m, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test_m, y_pred_multi))

print("\nModel Performance (Multiple):")
print(f"R² Score: {r2_multi:.4f}")
print(f"RMSE: {rmse_multi:.2f} kg")

# ==============================================================
# MODEL COMPARISON
# ==============================================================
print("\n" + "="*80)
print("MODEL COMPARISON: Simple vs. Multiple Linear Regression")
print("="*80)
print(f"Simple Model (Height only):     R² = {r2_simple:.4f}, RMSE = {rmse_simple:.2f} kg")
print(f"Multiple Model (All features):  R² = {r2_multi:.4f}, RMSE = {rmse_multi:.2f} kg")
print("="*80)
print("Comparison complete.")
