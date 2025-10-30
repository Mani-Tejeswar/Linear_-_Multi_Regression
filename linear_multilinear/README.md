# Body Weight Regression Project
Simple and multiple linear regression models predicting weight from body measurements.

## Features
- Handles missing values
- One-hot encoding for gender
- Simple and multiple linear regression
- Performance metrics (R², RMSE)
- Matplotlib visualization


Loading Dataset and Preparing Data

Dataset loaded successfully.

Checking for missing values:
age        0
gender     0
height    60
weight    60
waist     60
chest     60
hips      60
dtype: int64

Column Means Calculated:
Height: 168.19
Weight: 67.46
Waist : 86.36
Hips  : 103.36
Chest : 95.35

Missing values filled successfully.
Null check after imputation:
age       0
gender    0
height    0
weight    0
waist     0
chest     0
hips      0
dtype: int64

Encoding 'gender' column (One-Hot)...
Gender column encoded.



SIMPLE LINEAR REGRESSION: Height → Weight


Training set size: 1400 samples
Testing set size:  600 samples

Model Parameters:
Intercept (b0): -60.85
Coefficient for Height (b1): 0.76
Formula: weight = 0.76 * height + -60.85

Model Performance (Simple):
R² Score: 0.2625
RMSE: 11.47 kg


MULTIPLE LINEAR REGRESSION: All Features → Weight


Model Parameters:
Intercept: -46.04
Coefficients:
  age            : 0.01
  height         : -0.09
  waist          : 0.65
  chest          : 0.24
  hips           : 0.47
  gender_Male    : -1.01
  gender_Non-Binary: 4.32

Model Performance (Multiple):
R² Score: 0.8624
RMSE: 4.96 kg


MODEL COMPARISON: Simple vs. Multiple Linear Regression

Simple Model (Height only):     R² = 0.2625, RMSE = 11.47 kg
Multiple Model (All features):  R² = 0.8624, RMSE = 4.96 kg


![alt text](image.png)


Comparison complete.