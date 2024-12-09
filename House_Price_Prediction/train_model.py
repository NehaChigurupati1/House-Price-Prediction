import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv('C:/Users/NEHA/Downloads/House_Price_Prediction/neha/house_data.csv')  # Replace with your dataset path

# Display first few rows to understand structure
print("Dataset preview:")
print(df.head())

# Preprocessing: Handle missing values and drop unnecessary columns if needed
df = df.dropna()  # Drop rows with missing values, adjust as needed

# Drop non-numeric columns that can't be used in the model (like 'size_units' and 'lot_size_units')
df = df.drop(columns=['size_units', 'lot_size_units'])

# Convert non-numeric values in 'zip_code' to a numeric type (if needed)
df['zip_code'] = pd.to_numeric(df['zip_code'], errors='coerce')

# Convert categorical columns to numeric (if needed) using encoding, here I'm assuming they are non-numeric
# For example, 'zip_code' could be one to encode if it's not in a numeric format
# You can use pd.get_dummies() for this or simple label encoding
df = pd.get_dummies(df, drop_first=True)

# Assume 'price' is the target column, and the rest are features
X = df.drop(columns=['price'])  # Assuming 'price' is the target column
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2: {r2}")

# Plotting actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Line for perfect prediction
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.show()

# Feature importance plot (if applicable)
if hasattr(model, 'coef_'):
    feature_importances = model.coef_
    plt.figure(figsize=(8, 6))
    plt.barh(X.columns, feature_importances)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()
