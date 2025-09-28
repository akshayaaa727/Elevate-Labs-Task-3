import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Import and preprocess dataset
# For demonstration, we'll use a simple synthetic dataset (Housing Prices-like data)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Convert to DataFrame for clarity
df = pd.DataFrame(np.hstack([X, y]), columns=["Feature", "Target"])
print(df.head())

# 2. Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# 4. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"R²: {r2:.3f}")

# 5. Plot regression line (for simple regression)
plt.figure(figsize=(6, 4))
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

# Coefficients interpretation
print("Intercept (bias):", model.intercept_)
print("Coefficient (slope):", model.coef_)

# --- Multiple Linear Regression Example ---
# Generate synthetic data with 2 features
X_multi = 2 * np.random.rand(100, 2)
y_multi = 5 + 2*X_multi[:, 0] + 3*X_multi[:, 1] + np.random.randn(100)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

y_pred_m = model_multi.predict(X_test_m)

print("\nMultiple Linear Regression Evaluation:")
print(f"MAE: {mean_absolute_error(y_test_m, y_pred_m):.3f}")
print(f"MSE: {mean_squared_error(y_test_m, y_pred_m):.3f}")
print(f"R²: {r2_score(y_test_m, y_pred_m):.3f}")

print("Intercept (bias):", model_multi.intercept_)
print("Coefficients (slopes):", model_multi.coef_)