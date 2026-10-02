import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error,mean_squared_error

print("Loading data...")
dataset = pd.read_csv("World Energy Consumption.csv")

# Since the project is "India's Energy Crisis", let's filter for India
india_data = dataset[dataset['country'] == 'India'].dropna(subset=['year', 'primary_energy_consumption'])
print(f"Found {len(india_data)} records for India.")

# Use Year as X and Energy Consumption as y (similar to the notebook's intent)
X = india_data[['year']].values
y = india_data['primary_energy_consumption'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

regressor = LR()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X)

# Polynomial regression (Degree 6 as in the notebook)
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
regressor_2 = LR()
regressor_2.fit(X_poly, y)

MAE = mean_absolute_error(y, y_pred)
MSE = mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)

print(f"Linear Regression Results:")
print(f"MAE:  {MAE:.2f}")
print(f"MSE:  {MSE:.2f}")
print(f"RMSE: {RMSE:.2f}")

# Plotting
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='red', label='Actual Data')

# Sort X for plotting the polynomial regression line smoothly
X_grid = np.arange(X.min(), X.max(), 0.1).reshape(-1, 1)
plt.plot(X_grid, regressor_2.predict(poly_reg.fit_transform(X_grid)), color='blue', label='Polynomial Regression (Degree 6)')

plt.xlabel("Years")
plt.ylabel("Primary Energy Consumption (TWh)")
plt.title("India's Energy Consumption vs Years")
plt.legend()
plt.grid(True)
plt.savefig("India_Energy_Consumption_Plot.png")
print("Plot saved as India_Energy_Consumption_Plot.png")
