import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import PolynomialFeatures

print("Loading data for advanced analysis...")
dataset = pd.read_csv("World Energy Consumption.csv")
india_data = dataset[dataset['country'] == 'India'].copy()

# Sort by year
india_data = india_data.sort_values('year')
india_data = india_data.dropna(subset=['year', 'primary_energy_consumption'])

# ---------------------------------------------------------
# 1. Forecast to 2035
# ---------------------------------------------------------
X = india_data[['year']].values
y = india_data['primary_energy_consumption'].values

# Using degree 3 for a more stable and realistic forecast compared to degree 6
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
regressor = LR()
regressor.fit(X_poly, y)

future_years = np.arange(X.min(), 2036).reshape(-1, 1)
future_pred = regressor.predict(poly_reg.fit_transform(future_years))

plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='black', label='Historical Data', alpha=0.6)
plt.plot(future_years, future_pred, color='red', linestyle='--', label='Forecast to 2035 (Polynominal Deg=3)')
plt.title("India's Primary Energy Consumption: Historical & Forecast to 2035")
plt.xlabel("Year")
plt.ylabel("Energy Consumption (TWh)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("India_Energy_Forecast.png")
print("Forecast plot saved as India_Energy_Forecast.png.")

# ---------------------------------------------------------
# 2. Energy Mix Breakdown
# ---------------------------------------------------------
mix_cols = {
    'coal_consumption': 'Coal',
    'oil_consumption': 'Oil',
    'gas_consumption': 'Gas',
    'hydro_consumption': 'Hydro',
    'nuclear_consumption': 'Nuclear',
    'solar_consumption': 'Solar',
    'wind_consumption': 'Wind'
}

plt.figure(figsize=(12, 6))
for col, label in mix_cols.items():
    if col in india_data.columns:
        # Fill NA with 0 for plotting
        data_col = india_data[col].fillna(0)
        # Only plot if there's data
        if data_col.max() > 0:
            plt.plot(india_data['year'], data_col, label=label, linewidth=2)

plt.title("India's Energy Mix by Source Over Time")
plt.xlabel("Year")
plt.ylabel("Consumption (TWh)")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("India_Energy_Mix.png")
print("Energy Mix plot saved as India_Energy_Mix.png.")
