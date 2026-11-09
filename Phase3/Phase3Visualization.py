import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


BASE_DIR    = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
TRIMMED_DIR = os.path.join(BASE_DIR, "Phase3_TrimmedData")

PATH_WEATHER_SEL = os.path.join(TRIMMED_DIR, "combined_weather_selected.csv")
PATH_POWER_SEL   = os.path.join(TRIMMED_DIR, "southernedison_selected.csv")

# Load the trimmed datasets
df_weather_sel = pd.read_csv(PATH_WEATHER_SEL)
df_power_sel   = pd.read_csv(PATH_POWER_SEL)

df_weather_sel['Date'] = pd.to_datetime(df_weather_sel['Date'])
df_power_sel['Date']   = pd.to_datetime(df_power_sel['Date'])

# Merge trimmed datasets on Date
df_merged = pd.merge(df_weather_sel, df_power_sel, on='Date', how='inner').dropna()


# PLOT 1: Feature Correlation Heatmap using Trimmed Data

plt.figure(figsize=(8, 6))
corr_trimmed = df_merged.select_dtypes(include=np.number).corr()
sns.heatmap(corr_trimmed, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Trimmed Data: Weather Features vs. Peak Demand Correlation")
plt.tight_layout()
plt.savefig(os.path.join(TRIMMED_DIR, "trimmed_correlation_heatmap.png"))
plt.close()


# PLOT 2: Trendline Comparison (Linear vs. Polynomial Regression)

X = df_merged[['Max_Temp_C']].values
y = df_merged['Peak_Demand_MW'].values

sort_idx = np.argsort(X.flatten())
X_sorted = X[sort_idx]
y_sorted = y[sort_idx]

# Linear Fit
lin_model = LinearRegression()
lin_model.fit(X_sorted, y_sorted)
y_lin_pred = lin_model.predict(X_sorted)

# Polynomial Fit (Degree 2)
poly_feat = PolynomialFeatures(degree=2)
X_poly = poly_feat.fit_transform(X_sorted)
poly_model = LinearRegression()
poly_model.fit(X_poly, y_sorted)
y_poly_pred = poly_model.predict(X_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X_sorted, y_sorted, color='gray', alpha=0.5, label='Trimmed Monthly Data')
plt.plot(X_sorted, y_lin_pred, color='tab:blue', linewidth=2, label='Linear Trendline')
plt.plot(X_sorted, y_poly_pred, color='tab:red', linewidth=2, linestyle='--', label='Polynomial Trendline (Degree 2)')

plt.title("Trimmed Data: Temperature vs. Energy Demand (Linear vs. Polynomial)")
plt.xlabel('Maximum Temperature (Celsius)')
plt.ylabel('Peak Energy Demand (MW)')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(TRIMMED_DIR, "trimmed_regression_comparison.png"))
plt.close()

print("Successfully generated and saved both plots using the trimmed data in Phase3_TrimmedData!")
