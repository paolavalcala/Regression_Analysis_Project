from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# paths relative to this script's directory
current_dir = Path(__file__).resolve().parent
weather_path = current_dir.parent / "Phase3" / "Phase3_TrimmedData" / "combined_weather_selected.csv"
power_path = current_dir.parent / "Phase3" / "Phase3_TrimmedData" / "southernedison_selected.csv"

# Load the data using the paths
weather = pd.read_csv(weather_path)
power = pd.read_csv(power_path)

weather['Date'] = pd.to_datetime(weather['Date'])
power['Date'] = pd.to_datetime(power['Date'])

# condense both cities into one
daily = weather.groupby('Date', as_index=False)[['Max_Temp_C', 'Humidity_Pct', 'Solar_Rad']].mean()
daily = daily.sort_values('Date').reset_index(drop=True)

# track hot and cold days
daily['Hot_Day'] = (daily['Max_Temp_C'] >= daily['Max_Temp_C'].quantile(0.85)).astype(int)
daily['Cold_Day'] = (daily['Max_Temp_C'] <= daily['Max_Temp_C'].quantile(0.15)).astype(int)

# streak for hot & cold days
def longest_streak(series):
    streak = 0
    best = 0
    for val in series:
        if val == 1:
            streak += 1
            if streak > best:
                best = streak
        else:
            streak = 0
    return best

# converting daily rows into monthly
daily['Date'] = daily['Date'].dt.to_period('M').dt.to_timestamp()
monthly = daily.groupby('Date').agg(
    temp_mean=('Max_Temp_C', 'mean'),
    temp_max=('Max_Temp_C', 'max'),
    temp_min=('Max_Temp_C', 'min'),
    hot_days_count=('Hot_Day', 'sum'),
    cold_days_count=('Cold_Day', 'sum'),
    hot_days_streak=('Hot_Day', longest_streak),
    cold_days_streak=('Cold_Day', longest_streak),
    humid_mean=('Humidity_Pct', 'mean'),
    solar_rad_mean=('Solar_Rad', 'mean')
).reset_index()

df = pd.merge(monthly, power, on='Date', how='inner').sort_values('Date').reset_index(drop=True)

# previous month temp
df['time_lag1'] = df['temp_mean'].shift(1).bfill()

# 12 month season
N = len(df)
fft_vals = np.abs(fft(df['temp_mean'].values))[:N // 2]
fft_freqs = fftfreq(N, d=1)[:N // 2]
main_freq = fft_freqs[np.argmax(fft_vals[1:]) + 1]
df['fft_season_sin'] = np.sin(2 * np.pi * main_freq * np.arange(N))

# Keep features low for polynomial regression
features = ['temp_mean', 'hot_days_count', 'hot_days_streak', 'time_lag1', 'fft_season_sin']
X = df[features]
y = df['Peak_Demand_MW']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 80/20 split without shuffling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# training model
reg_multiplile = LinearRegression()
reg_multiplile.fit(X_train, y_train)

# predict
y_pred_multiple = reg_multiplile.predict(X_test)

# evaluate
print("\nMultiple Linear Regression")
print("MSE: ", mean_squared_error(y_test, y_pred_multiple))
print("R2: ", r2_score(y_test, y_pred_multiple))

# Polynomial Regression degree 2
degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model.fit(X_train, y_train)

y_pred = poly_model.predict(X_test)
print("\nPolynomial Regression (degree 2) MSE:", mean_squared_error(y_test, y_pred))
print("\nPolynomial Regression (degree 2) R2:", r2_score(y_test, y_pred))

# Neural Network Regression (MLPRegressor) with Hyperparameter Tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 25)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [2000]
}

# Initialize MLPRegressor
mlp = MLPRegressor(random_state=42)

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_mlp = grid_search.best_estimator_
y_pred_mlp = best_mlp.predict(X_test)

# Evaluate and print results
print("\nNeural Network (MLPRegressor)")
print("Best Hyperparameters:", grid_search.best_params_)
print("MSE: ", mean_squared_error(y_test, y_pred_mlp))
print("R2: ", r2_score(y_test, y_pred_mlp))
