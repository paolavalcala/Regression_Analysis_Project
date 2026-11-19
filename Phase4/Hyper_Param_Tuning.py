from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Paths relative to this script's directory
current_dir = Path(__file__).resolve().parent
weather_path = current_dir.parent / "Phase3" / "Phase3_TrimmedData" / "combined_weather_selected.csv"
power_path = current_dir.parent / "Phase3" / "Phase3_TrimmedData" / "southernedison_selected.csv"

# Define and ensure the graphs directory exists inside Phase4
graphs_dir = current_dir / "Phase4_Graphs"
graphs_dir.mkdir(parents=True, exist_ok=True)

# Load the data
weather = pd.read_csv(weather_path)
power = pd.read_csv(power_path)

weather['Date'] = pd.to_datetime(weather['Date'])
power['Date'] = pd.to_datetime(power['Date'])

# Condense both cities into one
daily = weather.groupby('Date', as_index=False)[['Max_Temp_C', 'Humidity_Pct', 'Solar_Rad']].mean()
daily = daily.sort_values('Date').reset_index(drop=True)

# Track hot and cold days
daily['Hot_Day'] = (daily['Max_Temp_C'] >= daily['Max_Temp_C'].quantile(0.85)).astype(int)
daily['Cold_Day'] = (daily['Max_Temp_C'] <= daily['Max_Temp_C'].quantile(0.15)).astype(int)

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

# Converting daily rows into monthly
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

# Previous month temp & seasonal feature
df['time_lag1'] = df['temp_mean'].shift(1).bfill()
N = len(df)
fft_vals = np.abs(fft(df['temp_mean'].values))[:N // 2]
fft_freqs = fftfreq(N, d=1)[:N // 2]
main_freq = fft_freqs[np.argmax(fft_vals[1:]) + 1]
df['fft_season_sin'] = np.sin(2 * np.pi * main_freq * np.arange(N))

features = ['temp_mean', 'hot_days_count', 'hot_days_streak', 'time_lag1', 'fft_season_sin']
X = df[features]
y = df['Peak_Demand_MW']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 80/20 split without shuffling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
test_dates = df['Date'].iloc[len(X_train):].reset_index(drop=True)

# --- Hyperparameter Tuning Loop Polynomial Degrees 1 to 4 --- Paola
degrees = [1, 2, 3, 4]
train_r2s = []
test_r2s = []
test_mses = []
train_mses = []
trained_models = {}

print("\n--- Tuning Polynomial Degree ---")
for deg in degrees:
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(X_train, y_train)
    trained_models[deg] = model

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Scores
    tr_r2 = r2_score(y_train, y_train_pred)
    te_r2 = r2_score(y_test, y_test_pred)
    tr_mse = mean_squared_error(y_train, y_train_pred)
    te_mse = mean_squared_error(y_test, y_test_pred)

    train_r2s.append(tr_r2)
    test_r2s.append(te_r2)
    train_mses.append(tr_mse)
    test_mses.append(te_mse)

    print(f"Degree {deg} -> Train R2: {tr_r2:.4f} | Test R2: {te_r2:.4f} | Test MSE: {te_mse:.2f}")

# --- Text Report Summary ---
print("\n" + "="*70)
print("POLYNOMIAL REGRESSION HYPERPARAMETER TUNING REPORT")
print("="*70)
print(f"{'Degree':<8} | {'Train R2':<10} | {'Test R2':<10} | {'Train MSE':<12} | {'Test MSE':<12}")
print("-" * 70)
for d, tr_r2, te_r2, tr_mse, te_mse in zip(degrees, train_r2s, test_r2s, train_mses, test_mses):
    print(f"{d:<8} | {tr_r2:<10.4f} | {te_r2:<10.4f} | {tr_mse:<12.2f} | {te_mse:<12.2f}")
print("="*70)

# Identify best degree based on Test R2
best_idx = np.argmax(test_r2s)
best_deg = degrees[best_idx]
print(f"\nConclusion: Polynomial Degree {best_deg} performed the best on the test set,
    achieving the highest Test R2 score of {test_r2s[best_idx]:.4f}.")
print("="*70)

# --- Visualization Graph 1 Hyperparameter Tuning Results (Train vs Test R2) --- Paola
plt.figure(figsize=(10, 5))
plt.plot(degrees, train_r2s, marker='o', label='Train R2', linewidth=2, color='blue')
plt.plot(degrees, test_r2s, marker='s', label='Test R2', linewidth=2, color='orange')
plt.title('Effect of Polynomial Degree on Model Performance (Overfitting Check)')
plt.xlabel('Polynomial Degree')
plt.ylabel('R2 Score')
plt.xticks(degrees)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

output_path_1 = graphs_dir / "polynomial_tuning_results.png"
plt.savefig(output_path_1, dpi=300)
print(f"Visualization 1 saved to: {output_path_1}")
plt.close()

# --- Visualization Graph 2 Actual vs Predicted Peak Demand for the Best Model ---
best_model = trained_models[best_deg]
y_test_pred_best = best_model.predict(X_test)

plt.figure(figsize=(11, 5))
plt.plot(test_dates, y_test, label='Actual Peak Demand', color='black', linewidth=2, marker='o')
plt.plot(test_dates, y_test_pred_best, label=f'Predicted (Best - Degree {best_deg})', color='crimson', linewidth=2, linestyle='--', marker='x')
plt.title(f'Actual vs Predicted Peak Energy Demand (Best Model: Degree {best_deg})')
plt.xlabel('Date')
plt.ylabel('Peak Demand (MW)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()

output_path_2 = graphs_dir / "actual_vs_predicted_fit.png"
plt.savefig(output_path_2, dpi=300)
print(f"Visualization 2 saved to: {output_path_2}")
plt.close()

# --- Visualizations Graphs 3-6 are Individual Actual vs Predicted for Each Degree (1 to 4) ---
for deg in degrees:
    model = trained_models[deg]
    y_test_pred_deg = model.predict(X_test)
    deg_r2 = r2_score(y_test, y_test_pred_deg)
    deg_mse = mean_squared_error(y_test, y_test_pred_deg)

    plt.figure(figsize=(11, 5))
    plt.plot(test_dates, y_test, label='Actual Peak Demand', color='black', linewidth=2, marker='o')
    plt.plot(test_dates, y_test_pred_deg, label=f'Predicted (Degree {deg})', color='purple', linewidth=2, linestyle='--', marker='x')
    plt.title(f'Actual vs Predicted Peak Demand - Polynomial Degree {deg} (Test R2: {deg_r2:.4f}, MSE: {deg_mse:.2f})')
    plt.xlabel('Date')
    plt.ylabel('Peak Demand (MW)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    deg_output_path = graphs_dir / f"actual_vs_predicted_degree_{deg}.png"
    plt.savefig(deg_output_path, dpi=300)
    print(f"Individual Degree {deg} visualization saved to: {deg_output_path}")
    plt.close()
