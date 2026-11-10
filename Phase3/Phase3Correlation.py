import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

import matplotlib
# Force macOS interactive backend
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

BASE_DIR   = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PHASE2_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Phase2", "Phase2_CleanData"))

TRIMMED_DIR = os.path.join(BASE_DIR, "Phase3_TrimmedData")
GRAPH_DIR   = os.path.join(BASE_DIR, "Phase3_Graphs")

os.makedirs(TRIMMED_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

PATH_WEATHER   = os.path.join(PHASE2_DIR, "combined_weather_cleaned.csv")
PATH_RIVERSIDE = os.path.join(PHASE2_DIR, "riverside_noaa_cleaned.csv")
PATH_SB        = os.path.join(PHASE2_DIR, "san_bernardino_noaa_cleaned.csv")
PATH_POWER     = os.path.join(PHASE2_DIR, "southernedisoncleaned.csv")

df_weather   = pd.read_csv(PATH_WEATHER)
df_riverside = pd.read_csv(PATH_RIVERSIDE)
df_sb        = pd.read_csv(PATH_SB)
df_power     = pd.read_csv(PATH_POWER)

# Standardize date format
df_weather['Date']   = pd.to_datetime(df_weather['Date'])
df_riverside['Date'] = pd.to_datetime(df_riverside['Date'])
df_sb['Date']        = pd.to_datetime(df_sb['Date'])
df_power['Date']     = pd.to_datetime(df_power['Date'])

# CORRELATION & PAIRPLOT
df_w_num   = df_weather.select_dtypes(include=np.number)
df_riv_num = df_riverside.select_dtypes(include=np.number)
df_sb_num  = df_sb.select_dtypes(include=np.number)

# Correlation matrix
corr_weather = df_w_num.corr()

# Internal weather feature heatmap (Saved to Phase3_Graphs)
plt.figure(figsize=(8, 6))
sns.heatmap(corr_weather, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Pearson Correlation Matrix: Combined Weather Features")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "weather_correlation_heatmap.png"))
plt.close()

# WEATHER FEATURES AGAINST ENERGY DEMAND (Daily to Monthly Aggregation)
df_monthly_weather = df_weather.groupby(pd.Grouper(key='Date', freq='MS')).mean(numeric_only=True).reset_index()

# Merge monthly weather with energy demand
df_merged = pd.merge(df_monthly_weather, df_power[['Date', 'Peak_Demand_MW']], on='Date', how='inner')

# Compute correlation matrix
corr_merged = df_merged.select_dtypes(include=np.number).corr()

print("--- Weather Feature Correlation with SCE Energy Peak Demand ---")
print(corr_merged['Peak_Demand_MW'].sort_values(ascending=False))

# Target correlation heatmap (Saved to Phase3_Graphs)
plt.figure(figsize=(8, 6))
sns.heatmap(corr_merged, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Weather Features vs. SCE Energy Peak Demand Correlation")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "target_correlation_heatmap.png"))
plt.close()

# NUMERIC FEATURE SELECTION & REDUNDANCY TRIMMING
# Dropped redundant features like Max_Temp_F, Min_Temp_C, and Dew_point_C due to high collinearity
selected_weather_cols = ['Date', 'City', 'Max_Temp_C', 'Humidity_Pct', 'Solar_Rad']
df_weather_selected   = df_weather[selected_weather_cols]

df_riverside_selected = df_riverside[['Date', 'NOAA_TMax', 'NOAA_PDSI']]
df_sb_selected        = df_sb[['Date', 'NOAA_TMax', 'NOAA_PDSI']]
df_power_selected     = df_power[['Date', 'Peak_Demand_MW']]

# CHI-SQUARE TEST FOR CATEGORICAL INDEPENDENCE
if 'Max_Temp_C' in df_weather.columns and 'City' in df_weather.columns:
    df_weather['Heat_Category'] = pd.qcut(df_weather['Max_Temp_C'], q=2, labels=['Normal', 'Extreme_Heat'])
    contingency_table = pd.crosstab(df_weather['City'], df_weather['Heat_Category'])

    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
    print(f"\n--- Chi-Square Test Results ---\nChi2: {chi2_stat:.4f}, p-value: {p_val:.4e}")

# EXPORT TRIMMED FEATURE DATASETS TO Phase3_TrimmedData
df_weather_selected.to_csv(os.path.join(TRIMMED_DIR, "combined_weather_selected.csv"), index=False)
df_riverside_selected.to_csv(os.path.join(TRIMMED_DIR, "riverside_noaa_selected.csv"), index=False)
df_sb_selected.to_csv(os.path.join(TRIMMED_DIR, "san_bernardino_noaa_selected.csv"), index=False)
df_power_selected.to_csv(os.path.join(TRIMMED_DIR, "southernedison_selected.csv"), index=False)

print("\nDatasets saved to Phase3_TrimmedData and plots routed to Phase3_Graphs.")
