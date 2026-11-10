import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib
# Force macOS interactive backend
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt


# FILE PATHS & DATE RANGE CONFIGURATION (2014 - 2024)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

# Point directly to the folders nested inside your 'Phase2' directory
DATA_DIR    = os.path.join(BASE_DIR, "Phase2_RawData")
CLEAN_DIR   = os.path.join(BASE_DIR, "Phase2_CleanData")
GRAPH_DIR   = os.path.join(BASE_DIR, "Phase2_Graphs")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

PATH_COMBINED  = os.path.join(DATA_DIR, "inland_empire_combined_10yr.csv")
PATH_RIVERSIDE = os.path.join(DATA_DIR, "riverside_noaa_extracted.csv")
PATH_SB        = os.path.join(DATA_DIR, "san_bernardino_noaa_extracted.csv")
PATH_EDISON    = os.path.join(DATA_DIR, "southernedisondataorganized.csv")

START_DATE = "2014-01-01"
END_DATE   = "2024-12-31"

# LOAD DATASETS & STANDARDIZE TEMPORAL BOUNDARIES
df_combined  = pd.read_csv(PATH_COMBINED)
df_riverside = pd.read_csv(PATH_RIVERSIDE)
df_sb        = pd.read_csv(PATH_SB)
df_edison    = pd.read_csv(PATH_EDISON)

df_combined['Date']  = pd.to_datetime(df_combined['Date'])
df_riverside['Date'] = pd.to_datetime(df_riverside['Year_Month'] + "-01")
df_sb['Date']        = pd.to_datetime(df_sb['Year_Month'] + "-01")
df_edison['Date']    = pd.to_datetime(df_edison['Date'])

# Filter 10-year study window (2014-2024)
df_combined  = df_combined[(df_combined['Date'] >= START_DATE) & (df_combined['Date'] <= END_DATE)].reset_index(drop=True)
df_riverside = df_riverside[(df_riverside['Date'] >= START_DATE) & (df_riverside['Date'] <= END_DATE)].reset_index(drop=True)
df_sb        = df_sb[(df_sb['Date'] >= START_DATE) & (df_sb['Date'] <= END_DATE)].reset_index(drop=True)
df_edison    = df_edison[(df_edison['Date'] >= START_DATE) & (df_edison['Date'] <= END_DATE)].reset_index(drop=True)

# Temperature unit transformation (°C to °F)
if 'Max_Temp_C' in df_combined.columns:
    df_combined['Max_Temp_F'] = (df_combined['Max_Temp_C'] * 1.8) + 32

# DISPLAY RAW SAMPLES & DESCRIPTIVE STATISTICAL SUMMARIES
print("--- Raw Combined Weather Sample ---")
print(df_combined.head(3))

print("\n--- Raw Riverside NOAA Sample ---")
print(df_riverside.head(3))

print("\n--- Raw San Bernardino NOAA Sample ---")
print(df_sb.head(3))

print("\n--- Raw Southern Edison Power Load Sample ---")
print(df_edison.head(3))

print("\n--- Combined Weather Descriptive Statistics ---")
print(df_combined.select_dtypes(include=np.number).describe().T[['mean', 'min', 'max', 'std']])

print("\n--- Riverside NOAA Descriptive Statistics ---")
print(df_riverside.select_dtypes(include=np.number).describe().T[['mean', 'min', 'max', 'std']])

print("\n--- San Bernardino NOAA Descriptive Statistics ---")
print(df_sb.select_dtypes(include=np.number).describe().T[['mean', 'min', 'max', 'std']])

print("\n--- Southern Edison Load Descriptive Statistics ---")
print(df_edison.select_dtypes(include=np.number).describe().T[['mean', 'min', 'max', 'std']])

# UNTRANSFORMED RAW SIGNALS
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Filter 2023 evaluation window
s_c   = df_combined[(df_combined['Date'] >= '2023-01-01') & (df_combined['Date'] <= '2023-12-31')]
s_riv = df_riverside[(df_riverside['Date'] >= '2023-01-01') & (df_riverside['Date'] <= '2023-12-31')]
s_sb  = df_sb[(df_sb['Date'] >= '2023-01-01') & (df_sb['Date'] <= '2023-12-31')]
s_ed  = df_edison[(df_edison['Date'] >= '2023-01-01') & (df_edison['Date'] <= '2023-12-31')]

# Subplot 1: Unscaled Meteorological Signals (°F)
ax1.plot(s_c['Date'], s_c['Max_Temp_F'], color='tab:red', alpha=0.6, label='Daily Weather Max Temp (°F)')
ax1.plot(s_riv['Date'], s_riv['NOAA_TMax'], color='tab:blue', linewidth=2, label='Riverside NOAA TMax (°F)')
ax1.plot(s_sb['Date'], s_sb['NOAA_TMax'], color='tab:green', linestyle='--', linewidth=2, label='San Bernardino NOAA TMax (°F)')
ax1.set_ylabel("Temperature (°F)")
ax1.set_title("Raw Signals: Meteorological Metrics vs. SCE Power Demand (2023 Window)")
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: Unscaled Power Grid Metric (Regional Total kWh / MW)
ax2.plot(s_ed['Date'], s_ed['Peak_Demand_MW'], color='tab:purple', linewidth=2.5, label='SCE Regional Energy Usage')
ax2.set_xlabel("Date")
ax2.set_ylabel("Energy Demand (kWh / MW)")
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
# Save raw plot into Phase2_Graphs
plt.savefig(os.path.join(GRAPH_DIR, "raw_signals_2023.png"))
plt.close()

# MISSING VALUES & DATA INTEGRITY
print("\n--- Missing Value Audit ---")
print("Combined Weather Missing Count:\n", df_combined.isnull().sum())
print("\nRiverside NOAA Missing Count:\n", df_riverside.isnull().sum())
print("\nSan Bernardino NOAA Missing Count:\n", df_sb.isnull().sum())
print("\nSouthern Edison Load Missing Count:\n", df_edison.isnull().sum())

# Sequential imputation
for df in [df_combined, df_riverside, df_sb, df_edison]:
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].ffill()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

# CATEGORICAL ENCODING & Z-SCORE STANDARDIZATION
for df in [df_combined, df_riverside, df_sb, df_edison]:
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

scaler = StandardScaler()

df_combined_clean = df_combined.copy()
num_c = df_combined_clean.select_dtypes(include=np.number).columns
df_combined_clean[num_c] = scaler.fit_transform(df_combined_clean[num_c])

df_riv_clean = df_riverside.copy()
num_r = df_riv_clean.select_dtypes(include=np.number).columns
df_riv_clean[num_r] = scaler.fit_transform(df_riv_clean[num_r])

df_sb_clean = df_sb.copy()
num_s = df_sb_clean.select_dtypes(include=np.number).columns
df_sb_clean[num_s] = scaler.fit_transform(df_sb_clean[num_s])

df_edison_clean = df_edison.copy()
num_e = df_edison_clean.select_dtypes(include=np.number).columns
df_edison_clean[num_e] = scaler.fit_transform(df_edison_clean[num_e])

# Export cleaned CSV files to Phase2_CleanData
df_combined_clean.to_csv(os.path.join(CLEAN_DIR, "combined_weather_cleaned.csv"), index=False)
df_riv_clean.to_csv(os.path.join(CLEAN_DIR, "riverside_noaa_cleaned.csv"), index=False)
df_sb_clean.to_csv(os.path.join(CLEAN_DIR, "san_bernardino_noaa_cleaned.csv"), index=False)
df_edison_clean.to_csv(os.path.join(CLEAN_DIR, "southernedisoncleaned.csv"), index=False)

print("\n✓ Phase 2 Complete! Successfully exported cleaned CSV files to Phase2_CleanData.")

# CLEANED SAMPLES & OVERLAID STANDARDIZED SIGNAL PLOTS
print("\n--- Cleaned Combined Weather Sample ---")
print(df_combined_clean.head(3))

print("\n--- Cleaned Riverside NOAA Sample ---")
print(df_riv_clean.head(3))

print("\n--- Cleaned San Bernardino NOAA Sample ---")
print(df_sb_clean.head(3))

print("\n--- Cleaned Southern Edison Sample ---")
print(df_edison_clean.head(3))

# Plot 7A: Overlaid Standardized Signals (2023 Zoom Window)
plt.figure(figsize=(12, 5))

sc_cw   = df_combined_clean[(df_combined_clean['Date'] >= '2023-01-01') & (df_combined_clean['Date'] <= '2023-12-31')]
sc_criv = df_riv_clean[(df_riv_clean['Date'] >= '2023-01-01') & (df_riv_clean['Date'] <= '2023-12-31')]
sc_csb  = df_sb_clean[(df_sb_clean['Date'] >= '2023-01-01') & (df_sb_clean['Date'] <= '2023-12-31')]
sc_ed   = df_edison_clean[(df_edison_clean['Date'] >= '2023-01-01') & (df_edison_clean['Date'] <= '2023-12-31')]

plt.plot(sc_cw['Date'], sc_cw['Max_Temp_F'], color='tab:red', alpha=0.5, label='Scaled Daily Weather')
plt.plot(sc_criv['Date'], sc_criv['NOAA_TMax'], color='tab:blue', linewidth=2, label='Scaled Riverside NOAA')
plt.plot(sc_csb['Date'], sc_csb['NOAA_TMax'], color='tab:green', linestyle='--', linewidth=2, label='Scaled San Bernardino NOAA')
plt.plot(sc_ed['Date'], sc_ed['Peak_Demand_MW'], color='tab:purple', linewidth=2.5, label='Scaled SCE Peak Load')

plt.title("Cleaned & Standardized Signals: Weather vs. Power (2023 Window)")
plt.xlabel("Date")
plt.ylabel("Standardized Value (Z-Score)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()

# Save Plot 7A into Phase2_Graphs
plt.savefig(os.path.join(GRAPH_DIR, "standardized_signals_2023.png"))
plt.close()


# Plot 7B: 10-Year Full Decade Timelapse (2014 - 2024 Trendlines)
plt.figure(figsize=(14, 5))

plt.plot(df_combined_clean['Date'], df_combined_clean['Max_Temp_F'], color='tab:red', alpha=0.25, label='Daily Temp Trend')
plt.plot(df_riv_clean['Date'], df_riv_clean['NOAA_TMax'], color='tab:blue', linewidth=1.5, label='Riverside NOAA Monthly')
plt.plot(df_sb_clean['Date'], df_sb_clean['NOAA_TMax'], color='tab:green', linestyle='--', linewidth=1.5, label='San Bernardino NOAA Monthly')
plt.plot(df_edison_clean['Date'], df_edison_clean['Peak_Demand_MW'], color='tab:purple', linewidth=2, label='SCE Power Demand Monthly')

plt.title("10-Year Full Timelapse: Standardized Weather & Energy Load Signals (2014 - 2024)")
plt.xlabel("Year")
plt.ylabel("Standardized Value (Z-Score)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()

# Save Plot 7B into Phase2_Graphs
plt.savefig(os.path.join(GRAPH_DIR, "decade_timelapse_2014_2024.png"))
plt.close()

print("\nAll graph outputs have been saved to the Phase2_Graphs folder.")
