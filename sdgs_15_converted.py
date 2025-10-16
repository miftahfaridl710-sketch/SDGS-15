# Auto-converted from notebook: sdgs_15 (3).ipynb

# --- BEGIN CONVERTED CELLS ---


# ---- cell 1 ----

# Install & Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score



# ---- cell 2 ----

# Tampilan dataframe yang lebih rapi
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')


# ---- cell 3 ----

# Load Dataset dari Google Drive / Colab Files
file_path = "/content/IDN.xlsx"

# Baca sheet Subnational 1 tree cover loss
df = pd.read_excel(file_path, sheet_name="Subnational 1 tree cover loss")

print("Dataset loaded with shape:", df.shape)
df.head()


# ---- cell 4 ----

# Filter threshold 30% (standar Global Forest Watch)
df = df[df["threshold"] == 30].reset_index(drop=True)
print("Filtered rows (threshold=30%):", len(df))


# ---- cell 5 ----

# Ubah format wide → long (yearly time series)
# Kolom yang memuat data loss per tahun (2001–2024)
loss_cols = [col for col in df.columns if "tc_loss_ha_" in col]

# Ubah menjadi long format
df_long = df.melt(
    id_vars=["country", "subnational1", "threshold", "extent_2000_ha"],
    value_vars=loss_cols,
    var_name="year",
    value_name="loss_ha"
)

# Ambil angka tahun dari nama kolom
df_long["year"] = df_long["year"].str.extract(r"(\d{4})").astype(int)

# Urutkan
df_long = df_long.sort_values(["subnational1", "year"]).reset_index(drop=True)

print("Long-format dataset ready:", df_long.shape)
df_long.head()


# ---- cell 6 ----

# Hitung Laju Deforestasi Tahunan (% dari luas awal)
# Gunakan extent_2000_ha sebagai baseline luas hutan
df_long["loss_rate_%"] = (df_long["loss_ha"] / df_long["extent_2000_ha"]) * 100

# Cek total per tahun (nasional)
national_trend = (
    df_long.groupby("year")["loss_ha"].sum().reset_index()
)
national_trend["loss_rate_%"] = national_trend["loss_ha"] / df_long["extent_2000_ha"].sum() * 100

national_trend.head()


# ---- cell 7 ----

# Visualisasi Contoh Per Provinsi
# Pilih 5 provinsi dengan rata-rata loss terbesar
top5 = (
    df_long.groupby("subnational1")["loss_ha"].mean()
    .sort_values(ascending=False)
    .head(5)
    .index
)

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_long[df_long["subnational1"].isin(top5)],
    x="year",
    y="loss_ha",
    hue="subnational1",
    marker="o"
)
plt.title("Tren Kehilangan Hutan di 5 Provinsi Teratas (2001–2024)")
plt.ylabel("Luas kehilangan (ha)")
plt.xlabel("Tahun")
plt.legend(title="Provinsi")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# ---- cell 8 ----

output_path = "/content/deforestation_ready.csv"
df_long.to_csv(output_path, index=False)
print("File siap modelling disimpan di:", output_path)


# ---- cell 9 ----

# Baca sheet Subnational 1 tree cover loss
df_carbon = pd.read_excel(file_path, sheet_name="Subnational 1 carbon data")

print("Dataset loaded with shape:", df.shape)
df_carbon = df_carbon[df_carbon["umd_tree_cover_density_2000__threshold"] == 30]
df_carbon.head()


# ---- cell 10 ----

# Memilih Kolom Carbon (stok karbon & emisi tahunan)
carbon_cols = [
    "country", "subnational1", "umd_tree_cover_density_2000__threshold",
    "gfw_aboveground_carbon_stocks_2000__Mg_C",
    "avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1"
]


# ---- cell 11 ----

# Kolom emisi tahunan (filter only columns ending with a year)
emission_cols = [c for c in df_carbon.columns if "gross_emissions_" in c]
carbon_long = df_carbon.melt(
    id_vars=carbon_cols,
    value_vars=emission_cols,
    var_name="year",
    value_name="emission_Mg_CO2e"
)


# ---- cell 12 ----

# Ambil angka tahun dari nama kolom
carbon_long["year"] = carbon_long["year"].str.extract(r"(\d{4})")

# Hapus baris tanpa angka tahun (NaN)
carbon_long = carbon_long.dropna(subset=["year"]).reset_index(drop=True)

# Konversi ke integer
carbon_long["year"] = carbon_long["year"].astype(int)
carbon_long["year"].unique()[:10]


# ---- cell 13 ----

# Gabungkan dengan tree cover loss
merged = pd.merge(
    df_long,
    carbon_long,
    left_on=["subnational1", "year"],
    right_on=["subnational1", "year"],
    how="left"
)

print("Dataset gabungan deforestasi + karbon:", merged.shape)
merged.head()


# ---- cell 14 ----

# Visualisasi Tren Tree Cover Loss Nasional
plt.figure(figsize=(10, 5))
sns.lineplot(data=national_trend, x="year", y="loss_ha", marker="o", color="darkgreen")
plt.title("Tren Kehilangan Tutupan Hutan Indonesia (2001–2024)")
plt.ylabel("Luas kehilangan (ha)")
plt.xlabel("Tahun")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# ---- cell 15 ----

# Total emisi karbon per tahun di Indonesia
carbon_trend = merged.groupby("year")["emission_Mg_CO2e"].sum().reset_index()

plt.figure(figsize=(10,5))
sns.lineplot(data=carbon_trend, x="year", y="emission_Mg_CO2e", color="red", marker="o")
plt.title("Tren Emisi Karbon Akibat Deforestasi (2001–2024)")
plt.xlabel("Tahun")
plt.ylabel("Emisi (Mg CO2e)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# ---- cell 16 ----

cols_keep = [
    "subnational1", "year",
    "extent_2000_ha", "loss_ha", "loss_rate_%",
    "gfw_aboveground_carbon_stocks_2000__Mg_C",
    "avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1",
    "emission_Mg_CO2e"
]

df_clean = merged[cols_keep].copy()


# ---- cell 17 ----

# Menambahkan Kolom Estimasi Karbon Hilang (C → CO₂e)
df_clean["emission_estimated_CO2e"] = (
    df_clean["loss_ha"] *
    df_clean["avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1"] *
    3.67 # konversi C ke CO₂e
)


# ---- cell 18 ----

# Cek jumlah duplikat berdasarkan provinsi dan tahun
df_clean.duplicated(subset=["subnational1", "year"]).sum()


# ---- cell 19 ----

df_clean.head()


# ---- cell 20 ----

df_clean["carbon_loss_MgC"] = df_clean["loss_ha"] * df_clean["avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1"] # menghitung carbon loss

df_clean[["subnational1", "year", "loss_ha", "carbon_loss_MgC", "emission_estimated_CO2e"]].head()


# ---- cell 21 ----

# Visualisasi Tren Emisi Carbon dengan Tree Cover Loss Nasional
national = (
    df_clean.groupby("year")[["loss_ha", "emission_estimated_CO2e"]]
    .sum().reset_index()
)

fig, ax1 = plt.subplots(figsize=(10, 5))

color = "tab:green"
ax1.set_xlabel("Tahun")
ax1.set_ylabel("Kehilangan Hutan (ha)", color=color)
ax1.plot(national["year"], national["loss_ha"], color=color, marker="o", label="Deforestasi (ha)")
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Emisi Karbon (CO₂e, Mg)", color=color)
ax2.plot(national["year"], national["emission_estimated_CO2e"], color=color, marker="x", linestyle="--", label="Emisi CO₂e")
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Tren Deforestasi dan Emisi Karbon Indonesia (2001–2024)")
fig.tight_layout()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# ---- cell 22 ----

output_path = "/content/deforestation_carbon_ready.csv"
df_clean.to_csv(output_path, index=False)
print("Dataset disimpan di:", output_path)


# ---- cell 23 ----

# Pastikan kolom year bertipe int
df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce').astype(int)


# ---- cell 24 ----

# Feature engineering: lags, rolling
df = df_clean.copy().sort_values(['subnational1','year']).reset_index(drop=True)

# Create lags and rolling per province
lags = [1,2,3]
roll_windows = [3]

for lag in lags:
    df[f'lag_{lag}'] = df.groupby('subnational1')['loss_ha'].shift(lag)

for w in roll_windows:
    df[f'roll_mean_{w}'] = df.groupby('subnational1')['loss_ha'].shift(1).rolling(w).mean()


# ---- cell 25 ----

# We'll drop initial rows with NaN in lag_1 to keep training clean
df = df.dropna(subset=[f'lag_{l}' for l in [1]]).reset_index(drop=True)

# Encode province (label encode)
le = LabelEncoder()
df['prov_enc'] = le.fit_transform(df['subnational1'].astype(str))

# Ensure numeric columns exist
df['extent_2000_ha'] = pd.to_numeric(df['extent_2000_ha'], errors='coerce').fillna(0)
df['avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1'] = pd.to_numeric(
    df['avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1'], errors='coerce').fillna(0)

# Feature list
FEATURES = [
    'lag_1','lag_2','lag_3',
    'roll_mean_3',
    'year',
    'extent_2000_ha',
    'avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1',
    'prov_enc'
]

# If some features missing (e.g., lag_3), drop them from list
FEATURES = [f for f in FEATURES if f in df.columns]

# Target
TARGET = 'loss_ha'


# ---- cell 26 ----

train_df = df[df['year'] <= 2021].copy()
val_df = df[(df['year'] >= 2022) & (df['year'] <= 2023)].copy()
test_df = df[df['year'] == 2024].copy()

print("Sizes: train", train_df.shape, "val", val_df.shape, "test", test_df.shape)


# ---- cell 27 ----

# Train
X_train = train_df[FEATURES]
y_train = train_df[TARGET]

# Validation
X_val = val_df[FEATURES]
y_val = val_df[TARGET]

# Test
X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ---- cell 28 ----

model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist'
)


# ---- cell 29 ----

# Konversi DataFrame ke DMatrix (format internal XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val, label=y_val)
dtest  = xgb.DMatrix(X_test, label=y_test)

# Parameter model
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

# List evaluasi
evals = [(dtrain, "train"), (dval, "val")]


# ---- cell 30 ----

# Train model dengan early stopping
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=50
)


# ---- cell 31 ----

# Prediksi
preds = model.predict(dtest)


# ---- cell 32 ----

model_path = "/content/model_xgb_deforestasi.json"


# ---- cell 33 ----

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

print(f"RMSE: {rmse:.2f}")


# ---- cell 34 ----

r2 = r2_score(y_test, preds)
nrmse = rmse / (y_test.max() - y_test.min())

print(f"R²: {r2:.3f}")
print(f"Normalized RMSE: {nrmse:.3f}")


# ---- cell 35 ----

# Tambahkan fitur lag dan rolling mean ke df historis
df = df.sort_values(["subnational1", "year"])
df["lag_1"] = df.groupby("subnational1")["loss_ha"].shift(1)
df["lag_2"] = df.groupby("subnational1")["loss_ha"].shift(2)
df["lag_3"] = df.groupby("subnational1")["loss_ha"].shift(3)
df["roll_mean_3"] = df.groupby("subnational1")["loss_ha"].rolling(3).mean().reset_index(level=0, drop=True)

# Drop baris awal yang tidak punya lag lengkap
df = df.dropna(subset=["lag_1", "lag_2", "lag_3", "roll_mean_3"])


# ---- cell 36 ----

forecast_years = [2025, 2026, 2027]
forecast_rows = []

for prov, series in df.groupby("subnational1"):
    series = series.sort_values("year").copy()

    for year in forecast_years:
        lag_1 = series["loss_ha"].iloc[-1]
        lag_2 = series["loss_ha"].iloc[-2]
        lag_3 = series["loss_ha"].iloc[-3]
        roll_mean_3 = series["loss_ha"].iloc[-3:].mean()

        f = {
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "roll_mean_3": roll_mean_3,
            "year": year,
            "extent_2000_ha": series["extent_2000_ha"].iloc[-1],
            "avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1": series["avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1"].iloc[-1],
            "prov_enc": series["prov_enc"].iloc[-1]
        }

        dnext = xgb.DMatrix(pd.DataFrame([f])[FEATURES])
        pred_loss = model.predict(dnext)[0]

        emission = pred_loss * f["avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1"] * 3.67

        forecast_rows.append({
            "subnational1": prov,
            "year": year,
            "pred_loss_ha": pred_loss,
            "emission_estimated_CO2e": emission
        })

        # update untuk tahun berikutnya
        new_row = f.copy()
        new_row["loss_ha"] = pred_loss
        series = pd.concat([series, pd.DataFrame([new_row])], ignore_index=True)

forecast_full = pd.DataFrame(forecast_rows)


# ---- cell 37 ----

# Agregasi Nasional
# Prediksi historis
dall = xgb.DMatrix(df[FEATURES])
df["pred_loss_ha"] = model.predict(dall)
df["emission_estimated_CO2e"] = (
    df["pred_loss_ha"] * df["avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1"] * 3.67
)

national_hist = df.groupby("year")[["loss_ha", "pred_loss_ha", "emission_estimated_CO2e"]].sum().reset_index()
national_forecast = forecast_full.groupby("year")[["pred_loss_ha", "emission_estimated_CO2e"]].sum().reset_index()

trend_all = pd.concat([
    national_hist,
    national_forecast
], ignore_index=True).sort_values("year")


# ---- cell 38 ----

# Simpan Output
forecast_full.to_csv("/content/forecast_full_2025_2027.csv", index=False)
trend_all.to_csv("/content/trend_deforestasi_emisi_2001_2027.csv", index=False)

print("File disimpan:")
print(" - forecast_full_2025_2027.csv")
print(" - trend_deforestasi_emisi_2001_2027.csv")


# ---- cell 39 ----

# Import Plotly dan Siapkan Data
import plotly.express as px
import plotly.graph_objects as go

# Baca hasil agregasi nasional (bisa juga dari trend_all variabel)
trend_all = pd.read_csv("/content/trend_deforestasi_emisi_2001_2027.csv")


# ---- cell 40 ----

fig_def = px.line(
    trend_all,
    x="year",
    y="pred_loss_ha",
    title="Tren Kehilangan Tutupan Hutan Indonesia (2001–2027)",
    labels={"year": "Tahun", "pred_loss_ha": "Luas Kehilangan Hutan (ha)"},
    markers=True
)

fig_def.update_traces(line_color="green", line_width=3)
fig_def.update_layout(
    template="plotly_white",
    hovermode="x unified",
    font=dict(size=13),
    title_font=dict(size=18),
    width=900,
    height=500
)

fig_def.show()


# ---- cell 41 ----

fig_emis = px.line(
    trend_all,
    x="year",
    y="emission_estimated_CO2e",
    title="Prediksi Emisi Karbon akibat Deforestasi (2001–2027)",
    labels={"year": "Tahun", "emission_estimated_CO2e": "Emisi Karbon (CO₂e, Mg)"},
    markers=True
)

fig_emis.update_traces(line_color="red", line_width=3, line_dash="dot")
fig_emis.update_layout(
    template="plotly_white",
    hovermode="x unified",
    font=dict(size=13),
    title_font=dict(size=18),
    width=900,
    height=500
)

fig_emis.show()


# ---- cell 42 ----

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Baca hasil forecast
forecast = pd.read_csv("/content/forecast_full_2025_2027.csv")

# Filter tahun 2027
forecast_2027 = forecast[forecast["year"] == 2027].copy()


# ---- cell 43 ----

import os
import urllib.request

url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_IDN_shp.zip"
output_path = "idn_shapefile.zip"

if not os.path.exists(output_path):
    urllib.request.urlretrieve(url, output_path)

# Baca shapefile
gdf = gpd.read_file("/content/idn_shapefile/gadm41_IDN_1.shp")


# ---- cell 44 ----

# Gabungkan Data Provinsi dengan Forecast
# Pastikan nama kolom sesuai (provinsi dalam shapefile & forecast)
gdf["NAME_1"] = gdf["NAME_1"].str.title()
forecast_2027["subnational1"] = forecast_2027["subnational1"].str.title()

# Merge berdasarkan nama provinsi
merged_map = gdf.merge(
    forecast_2027,
    how="left",
    left_on="NAME_1",
    right_on="subnational1"
)


# ---- cell 45 ----

# Peta Persebaran Deforestasi (ha)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
merged_map.plot(
    column="pred_loss_ha",
    cmap="YlGn",
    linewidth=0.5,
    edgecolor="gray",
    legend=True,
    legend_kwds={"label": "Predicted Deforestation (ha)", "orientation": "vertical"},
    ax=ax
)
ax.set_title("Persebaran Deforestasi Indonesia Tahun 2027", fontsize=15)
ax.axis("off")
plt.tight_layout()
plt.savefig("/content/peta_deforestasi_2027.png", dpi=200)
plt.show()


# ---- cell 46 ----

# Peta Persebaran Emisi Karbon (CO₂e)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
merged_map.plot(
    column="emission_estimated_CO2e",
    cmap="OrRd",
    linewidth=0.5,
    edgecolor="gray",
    legend=True,
    legend_kwds={"label": "Predicted Carbon Emission (CO₂e, Mg)", "orientation": "vertical"},
    ax=ax
)
ax.set_title("Persebaran Emisi Karbon akibat Deforestasi Tahun 2027", fontsize=15)
ax.axis("off")
plt.tight_layout()
plt.savefig("/content/peta_emisi_karbon_2027.png", dpi=200)
plt.show()


# --- END CONVERTED CELLS ---
