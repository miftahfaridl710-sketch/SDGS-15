# SDGs 15 - Analisis Deforestasi & Emisi Karbon
# Versi Streamlit-friendly (tanpa path /content dan tanpa perintah shell)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import geopandas as gpd
import streamlit as st

# --- SETUP STREAMLIT PAGE ---
st.set_page_config(page_title="SDGs 15 - Deforestasi & Emisi", layout="wide")
st.title("🌳 Analisis Deforestasi dan Emisi Karbon Indonesia")
st.write("Data dan prediksi dari SDGs 15 (Kehidupan di Darat)")

# --- PATH SETUP ---
os.makedirs("data", exist_ok=True)

# URL Dataset (dari repo kamu)
url_excel = "https://raw.githubusercontent.com/miftahfaridl710-sketch/SDGS-15/main/IDN.xlsx"

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_excel(url_excel, sheet_name="Subnational 1 tree cover loss")
    df = df[df["threshold"] == 30].reset_index(drop=True)
    return df

df = load_data()
st.subheader("Cuplikan Data Asli")
st.dataframe(df.head())

# --- Transformasi Data ---
loss_cols = [col for col in df.columns if "tc_loss_ha_" in col]
df_long = df.melt(
    id_vars=["country", "subnational1", "threshold", "extent_2000_ha"],
    value_vars=loss_cols,
    var_name="year",
    value_name="loss_ha"
)
df_long["year"] = df_long["year"].str.extract(r"(\d{4})").astype(int)
df_long = df_long.sort_values(["subnational1", "year"]).reset_index(drop=True)
df_long["loss_rate_%"] = (df_long["loss_ha"] / df_long["extent_2000_ha"]) * 100

st.subheader("Data Setelah Transformasi (Long Format)")
st.dataframe(df_long.head())

# --- Tren Nasional ---
national_trend = (
    df_long.groupby("year")["loss_ha"].sum().reset_index()
)
fig_trend = px.line(national_trend, x="year", y="loss_ha",
                    title="Tren Kehilangan Tutupan Hutan Nasional (2001–2024)",
                    labels={"loss_ha": "Luas Kehilangan (ha)", "year": "Tahun"},
                    markers=True)
st.plotly_chart(fig_trend, use_container_width=True)

# --- Top 5 Provinsi ---
top5 = (
    df_long.groupby("subnational1")["loss_ha"].mean()
    .sort_values(ascending=False)
    .head(5)
    .index
)

fig_top = px.line(
    df_long[df_long["subnational1"].isin(top5)],
    x="year",
    y="loss_ha",
    color="subnational1",
    markers=True,
    title="Tren Kehilangan Hutan di 5 Provinsi Teratas"
)
st.plotly_chart(fig_top, use_container_width=True)

# --- MODELING ---
df_model = df_long.copy()
df_model["lag_1"] = df_model.groupby("subnational1")["loss_ha"].shift(1)
df_model["lag_2"] = df_model.groupby("subnational1")["loss_ha"].shift(2)
df_model["lag_3"] = df_model.groupby("subnational1")["loss_ha"].shift(3)
df_model = df_model.dropna()

le = LabelEncoder()
df_model["prov_enc"] = le.fit_transform(df_model["subnational1"])
FEATURES = ["lag_1", "lag_2", "lag_3", "year", "prov_enc"]
TARGET = "loss_ha"

train_df = df_model[df_model["year"] <= 2021]
test_df = df_model[df_model["year"] > 2021]

model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
model.fit(train_df[FEATURES], train_df[TARGET])
preds = model.predict(test_df[FEATURES])

mse = mean_squared_error(test_df[TARGET], preds)
rmse = np.sqrt(mse)
r2 = r2_score(test_df[TARGET], preds)

st.subheader("📊 Evaluasi Model XGBoost")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**R²:** {r2:.3f}")

# --- Forecast 2025–2027 ---
forecast_years = [2025, 2026, 2027]
forecast_rows = []

for prov, series in df_model.groupby("subnational1"):
    series = series.sort_values("year").copy()
    for year in forecast_years:
        lag_1 = series["loss_ha"].iloc[-1]
        lag_2 = series["loss_ha"].iloc[-2]
        lag_3 = series["loss_ha"].iloc[-3]
        prov_enc = le.transform([prov])[0]
        f = pd.DataFrame([[lag_1, lag_2, lag_3, year, prov_enc]], columns=FEATURES)
        pred = model.predict(f)[0]
        forecast_rows.append({"subnational1": prov, "year": year, "pred_loss_ha": pred})

forecast_df = pd.DataFrame(forecast_rows)
st.subheader("🔮 Prediksi Deforestasi (2025–2027)")
st.dataframe(forecast_df.head())

fig_forecast = px.line(forecast_df, x="year", y="pred_loss_ha", color="subnational1",
                       title="Prediksi Deforestasi per Provinsi (2025–2027)")
st.plotly_chart(fig_forecast, use_container_width=True)

# --- Save outputs locally ---
forecast_df.to_csv("data/forecast_deforestation_2025_2027.csv", index=False)
st.success("✅ Hasil prediksi disimpan ke 'data/forecast_deforestation_2025_2027.csv'")
