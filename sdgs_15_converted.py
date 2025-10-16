# sdgs_15_dashboard.py
# 🌳 Dashboard Analisis Deforestasi & Emisi Karbon Indonesia (SDGs 15)
# Versi final: membaca shapefile langsung dari root repo

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="SDGs 15 Dashboard", layout="wide")
st.title("🌳 Dashboard Analisis Deforestasi & Emisi Karbon Indonesia")
st.write("Visualisasi data, prediksi, dan persebaran deforestasi serta emisi karbon berdasarkan data SDGs 15.")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/miftahfaridl710-sketch/SDGS-15/main/IDN.xlsx"
    df = pd.read_excel(url, sheet_name="Subnational 1 tree cover loss", engine="openpyxl")
    df = df[df["threshold"] == 30].reset_index(drop=True)
    loss_cols = [c for c in df.columns if c.startswith("tc_loss_ha_")]
    df_long = df.melt(
        id_vars=["country", "subnational1", "extent_2000_ha"],
        value_vars=loss_cols,
        var_name="year",
        value_name="loss_ha"
    )
    df_long["year"] = df_long["year"].str.extract(r"(\d{4})").astype(int)
    df_long["loss_rate_%"] = (df_long["loss_ha"] / df_long["extent_2000_ha"]) * 100
    return df, df_long

df, df_long = load_data()

# ------------------------------------------------------------
# 1️⃣ Tren Kehilangan Tutupan Hutan Nasional
# ------------------------------------------------------------
st.header("🇮🇩 Tren Kehilangan Tutupan Hutan di Indonesia (2001–2024)")
st.write("Menunjukkan total kehilangan tutupan hutan setiap tahun secara nasional berdasarkan data seluruh provinsi.")

national_trend = df_long.groupby("year", as_index=False)["loss_ha"].sum()
fig1 = px.line(
    national_trend, x="year", y="loss_ha", markers=True,
    title="Tren Kehilangan Tutupan Hutan Nasional (2001–2024)",
    labels={"loss_ha": "Luas Kehilangan (ha)", "year": "Tahun"}
)
st.plotly_chart(fig1, use_container_width=True)

st.download_button(
    "📥 Unduh Data Tren Nasional (CSV)",
    data=national_trend.to_csv(index=False).encode(),
    file_name="tren_kehilangan_hutan_nasional.csv",
    mime="text/csv"
)

# ------------------------------------------------------------
# 2️⃣ 5 Provinsi Teratas
# ------------------------------------------------------------
st.header("🌲 Tren Kehilangan Tutupan Hutan di 5 Provinsi Teratas")
st.write("Menampilkan provinsi dengan rata-rata kehilangan hutan tahunan terbesar selama 2001–2024.")

top5 = df_long.groupby("subnational1")["loss_ha"].mean().nlargest(5).index
df_top5 = df_long[df_long["subnational1"].isin(top5)]
fig2 = px.line(
    df_top5, x="year", y="loss_ha", color="subnational1", markers=True,
    title="Tren Kehilangan Hutan di 5 Provinsi Teratas (2001–2024)",
    labels={"loss_ha": "Luas Kehilangan (ha)", "subnational1": "Provinsi"}
)
st.plotly_chart(fig2, use_container_width=True)

st.download_button(
    "📥 Unduh Data 5 Provinsi Teratas (CSV)",
    data=df_top5.to_csv(index=False).encode(),
    file_name="top5_provinsi_kehilangan.csv",
    mime="text/csv"
)

# ------------------------------------------------------------
# 3️⃣ Estimasi Emisi Karbon
# ------------------------------------------------------------
st.header("💨 Tren Emisi Karbon Akibat Deforestasi")
st.write("Emisi karbon diestimasi berdasarkan kehilangan hutan dikalikan dengan stok karbon rata-rata dan faktor konversi 3.67 (CO₂ per C).")

if "avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1" in df.columns:
    df_merge = df_long.merge(
        df[["subnational1", "avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1"]].drop_duplicates(),
        on="subnational1", how="left"
    )
    df_merge["emission_CO2e"] = df_merge["loss_ha"] * df_merge["avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1"] * 3.67
else:
    df_merge = df_long.copy()
    df_merge["emission_CO2e"] = df_merge["loss_ha"] * 50 * 3.67  # asumsi kasar

carbon_trend = df_merge.groupby("year", as_index=False)["emission_CO2e"].sum()
fig3 = px.line(
    carbon_trend, x="year", y="emission_CO2e", markers=True,
    title="Tren Emisi Karbon Akibat Deforestasi (2001–2024)",
    labels={"emission_CO2e": "Emisi (Mg CO₂e)", "year": "Tahun"}
)
st.plotly_chart(fig3, use_container_width=True)

st.download_button(
    "📥 Unduh Data Emisi Karbon (CSV)",
    data=carbon_trend.to_csv(index=False).encode(),
    file_name="tren_emisi_karbon.csv",
    mime="text/csv"
)

# ------------------------------------------------------------
# 4️⃣ Tren Gabungan Deforestasi & Emisi
# ------------------------------------------------------------
st.header("📊 Tren Deforestasi dan Emisi Karbon di Indonesia")
st.write("Visualisasi ini memperlihatkan hubungan antara kehilangan hutan dan emisi karbon setiap tahun.")

combo = pd.merge(national_trend, carbon_trend, on="year")
fig4 = go.Figure()
fig4.add_trace(go.Bar(x=combo["year"], y=combo["loss_ha"], name="Kehilangan Hutan (ha)", marker_color="green", opacity=0.6))
fig4.add_trace(go.Line(x=combo["year"], y=combo["emission_CO2e"], name="Emisi Karbon (Mg CO₂e)", marker_color="red"))
fig4.update_layout(
    title="Tren Deforestasi dan Emisi Karbon (2001–2024)",
    yaxis=dict(title="Kehilangan Hutan (ha)"),
    yaxis2=dict(title="Emisi (Mg CO₂e)", overlaying="y", side="right")
)
st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------------------------
# 5️⃣ Modeling & Prediksi
# ------------------------------------------------------------
st.header("🔮 Prediksi Kehilangan Tutupan Hutan dan Emisi (2025–2027)")
st.write("Model XGBoost digunakan untuk memprediksi tren deforestasi dan emisi hingga tahun 2027.")

df_model = df_merge.copy()
df_model["lag1"] = df_model.groupby("subnational1")["loss_ha"].shift(1)
df_model["lag2"] = df_model.groupby("subnational1")["loss_ha"].shift(2)
df_model = df_model.dropna()

le = LabelEncoder()
df_model["prov_enc"] = le.fit_transform(df_model["subnational1"])

FEATURES = ["lag1", "lag2", "year", "prov_enc"]
TARGET = "loss_ha"

train = df_model[df_model["year"] <= 2021]
test = df_model[df_model["year"] > 2021]
model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
model.fit(train[FEATURES], train[TARGET])
preds = model.predict(test[FEATURES])

rmse = np.sqrt(mean_squared_error(test[TARGET], preds))
r2 = r2_score(test[TARGET], preds)
st.metric("RMSE", f"{rmse:,.2f}")
st.metric("R²", f"{r2:.3f}")

st.write("**Interpretasi:** Semakin tinggi R² (mendekati 1), semakin baik model menjelaskan variasi data historis.")

# Forecast 2025–2027
future_years = [2025, 2026, 2027]
forecast_rows = []
for prov in df_model["subnational1"].unique():
    prov_df = df_model[df_model["subnational1"] == prov].sort_values("year")
    lag1, lag2 = prov_df.iloc[-1]["loss_ha"], prov_df.iloc[-2]["loss_ha"]
    prov_enc = le.transform([prov])[0]
    for y in future_years:
        x = pd.DataFrame([[lag1, lag2, y, prov_enc]], columns=FEATURES)
        pred = model.predict(x)[0]
        emission = pred * 50 * 3.67
        forecast_rows.append({"subnational1": prov, "year": y, "pred_loss_ha": pred, "emission_pred": emission})
        lag1, lag2 = pred, lag1

forecast_df = pd.DataFrame(forecast_rows)
st.download_button(
    "📥 Unduh Data Prediksi (CSV)",
    data=forecast_df.to_csv(index=False).encode(),
    file_name="prediksi_deforestasi_2025_2027.csv",
    mime="text/csv"
)

# Plot Nasional (2001–2027)
national_pred = (
    pd.concat([
        national_trend.rename(columns={"loss_ha": "pred_loss_ha"}),
        forecast_df.groupby("year", as_index=False)["pred_loss_ha"].sum()
    ])
    .groupby("year", as_index=False)["pred_loss_ha"].sum()
)
fig5 = px.line(national_pred, x="year", y="pred_loss_ha", markers=True,
               title="Tren Kehilangan Tutupan Hutan di Indonesia (2001–2027)",
               labels={"pred_loss_ha": "Luas Kehilangan (ha)"})
st.plotly_chart(fig5, use_container_width=True)

# ------------------------------------------------------------
# 6️⃣ Peta Persebaran Tahun 2027
# ------------------------------------------------------------
st.header("🗺️ Persebaran Deforestasi dan Emisi Karbon Tahun 2027")
st.write("Menampilkan prediksi deforestasi dan emisi karbon pada tahun 2027 berdasarkan provinsi.")

shp_path = "gadm41_IDN_1.shp"
gdf = gpd.read_file(shp_path)
gdf["NAME_1"] = gdf["NAME_1"].str.title().str.strip()
forecast_df["subnational1"] = forecast_df["subnational1"].str.title().str.strip()
map_2027 = forecast_df[forecast_df["year"] == 2027]
gdf_merged = gdf.merge(map_2027, left_on="NAME_1", right_on="subnational1", how="left")

# Peta deforestasi
fig6, ax = plt.subplots(1, 2, figsize=(14, 7))
gdf_merged.plot(column="pred_loss_ha", cmap="YlGn", legend=True, ax=ax[0])
ax[0].set_title("Persebaran Deforestasi Tahun 2027")
ax[0].axis("off")
gdf_merged.plot(column="emission_pred", cmap="OrRd", legend=True, ax=ax[1])
ax[1].set_title("Persebaran Emisi Karbon Tahun 2027")
ax[1].axis("off")
st.pyplot(fig6)

st.success("✅ Dashboard berhasil dimuat sepenuhnya.")
