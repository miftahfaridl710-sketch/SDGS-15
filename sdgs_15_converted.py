# 🌳 SDGs 15 Dashboard — Analisis Deforestasi & Emisi Karbon Indonesia
# Interaktif, informatif, dan siap untuk publik
# Miftah Faridl © 2025

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# 🎨 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="SDGs 15 Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center; color:#2E8B57;'>🌳 SDGs 15 — Kehidupan di Darat</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Analisis Deforestasi dan Emisi Karbon Indonesia (2001–2027)</h3>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# 📂 LOAD DATA
# -------------------------------
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

# Sidebar filter
st.sidebar.header("⚙️ Filter Data")
years = sorted(df_long["year"].unique())
year_selected = st.sidebar.slider("Pilih Rentang Tahun", int(min(years)), int(max(years)), (2001, 2024))
provinsi_selected = st.sidebar.multiselect("Pilih Provinsi", sorted(df_long["subnational1"].unique()), default=None)

filtered = df_long[(df_long["year"].between(year_selected[0], year_selected[1]))]
if provinsi_selected:
    filtered = filtered[filtered["subnational1"].isin(provinsi_selected)]

# -------------------------------
# 📊 TAB NAVIGATION
# -------------------------------
tabs = st.tabs(["🏞 Kehilangan Hutan", "💨 Emisi Karbon", "📈 Model & Prediksi", "🗺 Persebaran"])

# -------------------------------
# 🏞 TAB 1: Kehilangan Hutan
# -------------------------------
with tabs[0]:
    st.subheader("Tren Kehilangan Tutupan Hutan Nasional")
    st.write("Visualisasi total kehilangan tutupan hutan di Indonesia dari tahun 2001 hingga 2024.")

    national_trend = filtered.groupby("year", as_index=False)["loss_ha"].sum()
    fig1 = px.line(
        national_trend, x="year", y="loss_ha", markers=True,
        color_discrete_sequence=["green"],
        labels={"loss_ha": "Luas Kehilangan (ha)", "year": "Tahun"},
        title="Tren Kehilangan Tutupan Hutan di Indonesia"
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.download_button("📥 Unduh Data", national_trend.to_csv(index=False).encode(), "tren_nasional.csv", "text/csv")

    st.markdown("**Insight:** Kecenderungan meningkatnya kehilangan tutupan hutan menunjukkan tekanan besar terhadap ekosistem darat. "
                "Periode 2015–2017 menjadi puncak kehilangan hutan akibat deforestasi masif di beberapa provinsi.")

    # 5 provinsi teratas
    st.subheader("Top 5 Provinsi dengan Kehilangan Hutan Tertinggi")
    top5 = filtered.groupby("subnational1")["loss_ha"].mean().nlargest(5).index
    df_top5 = filtered[filtered["subnational1"].isin(top5)]
    fig2 = px.line(df_top5, x="year", y="loss_ha", color="subnational1", markers=True,
                   title="Tren Kehilangan Hutan di 5 Provinsi Teratas")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# 💨 TAB 2: Emisi Karbon
# -------------------------------
with tabs[1]:
    st.subheader("Tren Emisi Karbon Akibat Deforestasi")
    st.write("Emisi karbon diestimasi berdasarkan kehilangan hutan × stok karbon rata-rata × faktor 3.67.")

    df_merge = df_long.copy()
    df_merge["emission_CO2e"] = df_merge["loss_ha"] * 50 * 3.67
    carbon_trend = df_merge.groupby("year", as_index=False)["emission_CO2e"].sum()

    fig3 = px.area(carbon_trend, x="year", y="emission_CO2e", title="Tren Emisi Karbon Akibat Deforestasi (2001–2024)",
                   labels={"emission_CO2e": "Emisi (Mg CO₂e)", "year": "Tahun"}, color_discrete_sequence=["#e85d04"])
    st.plotly_chart(fig3, use_container_width=True)

    st.info("📈 **Insight:** Meningkatnya emisi karbon sejalan dengan tingginya laju kehilangan tutupan hutan. "
            "Provinsi dengan deforestasi besar seperti Kalimantan Tengah dan Papua berkontribusi signifikan terhadap total emisi nasional.")

# -------------------------------
# 📈 TAB 3: Model & Prediksi
# -------------------------------
with tabs[2]:
    st.subheader("Model Prediksi Deforestasi (XGBoost)")
    st.write("Model ini memprediksi tren kehilangan hutan hingga tahun 2027 menggunakan algoritma XGBoost.")

    df_model = df_merge.copy()
    df_model["lag1"] = df_model.groupby("subnational1")["loss_ha"].shift(1)
    df_model["lag2"] = df_model.groupby("subnational1")["loss_ha"].shift(2)
    df_model.dropna(inplace=True)

    le = LabelEncoder()
    df_model["prov_enc"] = le.fit_transform(df_model["subnational1"])
    FEATURES = ["lag1", "lag2", "year", "prov_enc"]
    TARGET = "loss_ha"

    train = df_model[df_model["year"] <= 2021]
    test = df_model[df_model["year"] > 2021]
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=4)
    model.fit(train[FEATURES], train[TARGET])
    preds = model.predict(test[FEATURES])

    rmse = np.sqrt(mean_squared_error(test[TARGET], preds))
    r2 = r2_score(test[TARGET], preds)

    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:,.2f}")
    col2.metric("R²", f"{r2:.3f}")

    st.markdown("**Interpretasi:** Nilai R² mendekati 1 menunjukkan model mampu menjelaskan pola deforestasi dengan baik. "
                "RMSE mengukur deviasi rata-rata antara hasil prediksi dan data aktual.")

    # Prediksi masa depan
    forecast_rows = []
    for prov in df_model["subnational1"].unique():
        prov_df = df_model[df_model["subnational1"] == prov].sort_values("year")
        lag1, lag2 = prov_df.iloc[-1]["loss_ha"], prov_df.iloc[-2]["loss_ha"]
        prov_enc = le.transform([prov])[0]
        for y in [2025, 2026, 2027]:
            pred = model.predict(pd.DataFrame([[lag1, lag2, y, prov_enc]], columns=FEATURES))[0]
            forecast_rows.append({"subnational1": prov, "year": y, "pred_loss_ha": pred})
            lag1, lag2 = pred, lag1

    forecast_df = pd.DataFrame(forecast_rows)
    national_forecast = forecast_df.groupby("year", as_index=False)["pred_loss_ha"].sum()
    fig4 = px.line(national_forecast, x="year", y="pred_loss_ha", markers=True,
                   title="Prediksi Kehilangan Tutupan Hutan (2025–2027)",
                   labels={"pred_loss_ha": "Luas Kehilangan (ha)", "year": "Tahun"})
    st.plotly_chart(fig4, use_container_width=True)

# -------------------------------
# 🗺 TAB 4: Peta Persebaran
# -------------------------------
with tabs[3]:
    st.subheader("Persebaran Deforestasi dan Emisi Karbon Tahun 2027")

    shp_path = "gadm41_IDN_1.shp"
    gdf = gpd.read_file(shp_path)
    gdf["NAME_1"] = gdf["NAME_1"].str.title().str.strip()
    forecast_df["subnational1"] = forecast_df["subnational1"].str.title().str.strip()
    gdf_merged = gdf.merge(forecast_df[forecast_df["year"] == 2027], left_on="NAME_1", right_on="subnational1", how="left")

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    gdf_merged.plot(column="pred_loss_ha", cmap="YlGn", legend=True, ax=ax[0])
    ax[0].set_title("Persebaran Deforestasi Tahun 2027")
    gdf_merged.plot(column="pred_loss_ha" * 3.67, cmap="OrRd", legend=True, ax=ax[1])
    ax[1].set_title("Persebaran Emisi Karbon Tahun 2027")
    for a in ax: a.axis("off")
    st.pyplot(fig)

st.markdown("---")
st.caption("📊 Data: Global Forest Watch | Visualisasi oleh Miftah Faridl © 2025")
