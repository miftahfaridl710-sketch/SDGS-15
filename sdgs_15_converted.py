# sdgs_15_dashboard.py
# Streamlit dashboard: Tren Deforestasi & Emisi + Prediksi 2025-2027 + Peta 2027
# - membaca IDN.xlsx dari repo GitHub (raw)
# - mengunduh shapefile GADM dari UC Davis (urllib + zipfile)
# - membuat plot interaktif (plotly) + tombol unduh

import os
import io
import zipfile
import urllib.request
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="SDGs 15 — Deforestasi & Emisi (Dashboard)", layout="wide")
st.title("🌳 SDGs 15 — Dashboard Deforestasi & Emisi Karbon (Indonesia)")

DATA_DIR = Path("data")
SHAPE_DIR = DATA_DIR / "shapefile"
DATA_DIR.mkdir(exist_ok=True)
SHAPE_DIR.mkdir(parents=True, exist_ok=True)

# URL dataset (raw GitHub) — pastikan file IDN.xlsx ada di branch main
EXCEL_RAW_URL = "https://raw.githubusercontent.com/miftahfaridl710-sketch/SDGS-15/main/IDN.xlsx"

# langsung baca shapefile dari root repo
import geopandas as gpd
shp_path = "gadm41_IDN_1.shp"
gdf = gpd.read_file(shp_path)

# -------------------------
# HELPERS
# -------------------------
@st.cache_data(show_spinner=False)
def download_excel(url: str) -> pd.DataFrame:
    # baca excel langsung dari raw github url
    return pd.read_excel(url, sheet_name="Subnational 1 tree cover loss", engine="openpyxl")

@st.cache_data(show_spinner=False)
def download_and_extract_shapefile(zip_url: str, zip_target: Path, extract_to: Path) -> Path:
    # kalau sudah diekstrak dan ada shapefile, kembalikan path-nya
    # unduh zip jika belum ada
    if not zip_target.exists():
        with st.spinner("Mengunduh shapefile Indonesia (besar ~ beberapa MB)..."):
            urllib.request.urlretrieve(zip_url, zip_target)
    # ekstrak
    with zipfile.ZipFile(zip_target, "r") as z:
        z.extractall(extract_to)
    # cari file .shp
    for p in extract_to.rglob("*.shp"):
        return p
    raise FileNotFoundError("Shapefile .shp tidak ditemukan dalam zip GADM.")

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def fig_to_png_bytes(fig):
    # memerlukan kaleido di requirements agar berfungsi di Streamlit Cloud
    try:
        img_bytes = fig.to_image(format="png", scale=2)
        return img_bytes
    except Exception:
        # fallback: ekspor HTML
        return None

# -------------------------
# LOAD DATA
# -------------------------
st.sidebar.header("Kontrol")
if st.sidebar.button("(Re)load data & shapefile"):
    st.cache_data.clear()

with st.spinner("Membaca dataset..."):
    try:
        df_raw = download_excel(EXCEL_RAW_URL)
    except Exception as e:
        st.error("Gagal membaca dataset dari GitHub. Pastikan file IDN.xlsx ada dan public di repo.")
        st.stop()

st.sidebar.success("Dataset dimuat")

# Tampilkan ringkasan data
st.subheader("Ringkasan Dataset")
st.write("Data sheet: `Subnational 1 tree cover loss`")
st.write(f"Shape: {df_raw.shape}")
st.dataframe(df_raw.head(6))

# -------------------------
# PREPROCESSING
# -------------------------
# Filter threshold 30 and convert wide->long
df = df_raw.copy()
df = df[df["threshold"] == 30].reset_index(drop=True)

loss_cols = [c for c in df.columns if c.startswith("tc_loss_ha_")]
df_long = df.melt(
    id_vars=["country", "subnational1", "threshold", "extent_2000_ha"],
    value_vars=loss_cols,
    var_name="year",
    value_name="loss_ha"
)
df_long["year"] = df_long["year"].str.extract(r"(\d{4})").astype(int)
df_long = df_long.sort_values(["subnational1", "year"]).reset_index(drop=True)
df_long["loss_rate_%"] = (df_long["loss_ha"] / df_long["extent_2000_ha"]) * 100

# national trend (historical)
national_trend = df_long.groupby("year", as_index=False)["loss_ha"].sum()

# Aggregate carbon if carbon sheet exists in repo — attempt reading if present
carbon_ok = False
try:
    df_carbon = pd.read_excel(EXCEL_RAW_URL.replace("IDN.xlsx","IDN.xlsx"), sheet_name="Subnational 1 carbon data", engine="openpyxl")
    # process carbon similar to notebook if available
    carbon_ok = True
except Exception:
    # mungkin sheet carbon tidak ada — kita fallback dengan estimasi berbasis kolom yang ada jika memungkinkan
    carbon_ok = False

# Build merged dataset if carbon sheet available
if carbon_ok:
    # make long for carbon
    emission_cols = [c for c in df_carbon.columns if c.startswith("gross_emissions_")]
    carbon_cols = [
        "country", "subnational1", "umd_tree_cover_density_2000__threshold",
        "gfw_aboveground_carbon_stocks_2000__Mg_C",
        "avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1"
    ]
    carbon_long = df_carbon.melt(id_vars=carbon_cols, value_vars=emission_cols, var_name="year", value_name="emission_Mg_CO2e")
    carbon_long["year"] = carbon_long["year"].str.extract(r"(\d{4})").astype(int)
    merged = pd.merge(df_long, carbon_long, on=["subnational1","year"], how="left")
    merged["emission_Mg_CO2e"] = pd.to_numeric(merged["emission_Mg_CO2e"], errors="coerce")
else:
    # Estimate emissions from avg_gfw_aboveground_carbon_stocks if available in df (best-effort)
    merged = df_long.copy()
    if "avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1" in df.columns:
        merged = merged.merge(df[["subnational1","avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1","extent_2000_ha"]].drop_duplicates("subnational1"),
                              on="subnational1", how="left")
        merged["emission_estimated_CO2e"] = merged["loss_ha"] * merged["avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1"] * 3.67
    else:
        merged["emission_estimated_CO2e"] = np.nan

# -------------------------
# PLOT 1: Tren Kehilangan Tutupan Hutan Nasional
# -------------------------
st.markdown("---")
st.header("Tren Kehilangan Tutupan Hutan di Indonesia")
desc1 = ("Plot ini menunjukkan total luas kehilangan tutupan hutan tiap tahun "
         "(jumlah seluruh provinsi). Angka yang tampil adalah jumlah hektar kehilangan (loss_ha).")
st.write(desc1)

fig_trend = px.line(national_trend, x="year", y="loss_ha",
                    title="Tren Kehilangan Tutupan Hutan Indonesia (2001–2024)",
                    labels={"loss_ha": "Luas Kehilangan (ha)", "year": "Tahun"},
                    markers=True)
st.plotly_chart(fig_trend, use_container_width=True)

# download button for national_trend
csv_bytes = df_to_csv = df_long.groupby("year", as_index=False)["loss_ha"].sum().to_csv(index=False).encode()
st.download_button("Unduh data tren nasional (CSV)", data=csv_bytes, file_name="national_trend_loss_ha.csv", mime="text/csv")

# -------------------------
# PLOT 2: 5 Provinsi Teratas
# -------------------------
st.markdown("---")
st.header("Tren Kehilangan Hutan — 5 Provinsi Teratas")
desc2 = ("Menampilkan 5 provinsi dengan rata-rata kehilangan (loss_ha) terbesar sepanjang periode. "
         "Gunanya untuk melihat provinsi mana yang memberi kontribusi terbesar terhadap kehilangan hutan.")
st.write(desc2)

top5 = df_long.groupby("subnational1")["loss_ha"].mean().sort_values(ascending=False).head(5).index.tolist()
df_top5 = df_long[df_long["subnational1"].isin(top5)]

fig_top5 = px.line(df_top5, x="year", y="loss_ha", color="subnational1", markers=True,
                   title="Tren Kehilangan Hutan di 5 Provinsi Teratas (2001–2024)",
                   labels={"loss_ha":"Luas Kehilangan (ha)", "subnational1":"Provinsi"})
st.plotly_chart(fig_top5, use_container_width=True)
st.download_button("Unduh data top5 (CSV)", data=df_top5.to_csv(index=False).encode(), file_name="top5_provinsi_loss.csv", mime="text/csv")

# -------------------------
# PLOT 3: Tren Emisi Karbon akibat Deforestasi
# -------------------------
st.markdown("---")
st.header("Tren Emisi Karbon Akibat Deforestasi")
desc3 = ("Jika terdapat data emisi (sheet carbon), plot ini menampilkan total emisi per tahun (Mg CO2e). "
         "Jika sheet tidak tersedia, aplikasi menggunakan estimasi berdasarkan stok karbon rata-rata per ha.")
st.write(desc3)

st.write("Kolom yang tersedia:", merged.columns.tolist())
# PLOT 3: Tren Emisi Karbon akibat Deforestasi
# -------------------------
st.markdown("---")
st.header("Tren Emisi Karbon Akibat Deforestasi")
desc3 = ("Jika terdapat data emisi (sheet carbon), plot ini menampilkan total emisi per tahun (Mg CO2e). "
         "Jika sheet tidak tersedia, aplikasi menggunakan estimasi berdasarkan stok karbon rata-rata per ha.")
st.write(desc3)

# Pilih kolom yang tersedia untuk emisi
emission_col = None
for col in ["emission_Mg_CO2e", "emission_estimated_CO2e"]:
    if col in merged.columns:
        emission_col = col
        break

if emission_col is None or merged[emission_col].isna().all():
    st.error("❌ Tidak ditemukan kolom emisi (`emission_Mg_CO2e` atau `emission_estimated_CO2e`). "
             "Pastikan sheet carbon tersedia atau kolom stok karbon per ha ada di dataset.")
    st.stop()
else:
    carbon_trend = merged.groupby("year", as_index=False)[emission_col].sum()
    carbon_trend = carbon_trend.rename(columns={emission_col: "emission_Mg_CO2e"})

    fig_emis = px.line(
        carbon_trend, x="year", y="emission_Mg_CO2e",
        title="Tren Emisi Karbon akibat Deforestasi (2001–2024)",
        labels={"emission_Mg_CO2e": "Emisi (Mg CO2e)", "year": "Tahun"},
        markers=True
    )
    st.plotly_chart(fig_emis, use_container_width=True)
    st.download_button(
        "Unduh data emisi (CSV)",
        data=carbon_trend.to_csv(index=False).encode(),
        file_name="carbon_trend.csv",
        mime="text/csv"
    )

fig_emis = px.line(carbon_trend, x="year", y="emission_Mg_CO2e", title="Tren Emisi Karbon akibat Deforestasi (2001–2024)",
                   labels={"emission_Mg_CO2e":"Emisi (Mg CO2e)", "year":"Tahun"}, markers=True)
st.plotly_chart(fig_emis, use_container_width=True)
st.download_button("Unduh data emisi (CSV)", data=carbon_trend.to_csv(index=False).encode(), file_name="carbon_trend.csv", mime="text/csv")

# -------------------------
# PLOT 4: Tren Deforestasi & Emisi (dual axis style)
# -------------------------
st.markdown("---")
st.header("Tren Kehilangan Tutupan Hutan & Emisi Karbon (Dua Metode)")
desc4 = ("Plot gabungan: sumbu kiri = kehilangan hutan (ha), sumbu kanan = emisi (Mg CO2e). "
         "Memudahkan melihat korelasi temporal antara kehilangan hutan dan emisi.")
st.write(desc4)

# gabungkan untuk plotting
trend_combo = pd.merge(national_trend, carbon_trend, on="year", how="left")
fig_combo = go.Figure()
fig_combo.add_trace(go.Bar(x=trend_combo["year"], y=trend_combo["loss_ha"], name="Kehilangan Hutan (ha)", yaxis="y1", marker_color="green", opacity=0.6))
fig_combo.add_trace(go.Line(x=trend_combo["year"], y=trend_combo["emission_Mg_CO2e"], name="Emisi (Mg CO2e)", yaxis="y2", marker_color="red"))
fig_combo.update_layout(
    title="Tren Kehilangan Tutupan Hutan dan Emisi Karbon (2001–2024)",
    yaxis=dict(title="Kehilangan Hutan (ha)"),
    yaxis2=dict(title="Emisi (Mg CO2e)", overlaying="y", side="right")
)
st.plotly_chart(fig_combo, use_container_width=True)

# -------------------------
# MODELING & EVALUATION (ringkas)
# -------------------------
st.markdown("---")
st.header("Model: XGBoost — Prediksi & Evaluasi (ringkas)")
st.write("Model dibuat sebagai pendekatan time-series per provinsi menggunakan fitur lag 1-3 dan year serta encoding provinsi.")

# Prepare modeling table
df_model = merged.copy()
df_model["lag_1"] = df_model.groupby("subnational1")["loss_ha"].shift(1)
df_model["lag_2"] = df_model.groupby("subnational1")["loss_ha"].shift(2)
df_model["lag_3"] = df_model.groupby("subnational1")["loss_ha"].shift(3)
df_model = df_model.dropna(subset=["lag_1","lag_2","lag_3"]).reset_index(drop=True)

le = LabelEncoder()
df_model["prov_enc"] = le.fit_transform(df_model["subnational1"].astype(str))
FEATURES = [c for c in ["lag_1","lag_2","lag_3","year","prov_enc"] if c in df_model.columns]
TARGET = "loss_ha"

train_df = df_model[df_model["year"] <= 2021]
val_df = df_model[(df_model["year"] >= 2022) & (df_model["year"] <= 2023)]
test_df = df_model[df_model["year"] == 2024]

if len(test_df) == 0:
    st.warning("Tidak ada data test untuk tahun 2024 — evaluasi akan menggunakan split validasi sederhana.")
    test_df = val_df.copy()

# Train a compact XGBoost (faster)
model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(train_df[FEATURES], train_df[TARGET])

X_test = test_df[FEATURES]
y_test = test_df[TARGET]
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)
nrmse = rmse / (y_test.max() - y_test.min()) if (y_test.max() - y_test.min()) != 0 else np.nan

st.subheader("Hasil Evaluasi")
st.metric("RMSE", f"{rmse:.2f}")
st.metric("R²", f"{r2:.3f}")
st.metric("NRMSE", f"{nrmse:.3f}" if not np.isnan(nrmse) else "N/A")

# Interpretasi singkat
st.write("""
**Interpretasi:**  
- **RMSE** (Root Mean Squared Error) mengukur seberapa jauh prediksi rata-rata meleset dari nilai aktual (satuan: ha).
- **R²** mendekati 1 menunjukkan model menjelaskan sebagian besar variabilitas; nilai dekat 0 menandakan model kurang menjelaskan.
- **NRMSE** memberikan konteks relatif RMSE terhadap rentang data: semakin kecil semakin baik.
""")

# -------------------------
# FORECAST 2025-2027
# -------------------------
st.markdown("---")
st.header("Prediksi Deforestasi & Emisi (2025–2027)")
st.write("Membuat prediksi beruntun per provinsi menggunakan model terlatih (metode iteratif sederhana).")

forecast_years = [2025, 2026, 2027]
forecast_rows = []

# prepare per-prov series
prov_groups = df_model.groupby("subnational1")
for prov, series in prov_groups:
    s = series.sort_values("year").copy()
    # last known values
    for year in forecast_years:
        lag_1 = s["loss_ha"].iloc[-1]
        lag_2 = s["loss_ha"].iloc[-2] if len(s) >= 2 else lag_1
        lag_3 = s["loss_ha"].iloc[-3] if len(s) >= 3 else lag_1
        prov_enc = le.transform([prov])[0]
        Xf = pd.DataFrame([[lag_1, lag_2, lag_3, year, prov_enc]], columns=FEATURES)
        pred = model.predict(Xf)[0]
        # emission estimate if avg_gfw column exists
        avg_gfw = s["avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1"].iloc[-1] if "avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1" in s.columns else np.nan
        emission_est = pred * avg_gfw * 3.67 if not pd.isna(avg_gfw) else np.nan
        forecast_rows.append({"subnational1": prov, "year": year, "pred_loss_ha": pred, "emission_estimated_CO2e": emission_est})
        # append predicted into series for next iteration
        new_row = {"subnational1": prov, "year": year, "loss_ha": pred, "avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1": avg_gfw}
        s = pd.concat([s, pd.DataFrame([new_row])], ignore_index=True)

forecast_df = pd.DataFrame(forecast_rows)
st.dataframe(forecast_df.head(10))
st.download_button("Unduh hasil prediksi (CSV)", data=forecast_df.to_csv(index=False).encode(), file_name="forecast_2025_2027.csv", mime="text/csv")

# national aggregated forecast
national_hist = df_model.groupby("year", as_index=False).agg({"loss_ha":"sum"})
national_forecast = forecast_df.groupby("year", as_index=False).agg({"pred_loss_ha":"sum","emission_estimated_CO2e":"sum"})
trend_all = pd.concat([
    national_hist.rename(columns={"loss_ha":"pred_loss_ha"}), 
    national_forecast
], ignore_index=True).sort_values("year").reset_index(drop=True)

# Fill NaN emissions with zeros if needed for plotting
trend_all["emission_estimated_CO2e"] = trend_all["emission_estimated_CO2e"].fillna(0)

# -------------------------
# PLOT: Tren Deforestasi dan Prediksi 2001-2027
# -------------------------
st.markdown("---")
st.header("Tren Kehilangan Tutupan Hutan di Indonesia (2001–2027)")
st.write("Gabungan data historis dan prediksi hingga 2027.")

fig_2001_2027 = px.line(trend_all, x="year", y="pred_loss_ha", markers=True,
                        title="Tren Kehilangan Tutupan Hutan di Indonesia (2001–2027)",
                        labels={"pred_loss_ha":"Luas Kehilangan (ha)"})
st.plotly_chart(fig_2001_2027, use_container_width=True)
st.download_button("Unduh tren 2001-2027 (CSV)", data=trend_all.to_csv(index=False).encode(), file_name="trend_2001_2027.csv", mime="text/csv")

# -------------------------
# PLOT: Prediksi Emisi Karbon (2001-2027)
# -------------------------
st.markdown("---")
st.header("Prediksi Emisi Karbon akibat Deforestasi (2001–2027)")
st.write("Menggunakan estimasi emisi bila data emisi asli tidak tersedia.")

fig_emis_2001_2027 = px.line(trend_all, x="year", y="emission_estimated_CO2e", markers=True,
                             title="Prediksi Emisi Karbon akibat Deforestasi (2001–2027)",
                             labels={"emission_estimated_CO2e":"Emisi (Mg CO2e)"})
st.plotly_chart(fig_emis_2001_2027, use_container_width=True)
st.download_button("Unduh prediksi emisi 2001-2027 (CSV)", data=trend_all.to_csv(index=False).encode(), file_name="emission_trend_2001_2027.csv", mime="text/csv")

# -------------------------
# Peta: Persebaran 2027
# -------------------------
st.markdown("---")
st.header("Persebaran Deforestasi & Emisi Karbon Tahun 2027")
st.write("Peta memperlihatkan prediksi per provinsi untuk tahun 2027 (jika shapefile berhasil diunduh dan merge nama provinsi cocok).")

shp_path = None
try:
    shp_path = download_and_extract_shapefile(GADM_ZIP_URL, GADM_ZIP_LOCAL, SHAPE_DIR)
    gdf = gpd.read_file(shp_path)
    # Normalisasi nama provinsi untuk merge
    gdf["NAME_1"] = gdf["NAME_1"].str.title().str.strip()
    forecast_df["subnational1"] = forecast_df["subnational1"].str.title().str.strip()
    map_2027 = forecast_df[forecast_df["year"] == 2027].copy()
    merged_map = gdf.merge(map_2027, left_on="NAME_1", right_on="subnational1", how="left")
    # Plot deforestation map
    fig_map_def = merged_map.plot(column="pred_loss_ha", cmap="YlGn", legend=True, scheme=None)
    # but better using plotly choropleth if geometries convertible
    merged_map = merged_map.to_crs(epsg=4326)
    merged_map["center"] = merged_map["geometry"].centroid
    # convert to geojson for plotly
    merged_map_json = merged_map.__geo_interface__
    # Use geopandas -> plotly express choropleth_mapbox (requires mapbox token for tile backgrounds) 
    # We'll fallback to simple static plot using matplotlib saved to PNG and display via st.image
    # Save matplotlib plots to PNG to show
    import matplotlib.pyplot as plt
    fig1, ax1 = plt.subplots(1,1, figsize=(10,8))
    merged_map.plot(column="pred_loss_ha", cmap="YlGn", linewidth=0.3, edgecolor="gray", legend=True, ax=ax1)
    ax1.set_title("Persebaran Prediksi Deforestasi Indonesia Tahun 2027")
    ax1.axis("off")
    png_path_def = DATA_DIR / "peta_deforestasi_2027.png"
    fig1.savefig(png_path_def, dpi=200, bbox_inches="tight")
    st.image(str(png_path_def), caption="Persebaran Prediksi Deforestasi 2027", use_column_width=True)
    st.download_button("Unduh peta deforestasi (PNG)", data=open(png_path_def,"rb").read(), file_name="peta_deforestasi_2027.png", mime="image/png")
    # Emission map
    fig2, ax2 = plt.subplots(1,1, figsize=(10,8))
    merged_map.plot(column="emission_estimated_CO2e", cmap="OrRd", linewidth=0.3, edgecolor="gray", legend=True, ax=ax2)
    ax2.set_title("Persebaran Emisi Karbon akibat Deforestasi Tahun 2027")
    ax2.axis("off")
    png_path_emis = DATA_DIR / "peta_emisi_2027.png"
    fig2.savefig(png_path_emis, dpi=200, bbox_inches="tight")
    st.image(str(png_path_emis), caption="Persebaran Emisi Karbon 2027", use_column_width=True)
    st.download_button("Unduh peta emisi (PNG)", data=open(png_path_emis,"rb").read(), file_name="peta_emisi_2027.png", mime="image/png")
except Exception as e:
    st.warning(f"Gagal menyiapkan peta secara otomatis: {e}. Pastikan shapefile dapat diunduh atau upload shapefile ke folder data/shapefile.")
    st.info("Jika peta tidak tampil, kamu bisa upload shapefile (gadm41_IDN_1.shp + .dbf + .shx + dll) ke folder data/shapefile di repo.")

st.markdown("---")
st.write("Selesai — dashboard menampilkan plot tren, prediksi, peta (jika shapefile berhasil diunduh), dan tombol unduh untuk data serta gambar peta.")
st.write("Jika ingin penyesuaian (mis. styling, filter provinsi, atau export SVG/PNG untuk semua plot), beri tahu aku dan aku akan perbaiki kode ini.")
