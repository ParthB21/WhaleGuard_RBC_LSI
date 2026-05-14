import calendar
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WhaleGuard — NARW Habitat",
    page_icon="🐋",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "SST", "Chlorophyll", "Salinity", "Bathymetry",
    "SST_Gradient", "Is_Thermal_Front", "Month",
    "Bathy_Slope", "Dist_to_Shore_km", "Dist_to_Shelf_km",
]

MODELS_META = {
    "Random Forest": {
        "path": "models/rf_narw_sdm.joblib",
        "threshold": 0.1543,
        "type": "joblib",
        "auc": 0.9064,
        "recall": "80.0 %",
        "precision": "48.2 %",
        "f1": 0.602,
        "importance_img": "images/rf_feature_importance.png",
        "description": (
            "627-tree bagged ensemble. Highest AUC (0.9064) and fewest false alarms "
            "at the conservation threshold. Recommended for deployment."
        ),
    },
    "XGBoost": {
        "path": "models/xgb_narw_sdm.json",
        "threshold": 0.1757,
        "type": "xgb",
        "auc": 0.8986,
        "recall": "80.0 %",
        "precision": "45.1 %",
        "f1": 0.577,
        "importance_img": "images/feature_importance.png",
        "description": (
            "516-tree boosted ensemble. Handles missing values natively. "
            "Strong non-linear interaction modelling, especially for seasonality."
        ),
    },
    "Logistic Regression": {
        "path": "models/lr_narw_sdm.joblib",
        "threshold": 0.50,
        "type": "joblib",
        "auc": 0.8048,
        "recall": "83.2 %",
        "precision": "33.9 %",
        "f1": 0.481,
        "importance_img": "images/lr_coefficients.png",
        "description": (
            "Interpretable baseline. Coefficients map directly to ecological odds ratios. "
            "Cannot capture non-linear feature interactions."
        ),
    },
}

MONTH_ABBRS = {i: calendar.month_abbr[i] for i in range(1, 13)}

DATA_PATH = Path("data/processed/ML_Whale_Dataset_Final.csv")

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Is_Thermal_Front"] = df["Is_Thermal_Front"].astype(int)
    return df


@st.cache_resource(show_spinner="Loading model…")
def load_model(name: str):
    meta = MODELS_META[name]
    if meta["type"] == "xgb":
        m = xgb.XGBClassifier()
        m.load_model(meta["path"])
    else:
        m = joblib.load(meta["path"])
    return m


@st.cache_data(show_spinner="Computing predictions…")
def predict_all(model_name: str) -> np.ndarray:
    df = load_data()
    model = load_model(model_name)
    X = df[FEATURE_COLS].copy()
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("WhaleGuard")
    st.caption("NARW Habitat Prediction Dashboard")
    st.divider()

    model_name = st.selectbox(
        "Model",
        list(MODELS_META.keys()),
        help="Select which trained model to use for predictions.",
    )

    st.subheader("Date Filter")
    year = st.slider("Year", min_value=2002, max_value=2018, value=2012)
    month = st.select_slider(
        "Month",
        options=list(range(1, 13)),
        value=6,
        format_func=lambda x: MONTH_ABBRS[x],
    )

    st.subheader("Display Options")
    show_sightings = st.toggle("Overlay actual sightings", value=False)
    heatmap_radius = st.slider(
        "Heatmap radius (KDE smoothing)",
        min_value=10, max_value=60, value=30,
        help="Larger values = smoother heatmap; smaller = sharper point clusters.",
    )

    meta = MODELS_META[model_name]
    st.divider()
    st.subheader("Model Performance")
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", meta["auc"])
    c2.metric("F1-Score", meta["f1"])
    c1.metric("Recall @ τ", meta["recall"])
    c2.metric("Precision @ τ", meta["precision"])
    st.caption(f"Conservation threshold τ = **{meta['threshold']:.4f}**")


# ---------------------------------------------------------------------------
# Load data + predictions
# ---------------------------------------------------------------------------
df_full = load_data()
proba = predict_all(model_name)

df_full = df_full.copy()
df_full["probability"] = proba
threshold = meta["threshold"]
df_full["is_habitat"] = df_full["probability"] >= threshold

# ---------------------------------------------------------------------------
# Filter by year + month
# ---------------------------------------------------------------------------
df_view = df_full[(df_full["Year"] == year) & (df_full["Month"] == month)]
sparse_fallback = len(df_view) < 20

if sparse_fallback:
    st.warning(
        f"Only **{len(df_view)}** survey points found for "
        f"{MONTH_ABBRS[month]} {year}. "
        f"Showing all years for **{MONTH_ABBRS[month]}** instead."
    )
    df_view = df_full[df_full["Month"] == month]

# ---------------------------------------------------------------------------
# Header + KPI cards
# ---------------------------------------------------------------------------
st.title("WhaleGuard — NARW Habitat Prediction")
period_label = (
    f"All years · {MONTH_ABBRS[month]}" if sparse_fallback
    else f"{MONTH_ABBRS[month]} {year}"
)
st.caption(f"Showing **{period_label}** · Model: **{model_name}**")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Survey points", f"{len(df_view):,}")
k2.metric("Predicted habitat", f"{df_view['is_habitat'].mean():.1%}")
k3.metric("Avg. whale probability", f"{df_view['probability'].mean():.3f}")
k4.metric("Max probability", f"{df_view['probability'].max():.3f}")

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
fig = go.Figure()

# Density heatmap layer
fig.add_trace(
    go.Densitymapbox(
        lat=df_view["Lat"],
        lon=df_view["Lon"],
        z=df_view["probability"],
        radius=heatmap_radius,
        colorscale="YlOrRd",
        zmin=0,
        zmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text="P(Whale)", side="right"),
            thickness=14,
            len=0.75,
            tickvals=[0, threshold, 0.5, 1.0],
            ticktext=["0", f"τ={threshold:.2f}", "0.5", "1.0"],
        ),
        name="Whale probability",
        hovertemplate="Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>P(whale): %{z:.3f}<extra></extra>",
    )
)

# Optional: scatter layer for actual confirmed sightings
if show_sightings:
    sightings = df_view[df_view["Presence"] == 1]
    fig.add_trace(
        go.Scattermapbox(
            lat=sightings["Lat"],
            lon=sightings["Lon"],
            mode="markers",
            marker=dict(size=5, color="#1a6fb5", opacity=0.75),
            name="Confirmed sightings",
            hovertemplate="Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra>Confirmed NARW sighting</extra>",
        )
    )

fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        center=dict(lat=38, lon=-72),
        zoom=4,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.02,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255,255,255,0.8)",
    ),
    height=620,
    margin=dict(l=0, r=0, t=30, b=0),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Expandable panels
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    with st.expander("Feature Importance"):
        img_path = Path(meta["importance_img"])
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.info("Image not found — run training scripts to generate plots.")

with col_right:
    with st.expander("About this model"):
        st.markdown(meta["description"])
        st.markdown(
            f"""
| Metric | Value |
|---|---|
| ROC-AUC | {meta['auc']} |
| Recall @ τ | {meta['recall']} |
| Precision @ τ | {meta['precision']} |
| F1-Score | {meta['f1']} |
| Threshold (τ) | {meta['threshold']} |
"""
        )
        st.caption(
            "Threshold optimised for ≥ 80 % recall (endangered species precautionary principle). "
            "False negatives (missed whales) carry far higher cost than false positives (unneeded speed restrictions)."
        )
