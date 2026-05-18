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

DATA_PATH = Path("data/processed/Gulf_St_Lawrence_Grid_Features.csv")

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    # Grid CSV already has Year and Month columns; no Date column to parse.
    df = pd.read_csv(DATA_PATH)
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
    year = st.slider("Year", min_value=2002, max_value=2026, value=2024)
    month = st.select_slider(
        "Month",
        options=list(range(1, 13)),
        value=6,
        format_func=lambda x: MONTH_ABBRS[x],
    )

    st.subheader("Display Options")
    heatmap_radius = st.slider(
        "Heatmap radius (KDE smoothing)",
        min_value=10, max_value=60, value=20,
        help="Larger values = smoother heatmap; smaller = sharper grid cells.",
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
# Filter by year + month (grid CSV covers all 192 months — no fallback needed)
# ---------------------------------------------------------------------------
df_view = df_full[(df_full["Year"] == year) & (df_full["Month"] == month)]

# ---------------------------------------------------------------------------
# Header + KPI cards
# ---------------------------------------------------------------------------
st.title("WhaleGuard — NARW Habitat Prediction")
st.caption(f"Showing **{MONTH_ABBRS[month]} {year}** · Model: **{model_name}**")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Grid cells", f"{len(df_view):,}",
          help="Number of 0.25° ocean grid cells covering the Gulf of St. Lawrence for the selected month and year.")
k2.metric("Predicted habitat", f"{df_view['is_habitat'].mean():.1%}",
          help=(
              f"Fraction of grid cells where the model's whale-presence probability "
              f"meets or exceeds the conservation threshold τ = {threshold:.4f}. "
              f"A cell above τ is classified as 'predicted habitat' — meaning the model "
              f"considers it likely enough to contain a NARW that a speed restriction would "
              f"be warranted. The threshold is set to achieve ≥ 80 % recall, so the model "
              f"deliberately flags more cells than strictly necessary to avoid missing real whales."
          ))
k3.metric("Avg. whale probability", f"{df_view['probability'].mean():.3f}",
          help="Mean predicted probability of NARW presence across all grid cells this month. Ranges 0–1.")
k4.metric("Max probability", f"{df_view['probability'].max():.3f}",
          help="Highest single-cell predicted probability this month — the model's most confident prediction of whale presence.")

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
        radius=heatmap_radius,   # 20 fills gaps in the 0.25° grid at zoom 5
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

fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        center=dict(lat=48, lon=-63),   # Gulf of St. Lawrence
        zoom=5,
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
        st.caption(
            "**Gain** measures the average reduction in prediction error each time a feature "
            "is used to split a tree node. A high-gain feature (e.g. Dist_to_Shore_km) "
            "consistently produces accurate, well-separated branches; a low-gain feature "
            "adds little information when split on. Unlike frequency-based importance, "
            "gain rewards quality of splits, not just how often a feature is used."
        )

with col_right:
    with st.expander("About this model"):
        st.markdown(meta["description"])
        st.divider()
        st.markdown(
            f"""
| Metric | Value | What it means |
|---|---|---|
| **ROC-AUC** | {meta['auc']} | How well the model ranks whale locations above background — 1.0 is perfect, 0.5 is random. |
| **Recall @ τ** | {meta['recall']} | Of all real whale locations, the fraction correctly flagged. Primary metric — a missed whale risks a ship strike. |
| **Precision @ τ** | {meta['precision']} | Of all flagged locations, the fraction that actually had a whale. Lower precision = more false alarms, an acceptable trade-off. |
| **F1-Score** | {meta['f1']} | Harmonic mean of Recall and Precision. |
| **Threshold (τ)** | {meta['threshold']} | Probability cut-off for "habitat" decisions, tuned below 0.5 to guarantee ≥ 80 % recall. |
"""
        )
        st.caption(
            "Evaluated on a temporal hold-out (2015–2018) — data the model never saw during training."
        )
