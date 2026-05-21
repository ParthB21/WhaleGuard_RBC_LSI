import calendar
import heapq
import math
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
# Ports & Gateways — all coordinates snap to nearest ocean grid cell at runtime
# ---------------------------------------------------------------------------
PORTS = {
    # Quebec North Shore
    "Sept-Îles":        {"lat": 50.20, "lon": -66.38, "group": "Quebec North Shore"},
    "Port-Cartier":     {"lat": 50.03, "lon": -66.82, "group": "Quebec North Shore"},
    "Baie-Comeau":      {"lat": 49.22, "lon": -68.15, "group": "Quebec North Shore"},
    "Matane":           {"lat": 48.85, "lon": -67.57, "group": "Quebec North Shore"},
    # Quebec South Shore & Gaspé
    "Rimouski":         {"lat": 48.48, "lon": -68.52, "group": "Quebec South Shore"},
    "Gaspé":            {"lat": 48.83, "lon": -64.44, "group": "Quebec South Shore"},
    "Chandler":         {"lat": 48.35, "lon": -64.68, "group": "Quebec South Shore"},
    # New Brunswick
    "Belledune":        {"lat": 47.91, "lon": -65.83, "group": "New Brunswick"},
    "Dalhousie":        {"lat": 48.07, "lon": -66.37, "group": "New Brunswick"},
    # Prince Edward Island
    "Charlottetown":    {"lat": 46.23, "lon": -63.13, "group": "Prince Edward Island"},
    "Summerside":       {"lat": 46.39, "lon": -63.79, "group": "Prince Edward Island"},
    # Nova Scotia
    "Sydney":           {"lat": 46.14, "lon": -60.19, "group": "Nova Scotia"},
    "Port Hawkesbury":  {"lat": 45.62, "lon": -61.35, "group": "Nova Scotia"},
    # Newfoundland
    "Corner Brook":     {"lat": 48.95, "lon": -57.95, "group": "Newfoundland"},
    "Port aux Basques": {"lat": 47.57, "lon": -59.14, "group": "Newfoundland"},
}

GATEWAYS = {
    "→ Montreal / Upstream":     {"lat": 47.25, "lon": -70.50, "group": "Exit Points"},
    "→ Atlantic (Cabot Strait)": {"lat": 47.00, "lon": -59.75, "group": "Exit Points"},
    "→ Atlantic (Belle Isle)":   {"lat": 51.00, "lon": -57.25, "group": "Exit Points"},
}

ALL_LOCATIONS = {**PORTS, **GATEWAYS}

# ---------------------------------------------------------------------------
# Routing utilities
# ---------------------------------------------------------------------------
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def snap_to_grid(lat: float, lon: float, ocean_cells: set) -> tuple:
    """Find the nearest ocean grid cell to a given coordinate."""
    best, best_d = None, float("inf")
    for cell in ocean_cells:
        d = (cell[0] - lat) ** 2 + (cell[1] - lon) ** 2
        if d < best_d:
            best_d = d
            best = cell
    return best


def build_ocean_graph(ocean_cells: set) -> dict:
    """Build adjacency dict — each ocean cell connects to its 8 ocean neighbours.

    Also checks extended (0.5°) diagonal jumps to bridge the St. Lawrence River
    narrows where the 0.25° grid has a gap (e.g. between 47.25°N and 47.75°N).
    """
    offsets_primary = [
        (-0.25, 0), (0.25, 0), (0, -0.25), (0, 0.25),          # N S W E
        (-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25),  # diagonals
    ]
    offsets_bridge = [
        (-0.50, 0.25), (-0.50, 0.50), (-0.25, 0.50),           # far NE quadrant
        (0.50, 0.25), (0.50, 0.50), (0.25, 0.50),               # far SE quadrant
        (-0.50, -0.25), (-0.50, -0.50), (-0.25, -0.50),         # far NW quadrant
        (0.50, -0.25), (0.50, -0.50), (0.25, -0.50),            # far SW quadrant
    ]
    graph = {}
    for cell in ocean_cells:
        neighbours = []
        for dlat, dlon in offsets_primary:
            nb = (round(cell[0] + dlat, 2), round(cell[1] + dlon, 2))
            if nb in ocean_cells:
                neighbours.append(nb)
        # Only use bridge offsets if the cell has few primary neighbours
        # (i.e. it's at a channel bottleneck)
        if len(neighbours) <= 2:
            for dlat, dlon in offsets_bridge:
                nb = (round(cell[0] + dlat, 2), round(cell[1] + dlon, 2))
                if nb in ocean_cells and nb not in neighbours:
                    neighbours.append(nb)
        graph[cell] = neighbours
    return graph


@st.cache_data(show_spinner="Computing route…")
def find_route(
    ocean_cells_tuple: tuple,
    prob_dict_keys: tuple,
    prob_dict_vals: tuple,
    start: tuple,
    end: tuple,
    whale_weight: float = 0.0,
) -> list:
    """
    A* pathfinding over the ocean grid.

    whale_weight = 0   → shortest path (standard route)
    whale_weight = 500 → heavily penalises whale-probable cells (eco-route)

    Returns list of (lat, lon) waypoints, or empty list if no path.
    """
    ocean_cells = set(ocean_cells_tuple)
    prob_dict = dict(zip(prob_dict_keys, prob_dict_vals))
    graph = build_ocean_graph(ocean_cells)

    if start not in graph or end not in graph:
        return []

    # Priority queue: (f_score, counter, node)
    counter = 0
    open_set = [(0, counter, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == end:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for nb in graph.get(current, []):
            dist = haversine(current[0], current[1], nb[0], nb[1])
            whale_cost = whale_weight * prob_dict.get(nb, 0)
            tentative = g_score[current] + dist + whale_cost

            if tentative < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative
                h = haversine(nb[0], nb[1], end[0], end[1])
                f = tentative + h
                counter += 1
                heapq.heappush(open_set, (f, counter, nb))

    return []  # No path found


def route_metrics(path: list, prob_dict: dict) -> dict:
    """Compute distance, time, and risk metrics for a route."""
    if len(path) < 2:
        return {"distance_km": 0, "time_hrs": 0, "avg_risk": 0, "max_risk": 0}
    total_dist = sum(
        haversine(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
        for i in range(len(path) - 1)
    )
    risks = [prob_dict.get(cell, 0) for cell in path]
    return {
        "distance_km": total_dist,
        "time_hrs": total_dist / 18.52,  # 10 knots = 18.52 km/h
        "avg_risk": np.mean(risks) if risks else 0,
        "max_risk": max(risks) if risks else 0,
    }


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
    

    # -----------------------------------------------------------------------
    # Route Planner controls
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("🚢 Route Planner")

    location_names = list(ALL_LOCATIONS.keys())
    origin_name = st.selectbox("Origin", location_names, index=0)
    # Filter destination to exclude origin
    dest_options = [n for n in location_names if n != origin_name]
    dest_name = st.selectbox("Destination", dest_options, index=min(11, len(dest_options) - 1))
    show_eco = st.toggle("Show Eco-Route", value=True,
                         help="The eco-route avoids high whale-probability cells, potentially adding distance but reducing collision risk.")


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

# ---------------------------------------------------------------------------
# Route computation
# ---------------------------------------------------------------------------
ocean_cells = set(zip(df_view["Lat"], df_view["Lon"]))
prob_dict = dict(zip(zip(df_view["Lat"], df_view["Lon"]), df_view["probability"]))

origin_info = ALL_LOCATIONS[origin_name]
dest_info = ALL_LOCATIONS[dest_name]
start_cell = snap_to_grid(origin_info["lat"], origin_info["lon"], ocean_cells)
end_cell = snap_to_grid(dest_info["lat"], dest_info["lon"], ocean_cells)

# Hashable args for caching
oc_tuple = tuple(sorted(ocean_cells))
pk = tuple(prob_dict.keys())
pv = tuple(prob_dict.values())

std_route = find_route(oc_tuple, pk, pv, start_cell, end_cell, whale_weight=0.0)
eco_route = find_route(oc_tuple, pk, pv, start_cell, end_cell, whale_weight=500.0) if show_eco else []

# ---------------------------------------------------------------------------
# Add route lines to map
# ---------------------------------------------------------------------------
if std_route:
    fig.add_trace(go.Scattermapbox(
        lat=[p[0] for p in std_route],
        lon=[p[1] for p in std_route],
        mode="lines",
        line=dict(width=3, color="#2563EB"),
        name="Standard Route",
        hoverinfo="skip",
    ))

if eco_route and show_eco:
    fig.add_trace(go.Scattermapbox(
        lat=[p[0] for p in eco_route],
        lon=[p[1] for p in eco_route],
        mode="lines",
        line=dict(width=3.5, color="#10B981"),
        name="Eco-Route",
        hoverinfo="skip",
    ))

# Port markers
if start_cell and end_cell:
    fig.add_trace(go.Scattermapbox(
        lat=[start_cell[0], end_cell[0]],
        lon=[start_cell[1], end_cell[1]],
        mode="markers+text",
        marker=dict(size=12, color=["#2563EB", "#DC2626"], symbol="circle"),
        text=[origin_name, dest_name],
        textposition="top center",
        textfont=dict(size=11, color="#1E293B"),
        name="Ports",
        hovertemplate="%{text}<extra></extra>",
    ))

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
# Route comparison metrics
# ---------------------------------------------------------------------------
if std_route:
    std_m = route_metrics(std_route, prob_dict)

    if eco_route and show_eco:
        eco_m = route_metrics(eco_route, prob_dict)
        risk_reduction = (
            (1 - eco_m["avg_risk"] / std_m["avg_risk"]) * 100
            if std_m["avg_risk"] > 0 else 0
        )

        st.subheader("🚢 Route Comparison")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown("**Standard Route**")
            st.metric("Distance", f"{std_m['distance_km']:.0f} km")
            st.metric("Travel Time", f"{std_m['time_hrs']:.1f} hrs")
            st.metric("Avg. Whale Risk", f"{std_m['avg_risk']:.3f}")
        with rc2:
            st.markdown("**Eco-Route**")
            st.metric("Distance", f"{eco_m['distance_km']:.0f} km",
                       delta=f"+{eco_m['distance_km'] - std_m['distance_km']:.0f} km",
                       delta_color="off")
            st.metric("Travel Time", f"{eco_m['time_hrs']:.1f} hrs",
                       delta=f"+{eco_m['time_hrs'] - std_m['time_hrs']:.1f} hrs",
                       delta_color="off")
            st.metric("Avg. Whale Risk", f"{eco_m['avg_risk']:.3f}",
                       delta=f"{-risk_reduction:.0f}%",
                       delta_color="normal")
        with rc3:
            st.markdown("**Savings**")
            dist_pct = ((eco_m["distance_km"] - std_m["distance_km"]) / std_m["distance_km"] * 100) if std_m["distance_km"] > 0 else 0
            st.metric("Distance Added", f"+{dist_pct:.1f}%")
            st.metric("Risk Reduction", f"{risk_reduction:.0f}%")
            st.metric("Max Risk Cell", f"{std_m['max_risk']:.3f} → {eco_m['max_risk']:.3f}")
    else:
        st.subheader("🚢 Route Info")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Distance", f"{std_m['distance_km']:.0f} km")
        rc2.metric("Travel Time", f"{std_m['time_hrs']:.1f} hrs")
        rc3.metric("Avg. Whale Risk", f"{std_m['avg_risk']:.3f}")
        rc4.metric("Max Risk Cell", f"{std_m['max_risk']:.3f}")
elif start_cell and end_cell:
    st.warning("⚠️ No route found between the selected ports for this month. The ocean grid may not connect these locations.")

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
