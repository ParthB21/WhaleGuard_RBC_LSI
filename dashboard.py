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

# Custom colorscale: transparent at low probs, fading in, strong red at high probs
CUSTOM_COLORSCALE = [
    [0.00, "rgba(255,255,255,0)"],
    [0.08, "rgba(255,255,200,0)"],
    [0.15, "rgba(254,217,118,0.25)"],
    [0.25, "rgba(254,178,76,0.5)"],
    [0.40, "rgba(253,141,60,0.8)"],
    [0.55, "rgba(240,59,32,0.9)"],
    [0.75, "rgba(189,0,38,0.95)"],
    [1.00, "rgba(128,0,38,1)"],
]

# ---------------------------------------------------------------------------
# Ports & Gateways — all coordinates snap to nearest ocean grid cell at runtime
# ---------------------------------------------------------------------------
PORTS = {
    # Quebec North Shore
    "Sept-Îles":          {"lat": 50.20, "lon": -66.38, "group": "Quebec North Shore", "info": "Iron ore — 38.5M t/yr"},
    "Port-Cartier":       {"lat": 50.03, "lon": -66.82, "group": "Quebec North Shore", "info": "Iron ore, grain"},
    "Havre-Saint-Pierre": {"lat": 50.24, "lon": -63.61, "group": "Quebec North Shore", "info": "Bulk, cruise, year-round"},
    "Baie-Comeau":        {"lat": 49.22, "lon": -68.15, "group": "Quebec North Shore", "info": "Aluminum, grain, forestry"},
    "Matane":             {"lat": 48.85, "lon": -67.57, "group": "Quebec North Shore", "info": "Ferry terminal"},
    # Quebec South Shore & Gaspé
    "Rimouski":           {"lat": 48.48, "lon": -68.52, "group": "Quebec South Shore", "info": "Research vessels, regional"},
    "Gaspé":              {"lat": 48.83, "lon": -64.44, "group": "Quebec South Shore", "info": "Wind energy, regional"},
    "Chandler":           {"lat": 48.35, "lon": -64.68, "group": "Quebec South Shore", "info": "Regional"},
    # New Brunswick
    "Belledune":          {"lat": 47.91, "lon": -65.83, "group": "New Brunswick", "info": "CPA — industrial bulk"},
    "Dalhousie":          {"lat": 48.07, "lon": -66.37, "group": "New Brunswick", "info": "Bulk, rail access"},
    "Caraquet":           {"lat": 47.78, "lon": -64.93, "group": "New Brunswick", "info": "Fishing, merchant vessels"},
    # Prince Edward Island
    "Charlottetown":      {"lat": 46.23, "lon": -63.13, "group": "Prince Edward Island", "info": "Cruise, regional"},
    "Summerside":         {"lat": 46.39, "lon": -63.79, "group": "Prince Edward Island", "info": "Agri exports"},
    # Nova Scotia
    "Sydney":             {"lat": 46.14, "lon": -60.19, "group": "Nova Scotia", "info": "Bulk terminal, cruise"},
    "Port Hawkesbury":    {"lat": 45.62, "lon": -61.35, "group": "Nova Scotia", "info": "Industrial bulk"},
    "Pictou":             {"lat": 45.68, "lon": -62.74, "group": "Nova Scotia", "info": "Commercial terminal, ISPS"},
    "Chéticamp":          {"lat": 46.64, "lon": -61.01, "group": "Nova Scotia", "info": "Fishing harbour"},
    # Newfoundland
    "Corner Brook":       {"lat": 48.95, "lon": -57.95, "group": "Newfoundland", "info": "Paper mill, bulk"},
    "Port aux Basques":   {"lat": 47.57, "lon": -59.14, "group": "Newfoundland", "info": "Marine Atlantic ferry"},
}

GATEWAYS = {
    "→ Montreal / Upstream":     {"lat": 47.25, "lon": -70.50, "group": "Exit Points", "info": "→ Montreal, Trois-Rivières, Great Lakes"},
    "→ Atlantic (Cabot Strait)": {"lat": 47.00, "lon": -59.75, "group": "Exit Points", "info": "→ Halifax, US East Coast, Europe"},
    "→ Atlantic (Belle Isle)":   {"lat": 51.00, "lon": -57.25, "group": "Exit Points", "info": "→ Northern Europe (seasonal Jun–Nov)"},
}

ALL_LOCATIONS = {**PORTS, **GATEWAYS}

# ---------------------------------------------------------------------------
# NARW Carcass Data — 21 Canadian Gulf entries from the 2017/2019 UME
# Source: NOAA National Stranding Database Visualization Tool
# ---------------------------------------------------------------------------
CARCASS_DATA = [
    {"date": "Jun 7, 2017",  "lat": 47.3204, "lon": -63.8597, "sex": "M", "condition": "Moderate", "prov": "NB",  "id": "#3746"},
    {"date": "Jun 19, 2017", "lat": 47.4423, "lon": -63.5820, "sex": "M", "condition": "Moderate", "prov": "NB",  "id": "#1402"},
    {"date": "Jun 18, 2017", "lat": 47.7431, "lon": -63.3609, "sex": "M", "condition": "Advanced", "prov": "NB",  "id": "#3190"},
    {"date": "Jun 21, 2017", "lat": 48.2335, "lon": -63.0451, "sex": "F", "condition": "Fresh",    "prov": "QC",  "id": "#3603"},
    {"date": "Jun 22, 2017", "lat": 47.1501, "lon": -62.4634, "sex": "F", "condition": "Moderate", "prov": "PEI", "id": "#3512"},
    {"date": "Jun 23, 2017", "lat": 47.6155, "lon": -63.2234, "sex": "M", "condition": "Moderate", "prov": "NB",  "id": "#1207"},
    {"date": "Jul 6, 2017",  "lat": 47.5832, "lon": -62.6267, "sex": "M", "condition": "Advanced", "prov": "QC",  "id": "Unknown"},
    {"date": "Jul 19, 2017", "lat": 47.9170, "lon": -63.9086, "sex": "M", "condition": "Fresh",    "prov": "NB",  "id": "#2140"},
    {"date": "Jul 21, 2017", "lat": 49.3485, "lon": -58.2350, "sex": "M", "condition": "Advanced", "prov": "NL",  "id": "#2630"},
    {"date": "Jul 27, 2017", "lat": 47.6206, "lon": -59.2980, "sex": "F", "condition": "Advanced", "prov": "NL",  "id": "Unknown"},
    {"date": "Jul 30, 2017", "lat": 50.4263, "lon": -57.4990, "sex": "F", "condition": "Advanced", "prov": "NL",  "id": "#1911"},
    {"date": "Sep 15, 2017", "lat": 48.1503, "lon": -63.5009, "sex": "F", "condition": "Moderate", "prov": "QC",  "id": "#4504"},
    {"date": "Jun 4, 2019",  "lat": 48.2500, "lon": -63.1383, "sex": "M", "condition": "Moderate", "prov": "QC",  "id": "#4023"},
    {"date": "Jun 20, 2019", "lat": 47.6139, "lon": -60.7675, "sex": "F", "condition": "Moderate", "prov": "QC",  "id": "#1281"},
    {"date": "Jun 25, 2019", "lat": 47.8346, "lon": -64.0524, "sex": "M", "condition": "Moderate", "prov": "NB",  "id": "#1514"},
    {"date": "Jun 25, 2019", "lat": 47.5475, "lon": -62.5600, "sex": "F", "condition": "Advanced", "prov": "QC",  "id": "#3815"},
    {"date": "Jun 26, 2019", "lat": 49.0777, "lon": -61.7912, "sex": "F", "condition": "Moderate", "prov": "QC",  "id": "#3329"},
    {"date": "Jun 27, 2019", "lat": 48.3493, "lon": -63.1655, "sex": "F", "condition": "Moderate", "prov": "QC",  "id": "#3450"},
    {"date": "Jun 24, 2019", "lat": 46.3250, "lon": -59.8750, "sex": "U", "condition": "Moderate", "prov": "NS",  "id": "Unknown"},
    {"date": "Jul 18, 2019", "lat": 48.1640, "lon": -62.7512, "sex": "M", "condition": "Advanced", "prov": "QC",  "id": "#3421"},
    {"date": "Jul 21, 2019", "lat": 45.6881, "lon": -59.9561, "sex": "U", "condition": "Advanced", "prov": "NS",  "id": "Unknown"},
]

# ---------------------------------------------------------------------------
# Transport Canada Restriction Zones (Ship Safety Bulletin coordinates)
# ---------------------------------------------------------------------------
RESTRICTION_ZONES = {
    "Northern Static Zone": {
        "lats": [50.333, 49.217, 48.667, 48.667, 48.050, 47.968, 48.000, 49.067, 49.067, 49.717, 50.333, 50.333],
        "lons": [-65.000, -65.000, -64.217, -62.667, -61.125, -61.058, -61.000, -61.000, -62.000, -63.000, -63.000, -65.000],
        "color": "rgba(220,38,38,0.15)",
        "border": "rgba(220,38,38,0.6)",
        "label": "Northern Static (10 kn)",
    },
    "Southern Static Zone": {
        "lats": [48.667, 48.667, 48.050, 47.968, 47.167, 47.167, 48.667],
        "lons": [-65.000, -62.667, -61.125, -61.058, -62.500, -65.000, -65.000],
        "color": "rgba(249,115,22,0.15)",
        "border": "rgba(249,115,22,0.6)",
        "label": "Southern Static (10 kn)",
    },
    "Dynamic Zone A": {
        "lats": [49.683, 49.333, 49.183, 49.367, 49.683],
        "lons": [-65.000, -65.000, -64.000, -64.000, -65.000],
        "color": "rgba(16,185,129,0.15)",
        "border": "rgba(16,185,129,0.6)",
        "label": "Dynamic A",
    },
    "Dynamic Zone B": {
        "lats": [49.367, 49.183, 48.800, 49.000, 49.367],
        "lons": [-64.000, -64.000, -63.000, -63.000, -64.000],
        "color": "rgba(16,185,129,0.15)",
        "border": "rgba(16,185,129,0.6)",
        "label": "Dynamic B",
    },
    "Dynamic Zone C": {
        "lats": [49.000, 48.800, 48.400, 48.583, 49.000],
        "lons": [-63.000, -63.000, -62.000, -62.000, -63.000],
        "color": "rgba(16,185,129,0.15)",
        "border": "rgba(16,185,129,0.6)",
        "label": "Dynamic C",
    },
    "Dynamic Zone D": {
        "lats": [50.267, 50.000, 49.933, 50.267, 50.267],
        "lons": [-64.000, -64.000, -63.000, -63.000, -64.000],
        "color": "rgba(16,185,129,0.15)",
        "border": "rgba(16,185,129,0.6)",
        "label": "Dynamic D",
    },
}

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
        (-0.50, 0.25), (-0.50, 0.50), (-0.25, 0.50),
        (0.50, 0.25), (0.50, 0.50), (0.25, 0.50),
        (-0.50, -0.25), (-0.50, -0.50), (-0.25, -0.50),
        (0.50, -0.25), (0.50, -0.50), (0.25, -0.50),
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
    zone_cells_tuple: tuple,
    start: tuple,
    end: tuple,
    whale_penalty_hrs: float = 0.0,
) -> list:
    """
    A* pathfinding over the ocean grid, optimized for TRAVEL TIME.

    Standard route (whale_penalty_hrs = 0): Finds the fastest route, naturally 
    balancing shorter distance vs the 10-knot speed limits in restriction zones.
    
    Eco-route (whale_penalty_hrs > 0): Adds a time penalty for whale-probable cells.
    e.g., whale_penalty_hrs = 10 means the algorithm will take up to a 10-hour 
    detour to avoid a cell with 100% whale probability.

    Returns list of (lat, lon) waypoints, or empty list if no path.
    """
    SPEED_OPEN = 25.93  # 14 knots
    SPEED_ZONE = 18.52  # 10 knots

    ocean_cells = set(ocean_cells_tuple)
    prob_dict = dict(zip(prob_dict_keys, prob_dict_vals))
    zone_cells = set(zone_cells_tuple)
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
            # Time cost for this segment
            speed = SPEED_ZONE if nb in zone_cells else SPEED_OPEN
            time_cost = dist / speed
            
            # Additional risk penalty
            base_risk = whale_penalty_hrs * prob_dict.get(nb, 0)
            
            # CRITICAL: If a cell is in an active restriction zone, ships must 
            # travel at 10 knots. This mitigates the risk of a lethal strike by ~80%.
            # Therefore, we heavily discount the risk penalty for these cells.
            risk_cost = base_risk * 0.2 if nb in zone_cells else base_risk
            
            tentative = g_score[current] + time_cost + risk_cost

            if tentative < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative
                
                # Heuristic: straight line to target at max speed
                h = (haversine(nb[0], nb[1], end[0], end[1]) / SPEED_OPEN)
                f = tentative + h
                counter += 1
                heapq.heappush(open_set, (f, counter, nb))

    return []  # No path found


def _point_in_polygon(lat: float, lon: float, poly_lats: list, poly_lons: list) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(poly_lats)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = poly_lats[i], poly_lons[i]
        yj, xj = poly_lats[j], poly_lons[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


@st.cache_data(show_spinner=False)
def compute_all_zone_cells(ocean_cells_tuple: tuple) -> dict:
    """Pre-compute which ocean cells fall inside each restriction zone."""
    zone_cells_dict = {name: set() for name in RESTRICTION_ZONES}
    for cell in ocean_cells_tuple:
        for name, zone in RESTRICTION_ZONES.items():
            if _point_in_polygon(cell[0], cell[1], zone["lats"], zone["lons"]):
                zone_cells_dict[name].add(cell)
    return zone_cells_dict


def get_active_zone_cells(zone_cells_dict: dict, prob_dict: dict) -> set:
    """
    Determine which zones are active. 
    Static zones are always active.
    Dynamic zones are active if max P(whale) in the zone >= 0.40 (Core Habitat).
    Returns a flattened set of all active cells.
    """
    active_cells = set()
    for name, cells in zone_cells_dict.items():
        if "Static" in name:
            active_cells.update(cells)
        else:
            # Dynamic zone
            max_p = max((prob_dict.get(c, 0) for c in cells), default=0)
            if max_p >= 0.40:
                active_cells.update(cells)
    return active_cells


def smooth_path_moving_average(path: list, ocean_cells: set, iterations: int = 15) -> list:
    """
    Smooths a path using Laplacian smoothing (3-point rolling average)
    constrained to never drift onto land (remains snapped/close to ocean cells).
    """
    if not path or len(path) <= 2:
        return path
    
    smoothed = list(path)
    for _ in range(iterations):
        next_smoothed = [smoothed[0]]
        for i in range(1, len(smoothed) - 1):
            prev_p = smoothed[i - 1]
            curr_p = smoothed[i]
            next_p = smoothed[i + 1]
            
            # Weighted average
            avg_lat = 0.25 * prev_p[0] + 0.5 * curr_p[0] + 0.25 * next_p[0]
            avg_lon = 0.25 * prev_p[1] + 0.5 * curr_p[1] + 0.25 * next_p[1]
            
            # Verify snap-cell is in the ocean
            grid_lat = round(avg_lat * 4.0) / 4.0
            grid_lon = round(avg_lon * 4.0) / 4.0
            if (grid_lat, grid_lon) in ocean_cells:
                next_smoothed.append((avg_lat, avg_lon))
            else:
                next_smoothed.append(curr_p)  # Keep original to avoid land drift
        next_smoothed.append(smoothed[-1])
        smoothed = next_smoothed
    return smoothed


def route_metrics(path: list, prob_dict: dict, zone_cells: set) -> dict:
    """Compute distance, time (zone-aware), and risk metrics for a route.

    Coordinates are snapped to the nearest 0.25-degree cell to check zone limits 
    and lookup whale probability, ensuring smoothed paths report accurate metrics.
    """
    SPEED_OPEN = 25.93    # 14 knots in km/h (typical commercial speed)
    SPEED_ZONE = 18.52    # 10 knots in km/h (mandatory zone speed)

    if len(path) < 2:
        return {"distance_km": 0, "time_hrs": 0, "avg_risk": 0, "max_risk": 0, "zone_cells": 0}

    total_dist = 0
    total_time = 0
    n_zone = 0
    for i in range(len(path) - 1):
        seg_dist = haversine(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
        total_dist += seg_dist
        
        # Snap endpoints to grid to check zone presence
        cell1 = (round(path[i][0] * 4.0) / 4.0, round(path[i][1] * 4.0) / 4.0)
        cell2 = (round(path[i + 1][0] * 4.0) / 4.0, round(path[i + 1][1] * 4.0) / 4.0)
        
        if cell1 in zone_cells or cell2 in zone_cells:
            total_time += seg_dist / SPEED_ZONE
            n_zone += 1
        else:
            total_time += seg_dist / SPEED_OPEN

    # Snap cells to look up risk
    snapped_path = [(round(lat * 4.0) / 4.0, round(lon * 4.0) / 4.0) for lat, lon in path]
    risks = [prob_dict.get(cell, 0) for cell in snapped_path]
    return {
        "distance_km": total_dist,
        "time_hrs": total_time,
        "avg_risk": np.mean(risks) if risks else 0,
        "max_risk": max(risks) if risks else 0,
        "zone_cells": n_zone,
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
    st.title("🐋 WhaleGuard")
    st.caption("NARW Habitat Prediction Dashboard")
    st.divider()

    # Default to Random Forest — highest AUC (0.9064)
    model_name = "Random Forest"

    st.subheader("📅 Date Filter")
    year = st.slider("Year", min_value=2002, max_value=2026, value=2024)
    month = st.select_slider(
        "Month",
        options=list(range(1, 13)),
        value=6,
        format_func=lambda x: MONTH_ABBRS[x],
    )

    st.divider()
    st.subheader("🗺 Map Layers")
    heatmap_radius = st.slider(
        "Heatmap smoothing",
        min_value=10, max_value=60, value=30,
        help="Larger values = smoother heatmap; smaller = sharper grid cells.",
    )
    show_ports = st.toggle("Show Ports", value=True)
    show_carcasses = st.toggle("☠ Carcass Locations", value=True,
                               help="21 confirmed NARW carcasses from the 2017/2019 Unusual Mortality Events in the Gulf.")
    show_zones = st.toggle("🚧 Restriction Zones", value=True,
                           help="Transport Canada mandatory speed restriction zones (10 knots).")

    meta = MODELS_META[model_name]
    st.divider()
    st.subheader("Model Performance")
    st.metric("ROC-AUC", meta["auc"])
    st.caption("Random Forest · 627 trees · Temporal hold-out")

    # -----------------------------------------------------------------------
    # Route Planner controls
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("🚢 Route Planner")

    location_names = list(ALL_LOCATIONS.keys())
    origin_name = st.selectbox("Origin", location_names, index=0)
    # Filter destination to exclude origin
    dest_options = [n for n in location_names if n != origin_name]
    dest_name = st.selectbox("Destination", dest_options, index=min(13, len(dest_options) - 1))
    show_eco = st.toggle("Show WhaleGuard Route", value=True,
                         help="AI-optimized route that avoids whale habitat and restriction zones, potentially adding distance but reducing collision risk and zone slowdowns.")

    # Advanced — model switcher (hidden by default)
    with st.expander("⚙️ Advanced"):
        model_name = st.selectbox(
            "Switch Model",
            list(MODELS_META.keys()),
            help="Random Forest is recommended. Other models shown for comparison.",
        )
        meta = MODELS_META[model_name]


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
# Pre-compute environment and active zones
# ---------------------------------------------------------------------------
ocean_cells = set(zip(df_view["Lat"], df_view["Lon"]))
prob_dict = dict(zip(zip(df_view["Lat"], df_view["Lon"]), df_view["probability"]))

# Hashable args for caching
oc_tuple = tuple(sorted(ocean_cells))
pk = tuple(prob_dict.keys())
pv = tuple(prob_dict.values())

all_zone_cells_dict = compute_all_zone_cells(oc_tuple)
active_zone_cells_set = get_active_zone_cells(all_zone_cells_dict, prob_dict)
active_zone_cells_tuple = tuple(active_zone_cells_set)

# Determine zone active status for UI/rendering
zone_status = {}
for zname in RESTRICTION_ZONES:
    if "Static" in zname:
        zone_status[zname] = True
    else:
        # Dynamic zone triggered if max P >= 0.40
        max_p = max((prob_dict.get(c, 0) for c in all_zone_cells_dict[zname]), default=0)
        zone_status[zname] = (max_p >= 0.40)

# ---------------------------------------------------------------------------
# Header + KPI cards (3-tier risk stratification)
# ---------------------------------------------------------------------------
st.title("WhaleGuard — NARW Habitat Prediction")
st.caption(f"Showing **{MONTH_ABBRS[month]} {year}** · Model: **{model_name}** (AUC {meta['auc']})")

core_habitat = (df_view["probability"] >= 0.40).mean()
caution_zone = ((df_view["probability"] >= 0.20) & (df_view["probability"] < 0.40)).mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Grid cells", f"{len(df_view):,}",
          help="Number of 0.25° ocean grid cells covering the Gulf of St. Lawrence.")
k2.metric("🔴 Core Habitat", f"{core_habitat:.1%}",
          help=(
              "Fraction of cells with P(whale) ≥ 0.40 — high-confidence habitat. "
              "Ships should reroute around these areas when possible."
          ))
k3.metric("🟡 Caution Zone", f"{caution_zone:.1%}",
          help=(
              "Fraction of cells with 0.20 ≤ P(whale) < 0.40 — elevated risk. "
              "Speed reduction to 10 knots recommended."
          ))
k4.metric("Max probability", f"{df_view['probability'].max():.3f}",
          help="Highest single-cell predicted probability this month.")

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
fig = go.Figure()

# Density heatmap layer — custom colorscale with transparency at low values
fig.add_trace(
    go.Densitymapbox(
        lat=df_view["Lat"],
        lon=df_view["Lon"],
        z=df_view["probability"],
        radius=heatmap_radius,
        colorscale=CUSTOM_COLORSCALE,
        zmin=0,
        zmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text="P(Whale)", side="right"),
            thickness=14,
            len=0.75,
            tickvals=[0, 0.20, 0.40, 0.70, 1.0],
            ticktext=["0", "0.20", "0.40 🔴", "0.70", "1.0"],
        ),
        name="Whale probability",
        hovertemplate="Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>P(whale): %{z:.3f}<extra></extra>",
    )
)

# ---------------------------------------------------------------------------
# Transport Canada Restriction Zone polygons
# ---------------------------------------------------------------------------
if show_zones:
    for zone_name, zone in RESTRICTION_ZONES.items():
        is_active = zone_status[zone_name]
        
        # Fill polygon
        fig.add_trace(go.Scattermapbox(
            lat=zone["lats"],
            lon=zone["lons"],
            mode="lines",
            fill="toself" if is_active else "none",
            fillcolor=zone["color"] if is_active else "rgba(0,0,0,0)",
            line=dict(
                width=1.5 if is_active else 0.8, 
                color=zone["border"] if is_active else "rgba(156,163,175,0.3)",
            ),
            name=zone["label"],
            hovertemplate=f"<b>{zone['label']}</b><br>Status: {'Active (10kn)' if is_active else 'Inactive (Clear)'}<extra></extra>",
        ))

# ---------------------------------------------------------------------------
# Carcass markers
# ---------------------------------------------------------------------------
if show_carcasses:
    # Outer glow ring for visibility
    fig.add_trace(go.Scattermapbox(
        lat=[c["lat"] for c in CARCASS_DATA],
        lon=[c["lon"] for c in CARCASS_DATA],
        mode="markers",
        marker=dict(size=18, color="rgba(220,38,38,0.3)"),
        showlegend=False,
        hoverinfo="skip",
    ))
    # Main carcass markers
    fig.add_trace(go.Scattermapbox(
        lat=[c["lat"] for c in CARCASS_DATA],
        lon=[c["lon"] for c in CARCASS_DATA],
        mode="markers",
        marker=dict(
            size=11,
            color="#FF3B3B",
            opacity=0.95,
        ),
        name="☠ NARW Carcasses (21)",
        hovertemplate=[
            f"<b>☠ NARW Carcass</b><br>"
            f"{c['date']}<br>"
            f"Sex: {c['sex']} · Condition: {c['condition']}<br>"
            f"Province: {c['prov']} · Whale: {c['id']}<br>"
            f"Location: ({c['lat']:.4f}°N, {abs(c['lon']):.4f}°W)"
            "<extra></extra>"
            for c in CARCASS_DATA
        ],
    ))

# ---------------------------------------------------------------------------
# Permanent port markers
# ---------------------------------------------------------------------------
if show_ports:
    # Split ports into selected (highlighted) vs others
    selected_names = {origin_name, dest_name}

    other_ports = {k: v for k, v in PORTS.items() if k not in selected_names}
    sel_ports = {k: v for k, v in PORTS.items() if k in selected_names}

    # Other ports — subtle
    if other_ports:
        fig.add_trace(go.Scattermapbox(
            lat=[p["lat"] for p in other_ports.values()],
            lon=[p["lon"] for p in other_ports.values()],
            mode="markers+text",
            marker=dict(size=7, color="#94A3B8"),
            text=list(other_ports.keys()),
            textposition="top center",
            textfont=dict(size=9, color="#CBD5E1"),
            name="Ports",
            hovertemplate=[
                f"<b>{n}</b><br>{PORTS[n]['group']}<br>{PORTS[n]['info']}<extra></extra>"
                for n in other_ports
            ],
        ))

    # Selected ports — highlighted bright
    if sel_ports:
        fig.add_trace(go.Scattermapbox(
            lat=[p["lat"] for p in sel_ports.values()],
            lon=[p["lon"] for p in sel_ports.values()],
            mode="markers+text",
            marker=dict(size=12, color="#FBBF24", opacity=1.0),
            text=list(sel_ports.keys()),
            textposition="top center",
            textfont=dict(size=11, color="#FDE68A", family="Arial Black"),
            name="Selected Ports",
            hovertemplate=[
                f"<b>⚓ {n}</b><br>{PORTS[n]['group']}<br>{PORTS[n]['info']}<extra></extra>"
                for n in sel_ports
            ],
        ))

    # Gateway markers (always shown differently)
    other_gw = {k: v for k, v in GATEWAYS.items() if k not in selected_names}
    sel_gw = {k: v for k, v in GATEWAYS.items() if k in selected_names}

    if other_gw:
        fig.add_trace(go.Scattermapbox(
            lat=[g["lat"] for g in other_gw.values()],
            lon=[g["lon"] for g in other_gw.values()],
            mode="markers+text",
            marker=dict(size=8, color="#60A5FA"),
            text=list(other_gw.keys()),
            textposition="top center",
            textfont=dict(size=9, color="#93C5FD"),
            name="Gateways",
            hovertemplate=[
                f"<b>{n}</b><br>{GATEWAYS[n]['info']}<extra></extra>"
                for n in other_gw
            ],
        ))

    if sel_gw:
        fig.add_trace(go.Scattermapbox(
            lat=[g["lat"] for g in sel_gw.values()],
            lon=[g["lon"] for g in sel_gw.values()],
            mode="markers+text",
            marker=dict(size=12, color="#FBBF24", opacity=1.0),
            text=list(sel_gw.keys()),
            textposition="top center",
            textfont=dict(size=11, color="#FDE68A", family="Arial Black"),
            name="Selected Gateway",
            hovertemplate=[
                f"<b>⚓ {n}</b><br>{GATEWAYS[n]['info']}<extra></extra>"
                for n in sel_gw
            ],
        ))

# ---------------------------------------------------------------------------
# Route computation
# ---------------------------------------------------------------------------
origin_info = ALL_LOCATIONS[origin_name]
dest_info = ALL_LOCATIONS[dest_name]
start_cell = snap_to_grid(origin_info["lat"], origin_info["lon"], ocean_cells)
end_cell = snap_to_grid(dest_info["lat"], dest_info["lon"], ocean_cells)

# Standard = shortest path (what ships do today WITHOUT WhaleGuard)
# This deliberately goes through danger zones to illustrate the problem
std_route = find_route(oc_tuple, pk, pv, active_zone_cells_tuple, start_cell, end_cell,
                       whale_penalty_hrs=0.0)
std_route = smooth_path_moving_average(std_route, ocean_cells)

# Eco = WhaleGuard recommended route (avoids whale habitat + restriction zones)
eco_route = find_route(oc_tuple, pk, pv, active_zone_cells_tuple, start_cell, end_cell,
                       whale_penalty_hrs=20.0) if show_eco else []
eco_route = smooth_path_moving_average(eco_route, ocean_cells)

# ---------------------------------------------------------------------------
# Add route lines to map
# ---------------------------------------------------------------------------
if std_route:
    # Glow outline
    fig.add_trace(go.Scattermapbox(
        lat=[p[0] for p in std_route],
        lon=[p[1] for p in std_route],
        mode="lines",
        line=dict(width=7, color="rgba(239,68,68,0.25)"),
        name="Direct Route",
        showlegend=False,
        hoverinfo="skip",
    ))
    # Main line
    fig.add_trace(go.Scattermapbox(
        lat=[p[0] for p in std_route],
        lon=[p[1] for p in std_route],
        mode="lines",
        line=dict(width=3.5, color="#EF4444"),
        name="Direct Route (Current)",
        hoverinfo="skip",
    ))

if eco_route and show_eco:
    # Glow outline
    fig.add_trace(go.Scattermapbox(
        lat=[p[0] for p in eco_route],
        lon=[p[1] for p in eco_route],
        mode="lines",
        line=dict(width=7, color="rgba(16,185,129,0.25)"),
        name="WhaleGuard Route",
        showlegend=False,
        hoverinfo="skip",
    ))
    # Main line
    fig.add_trace(go.Scattermapbox(
        lat=[p[0] for p in eco_route],
        lon=[p[1] for p in eco_route],
        mode="lines",
        line=dict(width=4.5, color="#10B981"),
        name="WhaleGuard Route ✓",
        hoverinfo="skip",
    ))

# Origin / Destination markers (always on top)
if start_cell and end_cell:
    fig.add_trace(go.Scattermapbox(
        lat=[start_cell[0], end_cell[0]],
        lon=[start_cell[1], end_cell[1]],
        mode="markers+text",
        marker=dict(size=14, color=["#3B82F6", "#EF4444"], symbol="circle"),
        text=[origin_name, dest_name],
        textposition="top center",
        textfont=dict(size=12, color="#F1F5F9"),
        name="Route Endpoints",
        hovertemplate="%{text}<extra></extra>",
    ))

fig.update_layout(
    mapbox=dict(
        style="carto-darkmatter",
        center=dict(lat=48, lon=-63),   # Gulf of St. Lawrence
        zoom=5,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.02,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(0,0,0,0.6)",
        font=dict(color="#E2E8F0"),
    ),
    height=700,
    margin=dict(l=0, r=0, t=30, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
)

st.plotly_chart(fig, use_container_width=True)

# Three-layer context box
with st.expander("ℹ️ Understanding This Map — Three Layers of Intelligence", expanded=False):
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.markdown("""
**🔴 AI Habitat Prediction** *(Heatmap)*

Identifies where oceanographic conditions (SST, chlorophyll, bathymetry, distance to shore)
favor NARW presence. Trained on 64,000+ samples with 80% recall guarantee.

*Note: The model predicts environmental suitability, not exact whale locations.
Darker areas = higher probability of encounter.*
""")
    with lc2:
        st.markdown("""
**☠ Historical Mortality** *(Red Dots)*

21 confirmed NARW carcass locations from the 2017 and 2019 Unusual Mortality Events — the worst
die-offs in the species' history. These show where whales have actually been killed,
validating our predictions.

*Source: NOAA National Stranding Database*
""")
    with lc3:
        st.markdown("""
**🚧 Regulatory Zones** *(Outlined Areas)*

Transport Canada's mandatory 10-knot speed restriction zones.
The Northern and Southern Static Zones are active all season.
Dynamic Zones (A–D) activate for 15 days when a whale is detected.

*Source: Ship Safety Bulletin No. 04/2025*
""")

# ---------------------------------------------------------------------------
# Route comparison metrics
# ---------------------------------------------------------------------------
if std_route:
    std_m = route_metrics(std_route, prob_dict, active_zone_cells_set)

    if eco_route and show_eco:
        eco_m = route_metrics(eco_route, prob_dict, active_zone_cells_set)
        risk_reduction = (
            (1 - eco_m["avg_risk"] / std_m["avg_risk"]) * 100
            if std_m["avg_risk"] > 0 else 0
        )

        st.subheader("🚢 Route Comparison")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown("🔴 **Direct Route** *(Current Practice)*")
            st.metric("Distance", f"{std_m['distance_km']:.0f} km")
            st.metric("Travel Time", f"{std_m['time_hrs']:.1f} hrs",
                       help="14 kn open water, 10 kn inside restriction zones")
            st.metric("Avg. Whale Risk", f"{std_m['avg_risk']:.3f}")
            st.metric("Zone Slowdowns", f"{std_m['zone_cells']} segments",
                       help="Segments where the ship is forced to 10 kn due to Transport Canada zones")
        with rc2:
            st.markdown("🟢 **WhaleGuard Route** *(Recommended)*")
            st.metric("Distance", f"{eco_m['distance_km']:.0f} km",
                       delta=f"+{eco_m['distance_km'] - std_m['distance_km']:.0f} km",
                       delta_color="off")
            time_diff = eco_m['time_hrs'] - std_m['time_hrs']
            st.metric("Travel Time", f"{eco_m['time_hrs']:.1f} hrs",
                       delta=f"{time_diff:+.1f} hrs",
                       delta_color="inverse")
            st.metric("Avg. Whale Risk", f"{eco_m['avg_risk']:.3f}",
                       delta=f"{-risk_reduction:.0f}%",
                       delta_color="normal")
            st.metric("Zone Slowdowns", f"{eco_m['zone_cells']} segments",
                       delta=f"{eco_m['zone_cells'] - std_m['zone_cells']:+d}",
                       delta_color="inverse")
        with rc3:
            st.markdown("📊 **Impact Summary**")
            dist_pct = ((eco_m["distance_km"] - std_m["distance_km"]) / std_m["distance_km"] * 100) if std_m["distance_km"] > 0 else 0
            st.metric("Distance Trade-off", f"+{dist_pct:.1f}%")
            st.metric("Risk Reduction", f"{risk_reduction:.0f}%")
            st.metric("Max Exposure", f"{std_m['max_risk']:.3f} → {eco_m['max_risk']:.3f}")
            if time_diff <= 0:
                st.success(f"✅ Eco-route is **faster** by {abs(time_diff):.1f} hrs (avoids zone slowdowns)")
            elif dist_pct < 10:
                st.info(f"ℹ️ Only +{dist_pct:.1f}% distance for {risk_reduction:.0f}% less whale risk")
    else:
        st.subheader("🚢 Route Info")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Distance", f"{std_m['distance_km']:.0f} km")
        rc2.metric("Travel Time", f"{std_m['time_hrs']:.1f} hrs",
                   help="14 kn outside zones, 10 kn inside restriction zones")
        rc3.metric("Avg. Whale Risk", f"{std_m['avg_risk']:.3f}")
        rc4.metric("Zone Segments", f"{std_m['zone_cells']}")
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
