"""
Build a tabular whale presence dataset from local acoustic/visual CSVs plus optional remote env columns.

Why rows often have empty SST / Frontal / Chlorophyll / Salinity (flags all 0 and Any_env_missing = 1)
-----------------------------------------------------------------------------------------------
- **No fabrication**: if ERDDAP returns 404, empty grid, timeouts, or no valid numeric pixels, we
  leave values blank. That is expected for old dates, coastal/cloud gaps, or when the run has no
  network (e.g. sandbox).
- **Not caused by strict lat/lon**: coordinates are only checked for basic validity (-90..90,
  -180..180). Rounding keys to 3 decimals does not block ERDDAP; it deduplicates HTTP calls.

Paper (Ji et al., Sci Rep 2024) vs this script
----------------------------------------------
The paper merged **glider profiles** (temperature, salinity, oxygen, depth at 0.25 m) with
**satellite** fields (frontal value, water mass, SST, chlorophyll) from a **UDel ERDDAP** stack
(GOES / VIIRS / MODIS composites, with kriging / RF imputation for gaps in their pipeline).

This repository script does **not** pull Rutgers glider science profiles. For satellite-like columns
it queries **NOAA CoastWatch ERDDAP** (different server and products than the paper):

| Concept in paper        | This script (griddap on coastwatch.pfeg.noaa.gov)                         |
|-------------------------|---------------------------------------------------------------------------|
| Sea surface temperature | `jplMURSST41` → variable `analysed_sst` (JPL MUR, ~daily)                 |
| Frontal value (proxy)   | max |∇SST| on the same MUR subgrid (not Oliver & Irwin water-mass fronts)   |
| Chlorophyll             | `erdMH1chla8day` → `chlorophyll` (8-day composite; not VIIRS/MODIS paper) |
| Salinity (surface)      | `coastwatchSMOSv662SSS1day` → `sss` (SMOS; not glider salinity)           |

So: **same general idea** (remote sensing covariates), **not** the identical databases or the
paper’s gap-filling / kriging / SMOTE (training-only in the paper).

Speed
-----
With `--no-env`, there are no HTTP calls (very fast). With env fetch, work is **per unique
(DATE_KEY, LAT_KEY, LON_KEY)** — not per CSV row — but each key still does several ERDDAP requests
(SST+front, chlorophyll, salinity) with retries; local runs with good network are slower than a
sandbox with blocked or failed requests.

### Example commands (from project root)
```bash
source venv/bin/activate
# 1) Real-only NOAA CoastWatch products (MUR SST + gradient, 8-day chl, SMOS SSS)
python build_real_whale_dataset.py --mode coastwatch --random-run-dir --no-visual

# 2) Paper-style UDel ladder (GOES → VIIRS → MODIS) + M_WK / M_WK_G (+ optional SMOS salinity)
python build_real_whale_dataset.py --mode paper --random-run-dir --no-visual
```
Outputs land in `data/runs/run_<timestamp>_<hex>/` when using `--random-run-dir`.
"""

import argparse
import io
import json
import os
from pathlib import Path
import re
import secrets
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests



PROJECT_ROOT = str(Path(__file__).resolve().parent)
ACOUSTIC_ROOT = os.path.join(PROJECT_ROOT, "data", "raw", "acoustic")
VISUAL_CSV = os.path.join(PROJECT_ROOT, "otherData", "23305_RWSAS.csv")

COASTWATCH_GRIDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
UDEL_GRIDDAP = "https://basin.ceoe.udel.edu/erddap/griddap"

# Known coverage windows so we don't waste time querying dates outside product availability.
DATASET_DATE_BOUNDS: Dict[str, Tuple[str, str]] = {
    "jplMURSST41": ("2002-06-01", "2100-01-01"),
    "erdMH1chla8day": ("2003-01-01", "2022-06-14"),
    "coastwatchSMOSv662SSS1day": ("2010-06-01", "2100-01-01"),
    "coastwatchSMOSv662SSS3day": ("2010-06-01", "2100-01-01"),
    "daily_composite_JPL_SST": ("2018-01-01", "2025-04-08"),
    "VIIRS_NWATL": ("2012-01-03", "2100-01-01"),
    "MODIS_AQUA_3_day": ("2002-07-03", "2022-10-02"),
}


@dataclass
class EnvConfig:
    request_timeout_connect: int = 5
    request_timeout_read: int = 12
    polite_sleep_seconds: float = 0.05
    max_seconds_per_var: int = 25
    enable_env_fetch: bool = True
    mode: str = "coastwatch"
    verbose: bool = True
    env_cache_path: Optional[str] = None
    checkpoint_every_points: int = 25
    max_env_points_per_run: Optional[int] = None


def _safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_lat_lon_from_metadata(metadata: dict) -> Tuple[Optional[float], Optional[float]]:
    if not metadata:
        return None, None

    shape_value = metadata.get("SHAPE") or metadata.get("shape")
    if isinstance(shape_value, str):
        match = re.search(r"POINT \(([^ ]+) ([^ ]+)\)", shape_value)
        if match:
            try:
                lon = float(match.group(1))
                lat = float(match.group(2))
                return lat, lon
            except Exception:
                pass

    deployment = metadata.get("DEPLOYMENT", {})
    lat = deployment.get("DEPLOY_LAT")
    lon = deployment.get("DEPLOY_LON")
    try:
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    except Exception:
        return None, None
    return None, None


def load_acoustic_daily_presence(acoustic_root: str) -> pd.DataFrame:
    rows: List[dict] = []
    if not os.path.isdir(acoustic_root):
        return pd.DataFrame()

    for package_name in sorted(os.listdir(acoustic_root)):
        package_path = os.path.join(acoustic_root, package_name)
        if not os.path.isdir(package_path):
            continue

        metadata_dir = os.path.join(package_path, "metadata")
        data_dir = os.path.join(package_path, "data")
        if not os.path.isdir(metadata_dir) or not os.path.isdir(data_dir):
            continue

        metadata_files = [f for f in os.listdir(metadata_dir) if f.lower().endswith(".json")]
        csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
        if not metadata_files or not csv_files:
            continue

        metadata = _load_json(os.path.join(metadata_dir, metadata_files[0]))
        lat, lon = _extract_lat_lon_from_metadata(metadata or {})
        if lat is None or lon is None:
            continue

        for csv_name in csv_files:
            csv_path = os.path.join(data_dir, csv_name)
            df = _safe_read_csv(csv_path)
            if df.empty or "ISOStartTime" not in df.columns or "Presence" not in df.columns:
                continue

            temp = pd.DataFrame()
            temp["SIGHTINGDATE"] = pd.to_datetime(df["ISOStartTime"], errors="coerce").dt.date
            temp["Presence"] = pd.to_numeric(df["Presence"], errors="coerce")
            temp = temp.dropna(subset=["SIGHTINGDATE", "Presence"]).copy()
            temp["Presence"] = (temp["Presence"] > 0).astype(int)
            temp["LAT"] = float(lat)
            temp["LON"] = float(lon)
            temp["SOURCE"] = "Acoustic"
            temp["SOURCE_DETAIL"] = "SanctSound/NOAA daily product"
            temp["PACKAGE_ID"] = package_name
            rows.extend(temp.to_dict(orient="records"))

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Consolidate duplicate day/site rows by max presence.
    out = (
        out.groupby(["SIGHTINGDATE", "LAT", "LON", "SOURCE", "SOURCE_DETAIL", "PACKAGE_ID"], as_index=False)["Presence"]
        .max()
        .sort_values(["SIGHTINGDATE", "PACKAGE_ID"])
    )
    return out


def load_visual_positives(visual_csv_path: str) -> pd.DataFrame:
    df = _safe_read_csv(visual_csv_path)
    if df.empty:
        return pd.DataFrame()

    needed = {"SIGHTINGDATE", "LAT", "LON"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    if "CERTAINTY" in df.columns:
        certainty = df["CERTAINTY"].astype(str).str.strip().str.lower()
        df = df[certainty.isin(["definite", "probable"])].copy()

    if "DUPLICATE" in df.columns:
        duplicate = df["DUPLICATE"].astype(str).str.strip().str.lower().isin(["yes", "true", "1"])
        df = df[~duplicate].copy()

    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    parsed = pd.to_datetime(df["SIGHTINGDATE"], format="%d-%b-%y", errors="coerce")
    fallback_mask = parsed.isna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(df.loc[fallback_mask, "SIGHTINGDATE"], errors="coerce")
    df["SIGHTINGDATE"] = parsed.dt.date
    df = df.dropna(subset=["SIGHTINGDATE", "LAT", "LON"]).copy()

    # Correct occasional positive-longitude entries for western Atlantic context.
    mask = (df["LON"] > 0) & df["LAT"].between(20, 65)
    df.loc[mask, "LON"] = -df.loc[mask, "LON"]

    df = df[df["LAT"].between(-90, 90) & df["LON"].between(-180, 180)].copy()
    df["Presence"] = 1
    df["SOURCE"] = "Visual"
    df["SOURCE_DETAIL"] = "RWSAS sightings"
    df["PACKAGE_ID"] = pd.NA
    return df[["SIGHTINGDATE", "LAT", "LON", "SOURCE", "SOURCE_DETAIL", "PACKAGE_ID", "Presence"]]


def _parse_erddap_csv(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text), skiprows=[1])


def _build_griddap_url(
    base_griddap: str,
    dataset_id: str,
    variable: str,
    date_str: str,
    lat: float,
    lon: float,
    half_box_deg: float,
    altitude: Optional[float] = None,
) -> str:
    lat_min, lat_max = lat - half_box_deg, lat + half_box_deg
    lon_min, lon_max = lon - half_box_deg, lon + half_box_deg
    altitude_selector = f"[({altitude})]" if altitude is not None else ""
    return (
        f"{base_griddap}/{dataset_id}.csv"
        f"?{variable}[({date_str}T12:00:00Z)]"
        f"{altitude_selector}"
        f"[({lat_min}):({lat_max})][({lon_min}):({lon_max})]"
    )


def _vprint(env_cfg: EnvConfig, msg: str) -> None:
    if env_cfg.verbose:
        print(msg, flush=True)


def _date_in_dataset_range(dataset_id: str, date_str: str) -> bool:
    bounds = DATASET_DATE_BOUNDS.get(dataset_id)
    if not bounds:
        return True
    try:
        d = pd.Timestamp(date_str)
        start = pd.Timestamp(bounds[0])
        end = pd.Timestamp(bounds[1])
        return (d >= start) and (d <= end)
    except Exception:
        return True


def _series_valid_mean(series: pd.Series, invalid: Tuple[float, ...] = (-999.0, -999)) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce")
    for v in invalid:
        s = s.mask(s == v)
    val = s.mean()
    if np.isnan(val):
        return None
    return float(val)


def _empty_env_bag() -> Dict[str, Any]:
    return {
        "SST_Celsius": None,
        "Frontal_Value": None,
        "Chlorophyll_mg_m3": None,
        "Salinity_PSU": None,
        "Water_Mass_M_WK": np.nan,
        "SST_source": None,
        "Frontal_source": None,
        "Chlorophyll_source": None,
        "Salinity_source": None,
    }


def _save_env_lookup_cache(cache_path: str, env_lookup: Dict[Tuple[str, float, float], Dict[str, Any]]) -> None:
    if not cache_path:
        return
    rows: List[dict] = []
    for (date_key, lat_key, lon_key), bag in env_lookup.items():
        rows.append(
            {
                "DATE_KEY": date_key,
                "LAT_KEY": lat_key,
                "LON_KEY": lon_key,
                "SST_Celsius": bag.get("SST_Celsius"),
                "Frontal_Value": bag.get("Frontal_Value"),
                "Chlorophyll_mg_m3": bag.get("Chlorophyll_mg_m3"),
                "Salinity_PSU": bag.get("Salinity_PSU"),
                "Water_Mass_M_WK": bag.get("Water_Mass_M_WK"),
                "SST_source": bag.get("SST_source"),
                "Frontal_source": bag.get("Frontal_source"),
                "Chlorophyll_source": bag.get("Chlorophyll_source"),
                "Salinity_source": bag.get("Salinity_source"),
            }
        )

    cache_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_df.to_csv(cache_path, index=False)


def _load_env_lookup_cache(cache_path: Optional[str]) -> Dict[Tuple[str, float, float], Dict[str, Any]]:
    if not cache_path or not os.path.exists(cache_path):
        return {}
    try:
        cache_df = pd.read_csv(cache_path)
    except Exception:
        return {}

    needed = {"DATE_KEY", "LAT_KEY", "LON_KEY"}
    if not needed.issubset(set(cache_df.columns)):
        return {}

    out: Dict[Tuple[str, float, float], Dict[str, Any]] = {}
    for _, row in cache_df.iterrows():
        key = (str(row["DATE_KEY"]), float(row["LAT_KEY"]), float(row["LON_KEY"]))
        bag = _empty_env_bag()
        for col in [
            "SST_Celsius",
            "Frontal_Value",
            "Chlorophyll_mg_m3",
            "Salinity_PSU",
            "Water_Mass_M_WK",
            "SST_source",
            "Frontal_source",
            "Chlorophyll_source",
            "Salinity_source",
        ]:
            if col in cache_df.columns:
                bag[col] = row[col]

        # normalize NaNs to None for source fields
        for s_col in ["SST_source", "Frontal_source", "Chlorophyll_source", "Salinity_source"]:
            if pd.isna(bag.get(s_col)):
                bag[s_col] = None

        out[key] = bag
    return out


def _http_get_grid(
    session: requests.Session,
    url: str,
    env_cfg: EnvConfig,
    variable: str,
) -> Tuple[Optional[pd.DataFrame], int]:
    try:
        resp = session.get(url, timeout=(env_cfg.request_timeout_connect, env_cfg.request_timeout_read))
        code = resp.status_code
        if code == 404:
            return None, code
        resp.raise_for_status()
        grid_df = _parse_erddap_csv(resp.text)
        if grid_df.empty or variable not in grid_df.columns:
            return None, code
        return grid_df, code
    except Exception:
        return None, -1


def _fetch_regional_mean(
    session: requests.Session,
    dataset_id: str,
    variable: str,
    date_str: str,
    lat: float,
    lon: float,
    env_cfg: EnvConfig,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[float]:
    if not _date_in_dataset_range(dataset_id, date_str):
        return None

    day_offsets = [0, -1, 1, -3, 3]
    # Salinity near coasts often needs a wider box to find valid ocean pixels.
    if dataset_id in ("coastwatchSMOSv662SSS1day", "coastwatchSMOSv662SSS3day"):
        half_boxes = [0.2, 0.35, 0.5, 0.75, 1.0]
    else:
        half_boxes = [0.05, 0.1, 0.2, 0.35]
    altitude = 0.0 if dataset_id == "coastwatchSMOSv662SSS1day" else None
    started = time.time()

    for day_offset in day_offsets:
        query_date = (pd.Timestamp(date_str) + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
        for half_box in half_boxes:
            if (time.time() - started) > env_cfg.max_seconds_per_var:
                return None
            url = _build_griddap_url(
                COASTWATCH_GRIDDAP, dataset_id, variable, query_date, lat, lon, half_box, altitude
            )
            if log:
                log(url)
            grid_df, code = _http_get_grid(session, url, env_cfg, variable)
            if grid_df is None:
                continue
            val = pd.to_numeric(grid_df[variable], errors="coerce").mean()
            if not np.isnan(val):
                return float(val)
    return None


def _fetch_surface_salinity_psu(
    session: requests.Session,
    date_str: str,
    lat: float,
    lon: float,
    env_cfg: EnvConfig,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Surface salinity ladder: SMOS daily, then SMOS 3-day composite."""
    # 3-day composite is often more available for coastal points.
    for ds in ("coastwatchSMOSv662SSS3day", "coastwatchSMOSv662SSS1day"):
        v = _fetch_regional_mean(
            session=session,
            dataset_id=ds,
            variable="sss",
            date_str=date_str,
            lat=lat,
            lon=lon,
            env_cfg=env_cfg,
            log=log,
        )
        if v is not None:
            return v, f"{ds}.sss"
    return None, None


def _fetch_sst_and_front(
    session: requests.Session,
    date_str: str,
    lat: float,
    lon: float,
    env_cfg: EnvConfig,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    if not _date_in_dataset_range("jplMURSST41", date_str):
        return None, None

    started = time.time()
    for day_offset in [0, -1, 1]:
        query_date = (pd.Timestamp(date_str) + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
        for half_box in [0.05, 0.1, 0.2]:
            if (time.time() - started) > env_cfg.max_seconds_per_var:
                return None, None
            url = _build_griddap_url(
                COASTWATCH_GRIDDAP, "jplMURSST41", "analysed_sst", query_date, lat, lon, half_box
            )
            if log:
                log(url)
            grid_df, _ = _http_get_grid(session, url, env_cfg, "analysed_sst")
            if grid_df is None:
                continue

            sst = pd.to_numeric(grid_df["analysed_sst"], errors="coerce").mean()
            if np.isnan(sst):
                continue

            front = None
            if {"latitude", "longitude", "analysed_sst"}.issubset(grid_df.columns):
                pivot = grid_df.pivot_table(index="latitude", columns="longitude", values="analysed_sst", aggfunc="mean")
                grid_2d = pivot.values
                if grid_2d.shape[0] >= 2 and grid_2d.shape[1] >= 2:
                    gy, gx = np.gradient(grid_2d)
                    grad = np.sqrt(gx ** 2 + gy ** 2)
                    max_grad = np.nanmax(grad)
                    if not np.isnan(max_grad):
                        front = float(max_grad)
            return float(sst), front
    return None, None


def _udel_fetch_mean(
    session: requests.Session,
    dataset_id: str,
    variable: str,
    date_str: str,
    lat: float,
    lon: float,
    env_cfg: EnvConfig,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[float]:
    """Regional mean from UDel griddap; masks ERDDAP fill values (-999)."""
    if not _date_in_dataset_range(dataset_id, date_str):
        return None

    day_offsets = [0, -1, 1, -3, 3]
    half_boxes = [0.02, 0.05, 0.1, 0.2]
    started = time.time()
    for day_offset in day_offsets:
        query_date = (pd.Timestamp(date_str) + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
        for half_box in half_boxes:
            if (time.time() - started) > env_cfg.max_seconds_per_var:
                return None
            url = _build_griddap_url(UDEL_GRIDDAP, dataset_id, variable, query_date, lat, lon, half_box)
            if log:
                log(url)
            grid_df, _ = _http_get_grid(session, url, env_cfg, variable)
            if grid_df is None:
                continue
            val = _series_valid_mean(grid_df[variable])
            if val is not None:
                return val
    return None


def _paper_udel_env_for_point(
    session: requests.Session,
    date_str: str,
    lat: float,
    lon: float,
    env_cfg: EnvConfig,
    point_idx: int,
    total_points: int,
) -> Dict[str, Any]:
    """
    Paper-style ladder on UDel ERDDAP (Ji et al. 2024): GOES SST first, then VIIRS, then MODIS;
    chlorophyll VIIRS then MODIS; frontal (M_WK_G) and water mass (M_WK) from MODIS then VIIRS.
    Salinity in the paper came from gliders — we optionally add SMOS surface salinity from CoastWatch
    so the CSV still has a salinity column for ML (not identical to the paper's glider salinity).
    """
    def log_url(u: str) -> None:
        _vprint(env_cfg, f"    {u}")

    _vprint(
        env_cfg,
        f"  ({point_idx}/{total_points}) UDel ladder @ {date_str} ({lat:.4f}, {lon:.4f})",
    )

    sst_val = None
    sst_src = None
    for label, ds, var in (
        ("GOES16 daily JPL SST (UDel)", "daily_composite_JPL_SST", "sst"),
        ("VIIRS NW Atlantic", "VIIRS_NWATL", "sst"),
        ("MODIS Aqua 3-day", "MODIS_AQUA_3_day", "sst"),
    ):
        _vprint(env_cfg, f"    SST try: {label}")
        v = _udel_fetch_mean(session, ds, var, date_str, lat, lon, env_cfg, log_url)
        if v is not None:
            sst_val = v
            sst_src = f"{ds}.{var}"
            break

    chl_val = None
    chl_src = None
    for label, ds, var in (
        ("VIIRS NW Atlantic", "VIIRS_NWATL", "chl_oc3"),
        ("MODIS Aqua 3-day", "MODIS_AQUA_3_day", "chl_oc3"),
    ):
        _vprint(env_cfg, f"    Chlorophyll try: {label}")
        v = _udel_fetch_mean(session, ds, var, date_str, lat, lon, env_cfg, log_url)
        if v is not None:
            chl_val = v
            chl_src = f"{ds}.{var}"
            break

    frontal_val = None
    wm_val = None
    front_src = None
    for label, ds in (
        ("MODIS Aqua 3-day (paper: frontal/water mass from MODIS)", "MODIS_AQUA_3_day"),
        ("VIIRS NW Atlantic (fallback; same M_WK fields)", "VIIRS_NWATL"),
    ):
        _vprint(env_cfg, f"    Frontal M_WK_G + Water mass M_WK try: {label}")
        g = _udel_fetch_mean(session, ds, "M_WK_G", date_str, lat, lon, env_cfg, log_url)
        w = _udel_fetch_mean(session, ds, "M_WK", date_str, lat, lon, env_cfg, log_url)
        if g is not None:
            frontal_val = g
            front_src = f"{ds}.M_WK_G"
        if w is not None:
            wm_val = float(w) if not np.isnan(w) else None
        if frontal_val is not None or wm_val is not None:
            break

    _vprint(env_cfg, "    Salinity (optional): CoastWatch SMOS sss — paper used glider salinity")
    sal_val, sal_src = _fetch_surface_salinity_psu(
        session=session,
        date_str=date_str,
        lat=lat,
        lon=lon,
        env_cfg=env_cfg,
        log=log_url,
    )

    return {
        "SST_Celsius": sst_val,
        "SST_source": sst_src,
        "Chlorophyll_mg_m3": chl_val,
        "Chlorophyll_source": chl_src,
        "Frontal_Value": frontal_val,
        "Frontal_source": front_src,
        "Water_Mass_M_WK": wm_val,
        "Salinity_PSU": sal_val,
        "Salinity_source": sal_src,
    }

def _coastwatch_env_for_point(
    session: requests.Session,
    date_str: str,
    lat: float,
    lon: float,
    env_cfg: EnvConfig,
) -> Dict[str, Any]:
    """CoastWatch branch extracted so we can call it from parallel workers too."""
    def log_url(u: str) -> None:
        _vprint(env_cfg, f"    {u}")

    _vprint(env_cfg, f"  CoastWatch @ {date_str} ({lat:.4f}, {lon:.4f})")

    sst, frontal = _fetch_sst_and_front(session, date_str, lat, lon, env_cfg, log_url)

    _vprint(env_cfg, "    Chlorophyll: erdMH1chla8day.chlorophyll")
    chlorophyll = _fetch_regional_mean(
        session,
        "erdMH1chla8day",
        "chlorophyll",
        date_str,
        lat,
        lon,
        env_cfg,
        log_url,
    )

    _vprint(env_cfg, "    Salinity: coastwatchSMOSv662SSS1day.sss")
    salinity, salinity_src = _fetch_surface_salinity_psu(
        session=session,
        date_str=date_str,
        lat=lat,
        lon=lon,
        env_cfg=env_cfg,
        log=log_url,
    )

    return {
        "SST_Celsius": sst,
        "Frontal_Value": frontal,
        "Chlorophyll_mg_m3": chlorophyll,
        "Salinity_PSU": salinity,
        "Water_Mass_M_WK": np.nan,
        "SST_source": "jplMURSST41.analysed_sst" if sst is not None else None,
        "Frontal_source": "gradient(jplMURSST41.analysed_sst)" if frontal is not None else None,
        "Chlorophyll_source": "erdMH1chla8day.chlorophyll" if chlorophyll is not None else None,
        "Salinity_source": salinity_src,
    }


def add_environment_features(df: pd.DataFrame, env_cfg: EnvConfig) -> pd.DataFrame:
    out = df.copy()
    out["SIGHTINGDATE"] = pd.to_datetime(out["SIGHTINGDATE"], errors="coerce")
    out = out.dropna(subset=["SIGHTINGDATE", "LAT", "LON"]).copy()

    out["DATE_KEY"] = out["SIGHTINGDATE"].dt.strftime("%Y-%m-%d")
    out["LAT_KEY"] = out["LAT"].round(3)
    out["LON_KEY"] = out["LON"].round(3)
    unique_points = out[["DATE_KEY", "LAT_KEY", "LON_KEY"]].drop_duplicates().reset_index(drop=True)

    env_lookup: Dict[Tuple[str, float, float], Dict[str, Any]] = _load_env_lookup_cache(env_cfg.env_cache_path)
    if env_cfg.enable_env_fetch and env_cfg.env_cache_path:
        _vprint(env_cfg, f"Loaded env cache keys: {len(env_lookup)} from {env_cfg.env_cache_path}")

    if env_cfg.enable_env_fetch:
        pending: List[Tuple[str, float, float]] = []
        for _, row in unique_points.iterrows():
            key = (str(row["DATE_KEY"]), float(row["LAT_KEY"]), float(row["LON_KEY"]))
            if key not in env_lookup:
                pending.append(key)

        if env_cfg.max_env_points_per_run and env_cfg.max_env_points_per_run > 0:
            pending = pending[: env_cfg.max_env_points_per_run]

        total_pending = len(pending)
        _vprint(env_cfg, f"Pending env points this run: {total_pending}")

        if total_pending > 0:
            env_lookup_lock = threading.Lock()
            completed = [0]   # mutable counter for threads

            def fetch_worker(key: Tuple[str, float, float]):
                date_key, lat_key, lon_key = key
                if env_cfg.mode == "paper":
                    worker_session = requests.Session()
                    # progress number will be (0/total) in parallel — harmless
                    bag = _paper_udel_env_for_point(
                        worker_session, date_key, lat_key, lon_key, env_cfg, 0, total_pending
                    )
                else:
                    worker_session = requests.Session()
                    bag = _coastwatch_env_for_point(worker_session, date_key, lat_key, lon_key, env_cfg)

                with env_lookup_lock:
                    env_lookup[key] = bag
                    completed[0] += 1
                    cur = completed[0]
                    if env_cfg.env_cache_path and (cur % max(1, env_cfg.checkpoint_every_points) == 0):
                        _save_env_lookup_cache(env_cfg.env_cache_path, env_lookup)
                        _vprint(env_cfg, f"  checkpoint saved at point {cur}/{total_pending} (parallel)")

                if env_cfg.polite_sleep_seconds > 0:
                    time.sleep(env_cfg.polite_sleep_seconds)

            with ThreadPoolExecutor(max_workers=4) as executor:   # ← change to 3 or 5 if you want
                futures = [executor.submit(fetch_worker, p) for p in pending]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        _vprint(env_cfg, f"WARNING: Point fetch failed: {exc}")

            # final save after all workers finish
            if env_cfg.env_cache_path:
                _save_env_lookup_cache(env_cfg.env_cache_path, env_lookup)
                _vprint(env_cfg, f"Env cache saved: {env_cfg.env_cache_path} (parallel run)")

    # === build final DataFrame (same as before) ===
    rows_out = []
    for _, row in out.iterrows():
        key = (row["DATE_KEY"], float(row["LAT_KEY"]), float(row["LON_KEY"]))
        bag = env_lookup.get(key, _empty_env_bag())
        rows_out.append(bag)

    out["SST_Celsius"] = [r["SST_Celsius"] for r in rows_out]
    out["Frontal_Value"] = [r["Frontal_Value"] for r in rows_out]
    out["Chlorophyll_mg_m3"] = [r["Chlorophyll_mg_m3"] for r in rows_out]
    out["Salinity_PSU"] = [r["Salinity_PSU"] for r in rows_out]
    out["Water_Mass_M_WK"] = [r.get("Water_Mass_M_WK") for r in rows_out]
    out["SST_env_source"] = [r.get("SST_source") for r in rows_out]
    out["Chlorophyll_env_source"] = [r.get("Chlorophyll_source") for r in rows_out]
    out["Frontal_env_source"] = [r.get("Frontal_source") for r in rows_out]
    out["Salinity_env_source"] = [r.get("Salinity_source") for r in rows_out]

    out["SST_is_observed"] = pd.Series(out["SST_Celsius"]).notna().astype(int)
    out["Frontal_is_observed"] = pd.Series(out["Frontal_Value"]).notna().astype(int)
    out["Chlorophyll_is_observed"] = pd.Series(out["Chlorophyll_mg_m3"]).notna().astype(int)
    out["Salinity_is_observed"] = pd.Series(out["Salinity_PSU"]).notna().astype(int)
    out["Water_Mass_is_observed"] = pd.Series(out["Water_Mass_M_WK"]).notna().astype(int)

    all_obs = (
        (out["SST_is_observed"] == 1)
        & (out["Frontal_is_observed"] == 1)
        & (out["Chlorophyll_is_observed"] == 1)
        & (out["Salinity_is_observed"] == 1)
    )
    if env_cfg.mode == "paper":
        all_obs = all_obs & (out["Water_Mass_is_observed"] == 1)
    out["Any_env_missing"] = (~all_obs).astype(int)

    out = out.drop(columns=["DATE_KEY", "LAT_KEY", "LON_KEY"])
    return out


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SIGHTINGDATE"] = pd.to_datetime(out["SIGHTINGDATE"], errors="coerce")
    month_to_season = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    }
    out["Season"] = out["SIGHTINGDATE"].dt.month.map(month_to_season)
    return out


def make_run_directory(project_root: str) -> str:
    """Unique folder under data/runs/ (timestamp + random hex)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = secrets.token_hex(4)
    path = os.path.join(project_root, "data", "runs", f"run_{ts}_{rid}")
    os.makedirs(path, exist_ok=True)
    return path


def build_dataset(
    output_csv: str,
    report_json: str,
    include_visual: bool,
    env_cfg: EnvConfig,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    acoustic = load_acoustic_daily_presence(ACOUSTIC_ROOT)
    visual = load_visual_positives(VISUAL_CSV) if include_visual else pd.DataFrame()

    combined = pd.concat([acoustic, visual], ignore_index=True)
    combined["SIGHTINGDATE"] = pd.to_datetime(combined["SIGHTINGDATE"], errors="coerce")
    combined = combined.dropna(subset=["SIGHTINGDATE", "LAT", "LON", "Presence"]).copy()
    combined["Presence"] = pd.to_numeric(combined["Presence"], errors="coerce").fillna(0).astype(int)

    # If two sources share exact day/lat/lon, keep max presence and join source names.
    combined["SOURCE"] = combined["SOURCE"].astype(str)
    combined = (
        combined.groupby(["SIGHTINGDATE", "LAT", "LON"], as_index=False)
        .agg(
            Presence=("Presence", "max"),
            SOURCE=("SOURCE", lambda s: "|".join(sorted(set(s)))),
            SOURCE_DETAIL=("SOURCE_DETAIL", lambda s: "|".join(sorted(set(str(v) for v in s if pd.notna(v))))),
            PACKAGE_ID=("PACKAGE_ID", lambda s: "|".join(sorted(set(str(v) for v in s if pd.notna(v))))),
        )
    )

    if max_rows and max_rows > 0:
        combined = combined.sort_values("SIGHTINGDATE").head(max_rows).copy()

    combined = add_season(combined)
    combined = add_environment_features(combined, env_cfg=env_cfg)
    combined = combined.sort_values(["SIGHTINGDATE", "LAT", "LON"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined.to_csv(output_csv, index=False)

    env_counts = {
        "SST_Celsius": int(combined["SST_is_observed"].sum()),
        "Frontal_Value": int(combined["Frontal_is_observed"].sum()),
        "Chlorophyll_mg_m3": int(combined["Chlorophyll_is_observed"].sum()),
        "Salinity_PSU": int(combined["Salinity_is_observed"].sum()),
    }
    if env_cfg.mode == "paper" and "Water_Mass_is_observed" in combined.columns:
        env_counts["Water_Mass_M_WK"] = int(combined["Water_Mass_is_observed"].sum())

    if env_cfg.mode == "paper":
        erddap_sources = {
            "server_primary": UDEL_GRIDDAP,
            "SST_Celsius": "Ladder: daily_composite_JPL_SST.sst → VIIRS_NWATL.sst → MODIS_AQUA_3_day.sst",
            "Chlorophyll_mg_m3": "Ladder: VIIRS_NWATL.chl_oc3 → MODIS_AQUA_3_day.chl_oc3",
            "Frontal_Value": "Oliver & Irwin style M_WK_G: MODIS_AQUA_3_day → VIIRS_NWATL fallback",
            "Water_Mass_M_WK": "M_WK: MODIS_AQUA_3_day → VIIRS_NWATL fallback",
            "Salinity_PSU": "Extra: coastwatchSMOSv662SSS1day.sss (paper used glider salinity, not SMOS)",
        }
        disclaimer = (
            "Mirrors the paper's satellite *product ladder* on UDel ERDDAP. Does not reproduce kriging/RF "
            "gap filling or SMOTE. Glider profiles (temp/salinity/oxygen/depth) are not fetched here."
        )
        notes_extra = [
            "Paper frontal/water-mass variables align with MODIS/VIIRS M_WK_G and M_WK on basin.ceoe.udel.edu.",
            "Any_env_missing in paper mode requires SST, Frontal, Chlorophyll, Salinity, and Water_Mass present.",
        ]
    else:
        erddap_sources = {
            "server": COASTWATCH_GRIDDAP,
            "SST_Celsius": {"dataset": "jplMURSST41", "variable": "analysed_sst"},
            "Frontal_Value": {"derived_from": "jplMURSST41 analysed_sst grid", "method": "max_gradient_magnitude"},
            "Chlorophyll_mg_m3": {"dataset": "erdMH1chla8day", "variable": "chlorophyll"},
            "Salinity_PSU": {"dataset": "coastwatchSMOSv662SSS1day", "variable": "sss"},
        }
        disclaimer = (
            "NOAA CoastWatch griddap only (not UDel). Real measured composites; no imputed env values."
        )
        notes_extra = [
            "Frontal_Value is max |∇SST| on MUR subgrid (not UDel M_WK_G).",
            "Any_env_missing is 1 if any of the four main env columns is missing.",
        ]

    report = {
        "mode": env_cfg.mode,
        "output_csv": output_csv,
        "total_rows": int(len(combined)),
        "positive_rows": int((combined["Presence"] == 1).sum()),
        "negative_rows": int((combined["Presence"] == 0).sum()),
        "unique_points": int(combined[["LAT", "LON"]].drop_duplicates().shape[0]),
        "sources": sorted(set("|".join(combined["SOURCE"].dropna().astype(str)).split("|"))),
        "env_observed_counts": env_counts,
        "erddap_sources": erddap_sources,
        "paper_disclaimer": disclaimer,
        "notes": [
            "No KNN, no SMOTE, no interpolation in this dataset builder.",
            "Rows with unavailable environmental API values are kept with NaN and *_is_observed flags.",
        ]
        + notes_extra,
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return combined, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a real-data whale training CSV without synthetic feature imputation.")
    parser.add_argument(
        "--mode",
        choices=("coastwatch", "paper"),
        default="coastwatch",
        help="coastwatch: NOAA CoastWatch ERDDAP only. paper: UDel GOES→VIIRS→MODIS ladder + M_WK fields.",
    )
    parser.add_argument("--output", default=None, help="Output CSV path (defaults by mode if omitted).")
    parser.add_argument("--report", default=None, help="Output JSON report path.")
    parser.add_argument(
        "--random-run-dir",
        action="store_true",
        help="Write outputs under data/runs/run_<timestamp>_<random>/ (ignores default processed paths unless --output set).",
    )
    parser.add_argument("--no-visual", action="store_true", help="Exclude visual sightings file.")
    parser.add_argument("--no-env", action="store_true", help="Skip environmental API fetching.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick test runs.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep between environmental requests.")
    parser.add_argument("--connect-timeout", type=int, default=5, help="HTTP connect timeout.")
    parser.add_argument("--read-timeout", type=int, default=12, help="HTTP read timeout.")
    parser.add_argument("--max-seconds-per-var", type=int, default=25, help="Time budget per variable fetch.")
    parser.add_argument("--env-cache", default=None, help="Path to persistent env lookup cache CSV.")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Save env cache every N query points.")
    parser.add_argument(
        "--max-env-points-per-run",
        type=int,
        default=None,
        help="Optional cap on new environment query points for resumable batch runs.",
    )
    parser.add_argument("--quiet", action="store_true", help="Less terminal output (no per-URL lines).")
    return parser.parse_args()


def _resolve_output_paths(args: argparse.Namespace) -> Tuple[str, str, Optional[str]]:
    """Returns (output_csv, report_json, run_dir_or_none)."""
    run_dir: Optional[str] = None
    suf = "coastwatch" if args.mode == "coastwatch" else "paper"

    if args.random_run_dir:
        run_dir = make_run_directory(PROJECT_ROOT)
        output_csv = args.output or os.path.join(run_dir, f"whale_dataset_{suf}.csv")
        report_json = args.report or os.path.join(run_dir, f"report_{suf}.json")
        return output_csv, report_json, run_dir

    if args.mode == "coastwatch":
        output_csv = args.output or os.path.join(PROJECT_ROOT, "data", "processed", "REAL_Whale_coastwatch.csv")
        report_json = args.report or os.path.join(PROJECT_ROOT, "data", "processed", "report_coastwatch.json")
    else:
        output_csv = args.output or os.path.join(PROJECT_ROOT, "data", "processed", "REAL_Whale_paper_style.csv")
        report_json = args.report or os.path.join(PROJECT_ROOT, "data", "processed", "report_paper_style.json")
    return output_csv, report_json, run_dir


if __name__ == "__main__":
    args = parse_args()
    output_csv, report_json, run_dir = _resolve_output_paths(args)
    default_cache = os.path.join(PROJECT_ROOT, "data", "processed", f"env_lookup_cache_{args.mode}.csv")
    cfg = EnvConfig(
        request_timeout_connect=args.connect_timeout,
        request_timeout_read=args.read_timeout,
        polite_sleep_seconds=args.sleep,
        max_seconds_per_var=args.max_seconds_per_var,
        enable_env_fetch=not args.no_env,
        mode=args.mode,
        verbose=not args.quiet,
        env_cache_path=args.env_cache or default_cache,
        checkpoint_every_points=args.checkpoint_every,
        max_env_points_per_run=args.max_env_points_per_run,
    )
    _vprint(cfg, f"Mode: {args.mode} | output: {output_csv}")
    if run_dir:
        _vprint(cfg, f"Run directory: {run_dir}")

    dataset, report = build_dataset(
        output_csv=output_csv,
        report_json=report_json,
        include_visual=not args.no_visual,
        env_cfg=cfg,
        max_rows=args.max_rows,
    )
    print(f"Saved: {output_csv}")
    print(f"Rows: {len(dataset)} | positives: {(dataset['Presence'] == 1).sum()} | negatives: {(dataset['Presence'] == 0).sum()}")
    print(f"Report: {report_json}")
