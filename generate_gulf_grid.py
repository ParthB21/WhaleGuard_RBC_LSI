"""
==========================================================================
 WhaleGuard — Gulf of St. Lawrence Grid Feature Extraction
==========================================================================
 Generates a regular 0.25° spatial grid over the Gulf of St. Lawrence
 (45°N–51°N, -71°W to -56°W) and extracts environmental features for
 every ocean grid cell across 2002–2018.

 Architecture mirrors the original ETL pipeline:
   • Slab-based OPeNDAP fetches (pipeline.py architecture)
   • SST gradient computation (phase3_feature_engineering.py)
   • Spatial features: slope, dist-to-shore, dist-to-shelf
     (patch_slope_features.py logic)

 Outputs
 -------
   data/processed/Gulf_St_Lawrence_Grid_Features.csv
   data/processed/Gulf_Grid_checkpoint.csv   (resumable checkpoint)
   logs/grid_generation.log

 Usage
 -----
   python generate_gulf_grid.py

 The script is fully resumable: if interrupted, re-running it will
 skip already-completed months and continue from where it stopped.
==========================================================================
"""

import logging
import sys
import time
import warnings
from math import cos, radians
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", message=".*SerializationWarning.*")
warnings.filterwarnings("ignore", category=FutureWarning)


# =========================================================================
#  Configuration
# =========================================================================

# -- Grid bounds (Gulf of St. Lawrence) -----------------------------------
LAT_MIN, LAT_MAX = 45.0, 51.0      # °N
LON_MIN, LON_MAX = -71.0, -56.0    # °W
GRID_RES         = 0.25            # degrees per cell

# -- Temporal scope -------------------------------------------------------
YEAR_START, YEAR_END = 2002, 2018

# -- Paths ----------------------------------------------------------------
OUTPUT_CSV     = Path("data/processed/Gulf_St_Lawrence_Grid_Features.csv")
CHECKPOINT_CSV = Path("data/processed/Gulf_Grid_checkpoint.csv")
LOG_DIR        = Path("logs")

# -- ERDDAP OPeNDAP endpoints (same base as pipeline.py) ------------------
ERDDAP_BASE       = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
SST_URL           = f"{ERDDAP_BASE}/jplMURSST41"
CHLOROPHYLL_URL   = f"{ERDDAP_BASE}/erdMH1chlamday"
SALINITY_URL      = f"{ERDDAP_BASE}/erdSoda331oceanmday_LonPM180"
ETOPO_URL         = f"{ERDDAP_BASE}/etopo180"   # pm180 convention (-180 to 180)

# -- Physics & algorithm constants ----------------------------------------
SLAB_BUFFER_DEG   = 1.0            # pad around bounding box for slab downloads
SST_PIXEL_SIZE_DEG = 0.01          # MUR SST resolution (for gradient km conversion)
ETOPO_PIXEL_DEG   = 1.0 / 60.0    # ETOPO1 1 arc-minute resolution
FRONT_THRESHOLD   = 0.035          # °C/km — Tao et al., 2025
SHELF_BREAK_DEPTH = -200           # metres (200m isobath)
SHELF_BREAK_TOL   = 50             # metres (±)
EARTH_RADIUS_KM   = 6371.0
MAX_RETRIES       = 3


# =========================================================================
#  Logging (identical format to pipeline.py)
# =========================================================================

def _setup_logging() -> logging.Logger:
    """Dual console (INFO+) and file (DEBUG+) logging with timestamps."""
    logger = logging.getLogger("WhaleGuard_Grid")
    if logger.handlers:
        return logger  # already configured if imported elsewhere
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    LOG_DIR.mkdir(exist_ok=True)
    fh = logging.FileHandler(LOG_DIR / "grid_generation.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = _setup_logging()


# =========================================================================
#  Utilities
# =========================================================================

def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised great-circle distance in km."""
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _fetch_slab_with_retry(
    da: xr.DataArray,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    time_key=None,
    label: str = "slab",
) -> "xr.DataArray | None":
    """
    Fetch a spatial slab from an open DataArray with exponential backoff.
    Mirrors _extract_slab_group() in pipeline.py.
    """
    for attempt in range(MAX_RETRIES):
        try:
            if time_key is not None:
                slab = da.sel(time=time_key, method="nearest").sel(
                    latitude=slice(lat_min, lat_max),
                    longitude=slice(lon_min, lon_max),
                )
            else:
                slab = da.sel(
                    latitude=slice(lat_min, lat_max),
                    longitude=slice(lon_min, lon_max),
                )
            return slab.load()
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                sleep_sec = (2 ** attempt) * 5  # 5s, 10s, 20s
                log.debug(
                    f"    [{label}] Attempt {attempt + 1} failed: {exc}. "
                    f"Retrying in {sleep_sec}s…"
                )
                time.sleep(sleep_sec)
            else:
                log.warning(
                    f"    [{label}] Failed after {MAX_RETRIES} attempts: {exc}"
                )
                return None


def _extract_at_points(
    slab: xr.DataArray,
    lats: np.ndarray,
    lons: np.ndarray,
    use_nearest: bool = True,
) -> np.ndarray:
    """
    Extract slab values at arbitrary (lat, lon) coordinates.
    Tries bilinear interpolation first for coarse-res datasets,
    falls back to nearest-neighbour on error.
    """
    coords = {
        "latitude":  xr.DataArray(lats, dims="points"),
        "longitude": xr.DataArray(lons, dims="points"),
    }
    if use_nearest:
        try:
            return slab.sel(coords, method="nearest").values
        except Exception:
            return np.full(len(lats), np.nan)
    else:
        try:
            return slab.interp(coords, method="linear").values
        except Exception:
            try:
                return slab.sel(coords, method="nearest").values
            except Exception:
                return np.full(len(lats), np.nan)


# =========================================================================
#  Stage 1 — Build ocean-only grid
# =========================================================================

def build_ocean_grid() -> "tuple[np.ndarray, np.ndarray, xr.DataArray]":
    """
    Create the 0.25° lat/lon meshgrid, download ETOPO1 once, and return
    only the ocean grid points (altitude < 0 m).

    Returns
    -------
    ocean_lats : 1-D array of ocean-cell latitudes
    ocean_lons : 1-D array of ocean-cell longitudes
    etopo_slab : loaded 2-D DataArray (kept for Stage 2 feature computation)
    """
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║            STAGE 1 — GRID DEFINITION & OCEAN MASK          ║")
    log.info("╚══════════════════════════════════════════════════════════════╝\n")

    # Build raw meshgrid
    lat_vals = np.arange(LAT_MIN, LAT_MAX + GRID_RES / 2, GRID_RES)
    lon_vals = np.arange(LON_MIN, LON_MAX + GRID_RES / 2, GRID_RES)
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    all_lats = lat_grid.flatten()
    all_lons = lon_grid.flatten()
    n_raw = len(all_lats)

    log.info(f"  Raw grid: {len(lat_vals)} lat × {len(lon_vals)} lon = {n_raw:,} points")
    log.info(
        f"  Lat: {lat_vals[0]:.2f}°N → {lat_vals[-1]:.2f}°N  "
        f"Lon: {lon_vals[0]:.2f}°W → {lon_vals[-1]:.2f}°W"
    )
    cell_km_lat = GRID_RES * 111.32
    cell_km_lon = GRID_RES * 111.32 * cos(radians(48))
    log.info(f"  Cell size:  ~{cell_km_lat:.0f} km (N-S) × {cell_km_lon:.0f} km (E-W) at 48°N\n")

    # Download ETOPO1 slab (extended by SLAB_BUFFER_DEG for gradient edge effects)
    slab_lat_min = LAT_MIN - SLAB_BUFFER_DEG
    slab_lat_max = LAT_MAX + SLAB_BUFFER_DEG
    slab_lon_min = LON_MIN - SLAB_BUFFER_DEG
    slab_lon_max = LON_MAX + SLAB_BUFFER_DEG

    log.info(
        f"  Downloading ETOPO1 slab "
        f"[{slab_lat_min}°–{slab_lat_max}°N, {slab_lon_min}°–{slab_lon_max}°W]…"
    )
    t0 = time.time()
    try:
        ds_etopo = xr.open_dataset(ETOPO_URL, engine="netcdf4")
        log.info(f"    ✓ Connected — dims: {dict(ds_etopo.dims)}")
    except Exception as exc:
        log.error(f"    ✗ ETOPO1 connection failed: {exc}")
        raise

    etopo_slab = _fetch_slab_with_retry(
        ds_etopo["altitude"],
        slab_lat_min, slab_lat_max, slab_lon_min, slab_lon_max,
        label="ETOPO1",
    )
    if etopo_slab is None:
        raise RuntimeError("ETOPO1 slab download failed after all retries")

    ds_etopo.close()
    log.info(f"    ✓ ETOPO1 loaded in {time.time() - t0:.1f}s  (shape: {etopo_slab.shape})\n")

    # Ocean mask: keep points where seafloor depth < 0 m
    altitude_at_grid = _extract_at_points(etopo_slab, all_lats, all_lons, use_nearest=True)
    ocean_mask = altitude_at_grid < 0
    ocean_lats = all_lats[ocean_mask]
    ocean_lons = all_lons[ocean_mask]
    n_ocean = len(ocean_lats)
    n_land  = n_raw - n_ocean

    log.info(f"  Ocean mask applied:")
    log.info(f"    Ocean cells: {n_ocean:,}  ({n_ocean / n_raw * 100:.1f}%)")
    log.info(f"    Land cells:  {n_land:,}  (discarded)\n")

    return ocean_lats, ocean_lons, etopo_slab


# =========================================================================
#  Stage 2 — Static spatial features (computed once from ETOPO1)
# =========================================================================

def compute_static_features(
    ocean_lats: np.ndarray,
    ocean_lons: np.ndarray,
    etopo_slab: xr.DataArray,
) -> dict:
    """
    Compute Bathymetry, Bathy_Slope, Dist_to_Shore_km, Dist_to_Shelf_km
    from the pre-loaded ETOPO1 slab.

    Logic replicates patch_slope_features.py exactly:
      • Slope via np.gradient with km-scaled pixel spacing
      • Distance via cKDTree (approximate) + haversine (precise)
    """
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║        STAGE 2 — STATIC SPATIAL FEATURES (ETOPO1)         ║")
    log.info("╚══════════════════════════════════════════════════════════════╝\n")

    t0 = time.time()

    depth_values = etopo_slab.values          # shape: (n_lat, n_lon)
    grid_lats    = etopo_slab.latitude.values  # 1-D
    grid_lons    = etopo_slab.longitude.values # 1-D
    mean_lat     = float(np.mean(ocean_lats))
    cos_c        = cos(radians(mean_lat))

    # -- Bathymetry at ocean grid points (altitude = depth in metres) ------
    log.info("  [2a] Bathymetry at ocean grid points…")
    bathymetry = _extract_at_points(etopo_slab, ocean_lats, ocean_lons, use_nearest=True)
    log.info(
        f"    ✓ Range: [{np.nanmin(bathymetry):.0f} m, {np.nanmax(bathymetry):.0f} m]"
        f"  |  NaN: {np.isnan(bathymetry).sum()}\n"
    )

    # -- Bathy_Slope (m/km) ------------------------------------------------
    # Pixel spacing in km using ETOPO1's 1 arc-minute resolution
    log.info("  [2b] Bathy_Slope — central-difference gradient (m/km)…")
    dy_km = ETOPO_PIXEL_DEG * 111.32                  # latitude spacing (constant)
    dx_km = ETOPO_PIXEL_DEG * 111.32 * cos_c          # longitude spacing (lat-corrected)
    grad_y, grad_x = np.gradient(depth_values, dy_km, dx_km)
    slope_field_2d = np.sqrt(grad_x**2 + grad_y**2)

    slope_da = xr.DataArray(
        slope_field_2d,
        coords=etopo_slab.coords,
        dims=etopo_slab.dims,
        name="Bathy_Slope",
    )
    bathy_slope = _extract_at_points(slope_da, ocean_lats, ocean_lons, use_nearest=True)
    log.info(
        f"    ✓ Range: [{np.nanmin(bathy_slope):.3f}, {np.nanmax(bathy_slope):.3f}] m/km"
        f"  |  NaN: {np.isnan(bathy_slope).sum()}\n"
    )

    # -- Coordinate meshgrid for land/shelf detection ----------------------
    lon_mesh, lat_mesh = np.meshgrid(grid_lons, grid_lats)

    # -- Dist_to_Shore_km (nearest land cell) ------------------------------
    log.info("  [2c] Dist_to_Shore_km — cKDTree on land cells + haversine…")
    land_mask  = depth_values >= 0
    land_lats  = lat_mesh[land_mask]
    land_lons  = lon_mesh[land_mask]
    n_land_cells = int(land_lats.size)
    log.info(f"    Land cells in slab: {n_land_cells:,}")

    if n_land_cells == 0:
        log.warning("    No land cells found in ETOPO1 slab — Dist_to_Shore_km will be NaN")
        dist_shore = np.full(len(ocean_lats), np.nan)
    else:
        # Scale coordinates to approximate km for Euclidean tree search
        scaled_land  = np.column_stack([land_lats  * 111.32, land_lons  * 111.32 * cos_c])
        scaled_ocean = np.column_stack([ocean_lats * 111.32, ocean_lons * 111.32 * cos_c])
        tree_land    = cKDTree(scaled_land)
        _, idx       = tree_land.query(scaled_ocean, k=1)
        # Precise haversine distance to the identified nearest land cell
        dist_shore = _haversine_km(ocean_lats, ocean_lons, land_lats[idx], land_lons[idx])

    log.info(
        f"    ✓ Range: [{np.nanmin(dist_shore):.1f}, {np.nanmax(dist_shore):.1f}] km"
        f"  |  NaN: {np.isnan(dist_shore).sum()}\n"
    )

    # -- Dist_to_Shelf_km (nearest 200m isobath cell) ----------------------
    log.info(
        f"  [2d] Dist_to_Shelf_km — cKDTree on ±{SHELF_BREAK_TOL}m "
        f"of {SHELF_BREAK_DEPTH}m isobath…"
    )
    shelf_mask  = (
        (depth_values >= SHELF_BREAK_DEPTH - SHELF_BREAK_TOL) &
        (depth_values <= SHELF_BREAK_DEPTH + SHELF_BREAK_TOL)
    )
    shelf_lats  = lat_mesh[shelf_mask]
    shelf_lons  = lon_mesh[shelf_mask]
    n_shelf     = int(shelf_lats.size)
    log.info(f"    Shelf-break cells: {n_shelf:,}")

    if n_shelf == 0:
        log.warning("    No shelf-break cells found — Dist_to_Shelf_km will be NaN")
        dist_shelf = np.full(len(ocean_lats), np.nan)
    else:
        scaled_shelf = np.column_stack([shelf_lats * 111.32, shelf_lons * 111.32 * cos_c])
        tree_shelf   = cKDTree(scaled_shelf)
        _, idx2      = tree_shelf.query(scaled_ocean, k=1)
        dist_shelf   = _haversine_km(
            ocean_lats, ocean_lons, shelf_lats[idx2], shelf_lons[idx2]
        )

    log.info(
        f"    ✓ Range: [{np.nanmin(dist_shelf):.1f}, {np.nanmax(dist_shelf):.1f}] km"
        f"  |  NaN: {np.isnan(dist_shelf).sum()}\n"
    )

    log.info(f"  Stage 2 complete in {time.time() - t0:.1f}s\n")

    return {
        "Bathymetry":       bathymetry,
        "Bathy_Slope":      bathy_slope,
        "Dist_to_Shore_km": dist_shore,
        "Dist_to_Shelf_km": dist_shelf,
    }


# =========================================================================
#  SST gradient (identical to phase3_feature_engineering.py)
# =========================================================================

def _compute_gradient_magnitude(sst_slab: xr.DataArray) -> xr.DataArray:
    """
    Spatial gradient magnitude of SST in °C/km using central differences.
    Accounts for latitude-dependent longitude spacing.
    """
    sst_values = sst_slab.values
    lat_mean    = float(np.mean(sst_slab.latitude.values))
    cos_lat     = cos(radians(lat_mean))

    dy_km = SST_PIXEL_SIZE_DEG * 111.32           # latitude direction (constant)
    dx_km = SST_PIXEL_SIZE_DEG * 111.32 * cos_lat # longitude direction (lat-corrected)

    grad_y, grad_x = np.gradient(sst_values, dy_km, dx_km)
    magnitude      = np.sqrt(grad_x**2 + grad_y**2)

    return xr.DataArray(magnitude, coords=sst_slab.coords, dims=sst_slab.dims)


# =========================================================================
#  Stage 3 — Single-month temporal extraction
# =========================================================================

def extract_one_month(
    year: int,
    month: int,
    ocean_lats: np.ndarray,
    ocean_lons: np.ndarray,
    ds_sst,    # xr.Dataset or None
    ds_chl,    # xr.Dataset or None
    ds_sal,    # xr.Dataset or None
) -> dict:
    """
    Fetch SST, Chlorophyll, and Salinity for a single year+month,
    compute SST_Gradient and Is_Thermal_Front, and return a dict of arrays.
    All arrays have length == len(ocean_lats).
    """
    label  = f"{year}-{month:02d}"
    n_pts  = len(ocean_lats)
    t0     = time.time()

    # day-16 at 09:00 UTC: aligns with monthly product mid-point convention
    date_key = pd.Timestamp(year, month, 16, 9, 0)

    # Slab bounds = Gulf bounds + buffer
    slab_lat_min = LAT_MIN - SLAB_BUFFER_DEG
    slab_lat_max = LAT_MAX + SLAB_BUFFER_DEG
    slab_lon_min = LON_MIN - SLAB_BUFFER_DEG
    slab_lon_max = LON_MAX + SLAB_BUFFER_DEG

    # ── SST + Gradient + Thermal Front ────────────────────────────────────
    if ds_sst is not None:
        log.info(f"  [{label}] Fetching SST…")
        sst_slab = _fetch_slab_with_retry(
            ds_sst["analysed_sst"],
            slab_lat_min, slab_lat_max, slab_lon_min, slab_lon_max,
            time_key=date_key, label=f"{label}/SST",
        )
    else:
        sst_slab = None

    if sst_slab is not None and sst_slab.size >= 4:
        sst_vals      = _extract_at_points(sst_slab, ocean_lats, ocean_lons, use_nearest=True)
        gradient_field = _compute_gradient_magnitude(sst_slab)
        gradient_vals  = _extract_at_points(gradient_field, ocean_lats, ocean_lons, use_nearest=True)
        front_vals     = (gradient_vals > FRONT_THRESHOLD).astype(int)
        n_sst_v = int(np.count_nonzero(~np.isnan(sst_vals)))
        log.debug(f"    [{label}] SST: {n_sst_v}/{n_pts} valid  |  "
                  f"Gradient computed  |  Fronts: {int(front_vals.sum())}")
    else:
        if ds_sst is not None:
            log.warning(f"  [{label}] SST slab unavailable — filling NaN")
        sst_vals      = np.full(n_pts, np.nan)
        gradient_vals = np.full(n_pts, np.nan)
        front_vals    = np.zeros(n_pts, dtype=int)

    # ── Chlorophyll ───────────────────────────────────────────────────────
    if ds_chl is not None:
        log.info(f"  [{label}] Fetching Chlorophyll…")
        chl_slab = _fetch_slab_with_retry(
            ds_chl["chlorophyll"],
            slab_lat_min, slab_lat_max, slab_lon_min, slab_lon_max,
            time_key=date_key, label=f"{label}/Chl",
        )
    else:
        chl_slab = None

    if chl_slab is not None:
        chl_vals = _extract_at_points(chl_slab, ocean_lats, ocean_lons, use_nearest=False)
        n_chl_v  = int(np.count_nonzero(~np.isnan(chl_vals)))
        log.debug(f"    [{label}] Chl: {n_chl_v}/{n_pts} valid")
    else:
        if ds_chl is not None:
            log.warning(f"  [{label}] Chlorophyll slab unavailable — filling NaN")
        chl_vals = np.full(n_pts, np.nan)

    # ── Salinity ──────────────────────────────────────────────────────────
    if ds_sal is not None:
        log.info(f"  [{label}] Fetching Salinity…")
        try:
            sal_da = ds_sal["salt"].isel(depth=0)
        except Exception as exc:
            log.warning(f"  [{label}] Could not select Salinity surface layer: {exc}")
            sal_da = None

        if sal_da is not None:
            sal_slab = _fetch_slab_with_retry(
                sal_da,
                slab_lat_min, slab_lat_max, slab_lon_min, slab_lon_max,
                time_key=date_key, label=f"{label}/Sal",
            )
        else:
            sal_slab = None
    else:
        sal_slab = None

    if sal_slab is not None:
        sal_vals = _extract_at_points(sal_slab, ocean_lats, ocean_lons, use_nearest=False)
        n_sal_v  = int(np.count_nonzero(~np.isnan(sal_vals)))
        log.debug(f"    [{label}] Sal: {n_sal_v}/{n_pts} valid")
    else:
        if ds_sal is not None:
            log.warning(f"  [{label}] Salinity slab unavailable — filling NaN")
        sal_vals = np.full(n_pts, np.nan)

    elapsed = time.time() - t0
    log.debug(f"    [{label}] Month extraction complete in {elapsed:.1f}s")

    return {
        "SST":              sst_vals,
        "Chlorophyll":      chl_vals,
        "Salinity":         sal_vals,
        "SST_Gradient":     gradient_vals,
        "Is_Thermal_Front": front_vals,
    }


# =========================================================================
#  Checkpoint helpers
# =========================================================================

def _load_checkpoint() -> "tuple[pd.DataFrame, set]":
    """
    Load existing checkpoint CSV and return:
      - the DataFrame with previously processed rows
      - a set of (year, month) tuples already completed
    """
    if not CHECKPOINT_CSV.exists():
        return pd.DataFrame(), set()

    df = pd.read_csv(CHECKPOINT_CSV)
    completed = set(zip(df["Year"].astype(int), df["Month"].astype(int)))
    yrs = sorted(df["Year"].unique().tolist())
    log.info(
        f"  Checkpoint: {len(df):,} rows already saved, "
        f"years {yrs[0]}–{yrs[-1]}, "
        f"{len(completed)} months complete"
    )
    return df, completed


def _append_checkpoint(rows: list) -> None:
    """Append a list of row dicts to the checkpoint CSV (creates if absent)."""
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    CHECKPOINT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_CSV.exists():
        df_new.to_csv(CHECKPOINT_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(CHECKPOINT_CSV, index=False)


# =========================================================================
#  Main orchestrator
# =========================================================================

def main() -> None:
    """Execute the full Gulf of St. Lawrence grid feature extraction."""
    pipeline_start = time.time()

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  WhaleGuard — Gulf of St. Lawrence Grid Feature Extraction  ║")
    log.info("║  Target: North Atlantic Right Whale (Eubalaena glacialis)   ║")
    log.info("╚══════════════════════════════════════════════════════════════╝\n")
    log.info(f"  Gulf bounds:  {LAT_MIN}°–{LAT_MAX}°N,  {LON_MIN}°–{LON_MAX}°W")
    log.info(f"  Resolution:   {GRID_RES}°  (~{GRID_RES*111.32*cos(radians(48)):.0f} × "
             f"{GRID_RES*111.32:.0f} km at 48°N)")
    total_months = (YEAR_END - YEAR_START + 1) * 12
    log.info(f"  Temporal:     {YEAR_START}–{YEAR_END}  ({total_months} months)\n")

    # ── Stage 1: Grid + ocean mask ─────────────────────────────────────────
    ocean_lats, ocean_lons, etopo_slab = build_ocean_grid()
    n_pts = len(ocean_lats)

    # ── Stage 2: Static features ───────────────────────────────────────────
    static = compute_static_features(ocean_lats, ocean_lons, etopo_slab)
    del etopo_slab  # free memory

    # ── Checkpoint: determine which months are already done ────────────────
    log.info("─── Checkpoint Status ───────────────────────────────────────────")
    df_ckpt, completed_months = _load_checkpoint()
    n_already = len(completed_months)
    n_remaining = total_months - n_already
    if n_already:
        log.info(f"  Resuming from checkpoint — {n_remaining} months remaining\n")
    else:
        log.info("  No checkpoint found — starting fresh\n")

    # ── Open remote datasets once ──────────────────────────────────────────
    log.info("─── Connecting to ERDDAP Datasets ───────────────────────────────")
    log.info(f"  SST:         {SST_URL}")
    log.info(f"  Chlorophyll: {CHLOROPHYLL_URL}")
    log.info(f"  Salinity:    {SALINITY_URL}\n")

    try:
        ds_sst = xr.open_dataset(SST_URL, engine="netcdf4")
        log.info(f"  ✓ SST connected         dims: {dict(ds_sst.dims)}")
    except Exception as exc:
        log.error(f"  ✗ SST connection failed: {exc}")
        raise

    try:
        ds_chl = xr.open_dataset(CHLOROPHYLL_URL, engine="netcdf4")
        log.info(f"  ✓ Chlorophyll connected  dims: {dict(ds_chl.dims)}")
    except Exception as exc:
        log.warning(f"  ✗ Chlorophyll connection failed: {exc} — will be NaN")
        ds_chl = None

    try:
        ds_sal = xr.open_dataset(SALINITY_URL, engine="netcdf4")
        log.info(f"  ✓ Salinity connected     dims: {dict(ds_sal.dims)}")
        log.info(
            "    Note: Salinity coverage ends 2015; "
            "years 2016–2018 will be NaN (handled by model imputer)"
        )
    except Exception as exc:
        log.warning(f"  ✗ Salinity connection failed: {exc} — will be NaN")
        ds_sal = None

    log.info("")

    # ── Stage 3: Temporal loop ─────────────────────────────────────────────
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║         STAGE 3 — TEMPORAL FEATURE EXTRACTION              ║")
    log.info(f"║         {total_months} months × {n_pts:,} ocean cells              ║")
    log.info("╚══════════════════════════════════════════════════════════════╝\n")

    month_idx  = 0
    all_new_rows: list = []

    for year in range(YEAR_START, YEAR_END + 1):
        year_rows: list = []

        for month in range(1, 13):
            month_idx += 1
            label = f"{year}-{month:02d}"

            if (year, month) in completed_months:
                log.info(
                    f"  [{label}] Skipping (already in checkpoint) "
                    f"| {month_idx}/{total_months}"
                )
                continue

            log.info(
                f"  ─── [{label}] Processing "
                f"| {month_idx}/{total_months} "
                f"({month_idx / total_months * 100:.1f}%)"
            )

            dynamic = extract_one_month(
                year, month, ocean_lats, ocean_lons,
                ds_sst, ds_chl, ds_sal,
            )

            # Build one row dict per ocean grid point
            for i in range(n_pts):
                year_rows.append({
                    "Year":             year,
                    "Month":            month,
                    "Lat":              round(float(ocean_lats[i]), 4),
                    "Lon":              round(float(ocean_lons[i]), 4),
                    "SST":              dynamic["SST"][i],
                    "Chlorophyll":      dynamic["Chlorophyll"][i],
                    "Salinity":         dynamic["Salinity"][i],
                    "Bathymetry":       static["Bathymetry"][i],
                    "SST_Gradient":     dynamic["SST_Gradient"][i],
                    "Is_Thermal_Front": int(dynamic["Is_Thermal_Front"][i]),
                    "Bathy_Slope":      static["Bathy_Slope"][i],
                    "Dist_to_Shore_km": static["Dist_to_Shore_km"][i],
                    "Dist_to_Shelf_km": static["Dist_to_Shelf_km"][i],
                })

            # Per-month summary log
            n_sst_v = int(np.count_nonzero(~np.isnan(dynamic["SST"])))
            n_chl_v = int(np.count_nonzero(~np.isnan(dynamic["Chlorophyll"])))
            n_sal_v = int(np.count_nonzero(~np.isnan(dynamic["Salinity"])))
            n_front = int(dynamic["Is_Thermal_Front"].sum())
            log.info(
                f"  [{label}] Written {n_pts} rows | "
                f"SST {n_sst_v}/{n_pts} | "
                f"Chl {n_chl_v}/{n_pts} | "
                f"Sal {n_sal_v}/{n_pts} | "
                f"Fronts {n_front} | "
                f"{month_idx}/{total_months} ({month_idx / total_months * 100:.1f}%)"
            )

        # Checkpoint: save after completing each full year
        if year_rows:
            _append_checkpoint(year_rows)
            all_new_rows.extend(year_rows)
            n_ckpt_total = len(df_ckpt) + len(all_new_rows)
            elapsed_min  = (time.time() - pipeline_start) / 60
            remaining_months = total_months - month_idx
            rate = month_idx / elapsed_min if elapsed_min > 0 else 1
            eta_min = remaining_months / rate if rate > 0 else 0
            log.info(
                f"\n  ✓ Year {year} checkpoint saved — "
                f"{len(year_rows):,} new rows | "
                f"{n_ckpt_total:,} total | "
                f"Elapsed: {elapsed_min:.1f} min | "
                f"ETA: {eta_min:.0f} min\n"
            )

    # ── Close remote datasets ──────────────────────────────────────────────
    ds_sst.close()
    if ds_chl is not None:
        ds_chl.close()
    if ds_sal is not None:
        ds_sal.close()

    # ── Assemble final output CSV ──────────────────────────────────────────
    log.info("─── Assembling Final CSV ────────────────────────────────────────")
    frames = []
    if not df_ckpt.empty:
        frames.append(df_ckpt)
    if all_new_rows:
        frames.append(pd.DataFrame(all_new_rows))

    if not frames:
        log.error("No data was collected. Output CSV not written.")
        return

    df_final = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["Year", "Month", "Lat", "Lon"])
        .reset_index(drop=True)
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False)

    elapsed_total = time.time() - pipeline_start
    log.info(f"\n╔══════════════════════════════════════════════════════════════╗")
    log.info(f"║                    PIPELINE COMPLETE                        ║")
    log.info(f"╚══════════════════════════════════════════════════════════════╝")
    log.info(f"  Output:    {OUTPUT_CSV}")
    log.info(f"  Rows:      {len(df_final):,}")
    log.info(f"  Columns:   {list(df_final.columns)}")
    log.info(f"  Elapsed:   {elapsed_total / 60:.1f} minutes")
    log.info(f"  Log file:  {LOG_DIR / 'grid_generation.log'}\n")


if __name__ == "__main__":
    main()
