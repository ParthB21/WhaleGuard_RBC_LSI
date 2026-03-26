import os
import pandas as pd
import numpy as np
import requests
import io
import time
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Reuse one HTTP session for faster/more reliable API calls
SESSION = requests.Session()

# --- FOLDER SETUP ---
PROJECT_ROOT = "/Users/vishnu/Documents/code/Whaleguard"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_CSV = os.path.join(PROCESSED_DIR, "Master_Acoustic_Sightings.csv")
VISUAL_CSV = os.path.join(PROJECT_ROOT, "otherData", "23305_RWSAS.csv")

# We will save individual files as requested, plus the final master file
SST_CSV = os.path.join(PROCESSED_DIR, "1_Acoustic_with_SST_Fronts.csv")
CHL_CSV = os.path.join(PROCESSED_DIR, "2_Acoustic_with_Chlorophyll.csv")
SAL_CSV = os.path.join(PROCESSED_DIR, "3_Acoustic_with_Salinity.csv")
MASTER_CSV = os.path.join(PROCESSED_DIR, "FINAL_Master_Training_Data.csv")
REQUEST_SLEEP_SECONDS = 0.1
ENV_CACHE_CSV = os.path.join(PROCESSED_DIR, "env_lookup_cache.csv")

# Keep network calls bounded so one bad location doesn't stall the whole run
REQUEST_TIMEOUT = (3, 8)  # (connect, read)
MAX_SECONDS_PER_VARIABLE = 25


def _parse_griddap_csv(csv_text):
    """ERDDAP CSV responses include a units row at index 1."""
    return pd.read_csv(io.StringIO(csv_text), skiprows=[1])


def _build_griddap_url(dataset_id, variable, date_str, lat, lon, half_box_deg, altitude=None):
    lat_min, lat_max = lat - half_box_deg, lat + half_box_deg
    lon_min, lon_max = lon - half_box_deg, lon + half_box_deg
    altitude_selector = f"[({altitude})]" if altitude is not None else ""
    return (
        f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/{dataset_id}.csv"
        f"?{variable}[({date_str}T12:00:00Z)]"
        f"{altitude_selector}"
        f"[({lat_min}):({lat_max})][({lon_min}):({lon_max})]"
    )

# --- 1. HELPER FUNCTIONS TO FETCH REAL DATA ---
def fetch_regional_mean(dataset_id, variable, date_str, lat, lon):
    """
    Fetches regional means with temporal/spatial fallback.
    This greatly reduces salinity/chlorophyll misses near coast/cloud coverage.
    """
    cache_key = (dataset_id, variable, date_str, round(float(lat), 2), round(float(lon), 2))
    if cache_key in fetch_regional_mean._cache:
        return fetch_regional_mean._cache[cache_key]

    day_offsets = [0, -1, 1, -3, 3]
    half_boxes = [0.20, 0.35, 0.50, 0.75]
    altitude = 0.0 if dataset_id == 'coastwatchSMOSv662SSS1day' else None
    started = time.time()

    saw_404 = False
    for day_offset in day_offsets:
        query_date = (pd.Timestamp(date_str) + pd.Timedelta(days=day_offset)).strftime('%Y-%m-%d')
        for half_box in half_boxes:
            try:
                url = _build_griddap_url(dataset_id, variable, query_date, lat, lon, half_box, altitude=altitude)
                response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
                if response.status_code == 404:
                    saw_404 = True
                    continue
                response.raise_for_status()

                df_grid = _parse_griddap_csv(response.text)
                if df_grid.empty or variable not in df_grid.columns:
                    continue

                value = pd.to_numeric(df_grid[variable], errors='coerce').mean()
                if np.isnan(value):
                    continue

                result = float(value)
                fetch_regional_mean._cache[cache_key] = result
                return result

            except requests.exceptions.Timeout:
                continue
            except requests.exceptions.HTTPError:
                continue
            except Exception:
                continue

            if (time.time() - started) > MAX_SECONDS_PER_VARIABLE:
                fetch_regional_mean._cache[cache_key] = np.nan
                return np.nan

        if (time.time() - started) > MAX_SECONDS_PER_VARIABLE:
            fetch_regional_mean._cache[cache_key] = np.nan
            return np.nan

    if saw_404:
        print(f"      [!] {variable}: no valid grid values found after fallback attempts.")
    fetch_regional_mean._cache[cache_key] = np.nan
    return np.nan


fetch_regional_mean._cache = {}

def get_sst_and_front(date_str, lat, lon):
    """Fetches a tiny grid of SST to calculate the Frontal Value"""
    cache_key = (date_str, round(float(lat), 2), round(float(lon), 2))
    if cache_key in get_sst_and_front._cache:
        return get_sst_and_front._cache[cache_key]

    day_offsets = [0, -1, 1]
    half_boxes = [0.05, 0.10, 0.20]
    started = time.time()

    for day_offset in day_offsets:
        query_date = (pd.Timestamp(date_str) + pd.Timedelta(days=day_offset)).strftime('%Y-%m-%d')
        for half_box in half_boxes:
            try:
                url = _build_griddap_url('jplMURSST41', 'analysed_sst', query_date, lat, lon, half_box)
                response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
                if response.status_code == 404:
                    continue
                response.raise_for_status()

                df_grid = _parse_griddap_csv(response.text)
                if df_grid.empty or 'analysed_sst' not in df_grid.columns:
                    continue

                center_sst = pd.to_numeric(df_grid['analysed_sst'], errors='coerce').mean()
                if np.isnan(center_sst):
                    continue

                frontal_value = np.nan
                if {'latitude', 'longitude', 'analysed_sst'}.issubset(df_grid.columns):
                    pivot = df_grid.pivot_table(
                        index='latitude', columns='longitude', values='analysed_sst', aggfunc='mean'
                    )
                    grid_2d = pivot.values
                    if grid_2d.shape[0] >= 2 and grid_2d.shape[1] >= 2:
                        gy, gx = np.gradient(grid_2d)
                        frontal_value = np.nanmax(np.sqrt(gx**2 + gy**2))

                result = (float(center_sst), float(frontal_value) if not np.isnan(frontal_value) else np.nan)
                get_sst_and_front._cache[cache_key] = result
                return result

            except Exception:
                continue

            if (time.time() - started) > MAX_SECONDS_PER_VARIABLE:
                result = (np.nan, np.nan)
                get_sst_and_front._cache[cache_key] = result
                return result

        if (time.time() - started) > MAX_SECONDS_PER_VARIABLE:
            result = (np.nan, np.nan)
            get_sst_and_front._cache[cache_key] = result
            return result

    result = (np.nan, np.nan)
    get_sst_and_front._cache[cache_key] = result
    return result


get_sst_and_front._cache = {}


def load_combined_sightings():
    """Load acoustic base and append visual sightings if present."""
    acoustic = pd.read_csv(INPUT_CSV)
    acoustic['SOURCE'] = acoustic.get('SOURCE', 'Acoustic')
    acoustic['Presence'] = pd.to_numeric(acoustic.get('Presence', 1), errors='coerce').fillna(1)

    use_cols = ['SIGHTINGDATE', 'LAT', 'LON', 'SOURCE', 'Presence']
    acoustic = acoustic[use_cols].copy()

    include_visual = os.getenv('WG_INCLUDE_VISUAL', '1') != '0'
    if (not include_visual) or (not os.path.exists(VISUAL_CSV)):
        return acoustic

    visual = pd.read_csv(VISUAL_CSV)
    required = {'SIGHTINGDATE', 'LAT', 'LON'}
    if not required.issubset(set(visual.columns)):
        return acoustic

    # Keep confirmed sightings, but retain probable if needed for volume
    if 'CERTAINTY' in visual.columns:
        visual = visual[visual['CERTAINTY'].astype(str).str.lower().isin(['definite', 'probable'])].copy()

    # Fix occasional sign issues in this export (positive longitude where west expected)
    visual['LON'] = pd.to_numeric(visual['LON'], errors='coerce')
    visual['LAT'] = pd.to_numeric(visual['LAT'], errors='coerce')
    visual.loc[(visual['LON'] > 0) & (visual['LAT'].between(20, 65)), 'LON'] *= -1

    visual['SIGHTINGDATE'] = pd.to_datetime(visual['SIGHTINGDATE'], format='%d-%b-%y', errors='coerce')
    visual = visual.dropna(subset=['SIGHTINGDATE', 'LAT', 'LON'])
    visual = visual[(visual['LAT'].between(-90, 90)) & (visual['LON'].between(-180, 180))]

    # Optional duplicate flag from source file
    if 'DUPLICATE' in visual.columns:
        dup_mask = visual['DUPLICATE'].astype(str).str.strip().str.lower().isin(['yes', 'true', '1'])
        visual = visual[~dup_mask]

    visual['SOURCE'] = 'Visual'
    visual['Presence'] = 1
    visual = visual[use_cols].copy()

    combined = pd.concat([acoustic, visual], ignore_index=True)
    combined = combined.drop_duplicates(subset=['SIGHTINGDATE', 'LAT', 'LON', 'SOURCE'])
    return combined


def fill_missing_environment(df, columns):
    """Fill remaining missing env values so all rows can be retained for training."""
    out = df.copy()
    for col in columns:
        missing_before = out[col].isna().sum()
        if missing_before == 0:
            continue

        # 1) same site climatology
        out[col] = out[col].fillna(out.groupby(['LAT', 'LON'])[col].transform('median'))
        # 2) monthly climatology
        out[col] = out[col].fillna(out.groupby(out['SIGHTINGDATE'].dt.month)[col].transform('median'))
        # 3) global fallback
        out[col] = out[col].fillna(out[col].median())

        missing_after = out[col].isna().sum()
        print(f"  -> {col}: filled {missing_before - missing_after}/{missing_before} missing values")
    return out


def load_env_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}
    try:
        cdf = pd.read_csv(cache_path)
        required = {'DATE_KEY', 'LAT_KEY', 'LON_KEY', 'SST_Celsius', 'Frontal_Value', 'Chlorophyll_mg_m3', 'Salinity_PSU'}
        if not required.issubset(set(cdf.columns)):
            return {}
        out = {}
        for _, r in cdf.iterrows():
            key = (str(r['DATE_KEY']), float(r['LAT_KEY']), float(r['LON_KEY']))
            out[key] = (
                pd.to_numeric(r['SST_Celsius'], errors='coerce'),
                pd.to_numeric(r['Frontal_Value'], errors='coerce'),
                pd.to_numeric(r['Chlorophyll_mg_m3'], errors='coerce'),
                pd.to_numeric(r['Salinity_PSU'], errors='coerce')
            )
        return out
    except Exception:
        return {}


def save_env_cache(cache_path, env_lookup):
    if not env_lookup:
        return
    rows = []
    for (date_key, lat_key, lon_key), values in env_lookup.items():
        rows.append({
            'DATE_KEY': date_key,
            'LAT_KEY': lat_key,
            'LON_KEY': lon_key,
            'SST_Celsius': values[0],
            'Frontal_Value': values[1],
            'Chlorophyll_mg_m3': values[2],
            'Salinity_PSU': values[3]
        })
    pd.DataFrame(rows).to_csv(cache_path, index=False)

# --- 2. LOAD DATA & ADD SEASON ---
print("Loading Base Data...")
df = load_combined_sightings()
df['SIGHTINGDATE'] = pd.to_datetime(df['SIGHTINGDATE'], errors='coerce')
df = df.dropna(subset=['SIGHTINGDATE', 'LAT', 'LON']).reset_index(drop=True)

month_to_season = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
}
df['Season'] = df['SIGHTINGDATE'].dt.month.map(month_to_season)

# --- 3. FETCH DATA ITERATIVELY ---
print(f"Beginning fetch for {len(df)} rows. This uses real NOAA APIs.")

sst_list, front_list, chl_list, sal_list = [], [], [], []

df['DATE_KEY'] = df['SIGHTINGDATE'].dt.strftime('%Y-%m-%d')
df['LAT_KEY'] = df['LAT'].round(2)
df['LON_KEY'] = df['LON'].round(2)
query_df = df[['DATE_KEY', 'LAT_KEY', 'LON_KEY']].drop_duplicates().reset_index(drop=True)
max_keys_env = os.getenv('WG_MAX_QUERY_KEYS')
if max_keys_env:
    try:
        max_keys = int(max_keys_env)
        if max_keys > 0:
            query_df = query_df.head(max_keys).copy()
            print(f"WG_MAX_QUERY_KEYS set -> running first {len(query_df)} unique query keys")
    except ValueError:
        pass

print(f"Unique environmental query points: {len(query_df)}")

env_lookup = load_env_cache(ENV_CACHE_CSV)
print(f"Loaded cached query points: {len(env_lookup)}")

for index, row in query_df.iterrows():
    date_str = row['DATE_KEY']
    lat = row['LAT_KEY']
    lon = row['LON_KEY']
    key = (date_str, lat, lon)

    if key in env_lookup:
        continue

    if index % 10 == 0:
        print(f"  -> Processing key {index}/{len(query_df)}: {date_str} ({lat}, {lon})")

    sst, front = get_sst_and_front(date_str, lat, lon)
    chl = fetch_regional_mean('erdMH1chla8day', 'chlorophyll', date_str, lat, lon)
    sal = fetch_regional_mean('coastwatchSMOSv662SSS1day', 'sss', date_str, lat, lon)

    env_lookup[key] = (sst, front, chl, sal)

    if (index + 1) % 10 == 0:
        save_env_cache(ENV_CACHE_CSV, env_lookup)

    time.sleep(REQUEST_SLEEP_SECONDS) # Polite pause for the server

save_env_cache(ENV_CACHE_CSV, env_lookup)

for index, row in df.iterrows():
    date_str = row['DATE_KEY']
    lat = row['LAT_KEY']
    lon = row['LON_KEY']
    key = (date_str, lat, lon)
    if key in env_lookup:
        sst, front, chl, sal = env_lookup[key]
    else:
        sst, front, chl, sal = (np.nan, np.nan, np.nan, np.nan)
    sst_list.append(sst)
    front_list.append(front)
    chl_list.append(chl)
    sal_list.append(sal)

df = df.drop(columns=['DATE_KEY', 'LAT_KEY', 'LON_KEY'])

# --- 4. SAVE INDIVIDUAL DATASETS ---
print("Saving individual citation datasets...")

df_sst = df.copy()
df_sst['SST_Celsius'] = sst_list
df_sst['Frontal_Value'] = front_list
df_sst.to_csv(SST_CSV, index=False)

df_chl = df.copy()
df_chl['Chlorophyll_mg_m3'] = chl_list
df_chl.to_csv(CHL_CSV, index=False)

df_sal = df.copy()
df_sal['Salinity_PSU'] = sal_list
df_sal.to_csv(SAL_CSV, index=False)

# --- 5. BUILD AND CLEAN THE MASTER DATASET ---
print("Building Final Master Dataset...")
df_master = df.copy()
df_master['SST_Celsius'] = sst_list
df_master['Frontal_Value'] = front_list
df_master['Chlorophyll_mg_m3'] = chl_list
df_master['Salinity_PSU'] = sal_list

# Fill remaining missing values so all sightings can be preserved
print("Filling remaining missing environmental values...")
df_master = fill_missing_environment(
    df_master,
    columns=['SST_Celsius', 'Frontal_Value', 'Chlorophyll_mg_m3', 'Salinity_PSU']
)

if len(df_master) >= 3: # Need at least 3 rows to make 3 clusters
    # --- 6. ENGINEER 'WATER MASS' USING K-MEANS ---
    print("Running Machine Learning (K-Means) to classify Water Mass...")
    features = df_master[['SST_Celsius', 'Salinity_PSU']]
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_master['Water_Mass_ID'] = kmeans.fit_predict(features)
    
    centers = kmeans.cluster_centers_
    sorted_idx = np.argsort(centers[:, 0]) 
    mapping = {sorted_idx[0]: 'Cold/Fresh', sorted_idx[1]: 'Temperate/Mixed', sorted_idx[2]: 'Warm/Salty'}
    df_master['Water_Mass_Name'] = df_master['Water_Mass_ID'].map(mapping)

    df_master.to_csv(MASTER_CSV, index=False)
    print(f"\nSUCCESS! Master dataset ready for model training at: {MASTER_CSV}")
else:
    print("\nWarning: Not enough complete rows to run clustering.")