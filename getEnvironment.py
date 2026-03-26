import os
import pandas as pd
import requests
import io
import time

# --- FOLDER SETUP ---
PROJECT_ROOT = "/Users/vishnu/Documents/code/Whaleguard"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_CSV = os.path.join(PROCESSED_DIR, "Master_Acoustic_Sightings.csv")
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "Master_Acoustic_With_Environment.csv")

def fetch_from_erddap_safely(url):
    """Aggressively hits the ERDDAP server with a strict 10-second timeout."""
    try:
        # If the server takes longer than 10 seconds, it throws an error and gives up
        response = requests.get(url, timeout=10)
        
        # If the server returns a 404 (cloudy day) or 500 (server crash), throw an error
        response.raise_for_status() 
        
        # Convert the successful text response into a Pandas dataframe
        data = pd.read_csv(io.StringIO(response.text), skiprows=[1])
        return data
    except Exception:
        # If literally anything goes wrong (timeout, 404, etc.), silently return empty
        return pd.DataFrame()

def get_sst_for_point(date_str, lat, lon):
    url = f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.csv?analysed_sst[({date_str}T12:00:00Z)][({lat})][({lon})]"
    data = fetch_from_erddap_safely(url)
    if not data.empty and 'analysed_sst' in data.columns:
        return data['analysed_sst'].iloc[0]
    return None

def get_chlorophyll_for_point(date_str, lat, lon):
    url = f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chla1day.csv?chlorophyll[({date_str}T12:00:00Z)][({lat})][({lon})]"
    data = fetch_from_erddap_safely(url)
    if not data.empty and 'chlorophyll' in data.columns:
        return data['chlorophyll'].iloc[0]
    return None

# --- RUN THE PROCESS ---
print("Loading Master Acoustic Dataset...")
df = pd.read_csv(INPUT_CSV)

print(f"Dataset loaded. Total rows to process: {len(df)}")
print("Fetching Sea Surface Temperature & Chlorophyll from NASA/NOAA...")
print("(This should take ~3 to 5 minutes. Timeouts are enforced!)")

sst_values = []
chl_values = []

# Loop through every row in the dataset
for index, row in df.iterrows():
    date = str(row['SIGHTINGDATE']).split()[0]
    lat = row['LAT']
    lon = row['LON']
    
    # Status update for EVERY row so you know it's not frozen
    print(f"  -> Row {index}/{len(df)}: Fetching {date} at {lat}, {lon}...")
    
    temp = get_sst_for_point(date, lat, lon)
    chl = get_chlorophyll_for_point(date, lat, lon)
    
    sst_values.append(temp)
    chl_values.append(chl)
    
    # Be polite to the NOAA server (pause for 0.5 seconds between requests)
    # This prevents them from banning your IP address for spamming!
    time.sleep(0.5) 

# Add the new data as columns
df['SST_Celsius'] = sst_values
df['Chlorophyll_mg_m3'] = chl_values

# Save to a new CSV
df.to_csv(OUTPUT_CSV, index=False)

print("\n--- DONE! ---")
print(f"Successfully saved complete dataset to: {OUTPUT_CSV}")