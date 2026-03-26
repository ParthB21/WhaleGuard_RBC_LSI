import os
import json
import pandas as pd
import re
import subprocess

# --- FOLDER SETUP ---
# Explicitly setting your project root based on your location
PROJECT_ROOT = "/Users/vishnu/Documents/code/Whaleguard"

# Create the specific data paths for a clean project structure
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "acoustic")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Create the folders on your Mac if they don't exist yet
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# The final destination for your master dataset
OUTPUT_CSV_PATH = os.path.join(PROCESSED_DIR, "Master_Acoustic_Sightings.csv")


# 1. Your list of 53 NOAA Google Cloud Storage URLs
gcs_urls = [
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/fk01/sanctsound_fk01_01_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/fk02/sanctsound_fk02_01_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/fk03/sanctsound_fk03_01_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/gr01/sanctsound_gr01_01_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/gr01/sanctsound_gr01_02_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/gr01/sanctsound_gr01_03_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/gr01/sanctsound_gr01_04_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/gr01/sanctsound_gr01_05_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/gr02/sanctsound_gr02_02_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/gr03/sanctsound_gr03_02_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb01/sanctsound_sb01_01_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb01/sanctsound_sb01_02_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb01/sanctsound_sb01_03_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb01/sanctsound_sb01_04_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb01/sanctsound_sb01_05_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb01/sanctsound_sb01_06_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb01/sanctsound_sb01_07_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb01/sanctsound_sb01_08_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb01/sanctsound_sb01_09_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb01/sanctsound_sb01_10_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb01/sanctsound_sb01_11_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb01/sanctsound_sb01_12_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb02/sanctsound_sb02_01_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb02/sanctsound_sb02_02_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb02/sanctsound_sb02_03_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb02/sanctsound_sb02_04_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb02/sanctsound_sb02_05_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb02/sanctsound_sb02_06_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb02/sanctsound_sb02_07_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb02/sanctsound_sb02_08_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb02/sanctsound_sb02_09_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb02/sanctsound_sb02_10_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb02/sanctsound_sb02_11_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb02/sanctsound_sb02_12_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb03/sanctsound_sb03_01_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb03/sanctsound_sb03_02_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb03/sanctsound_sb03_03_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb03/sanctsound_sb03_04_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb03/sanctsound_sb03_05_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb03/sanctsound_sb03_06_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb03/sanctsound_sb03_07_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb03/sanctsound_sb03_08_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb03/sanctsound_sb03_09_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb03/sanctsound_sb03_10_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb03/sanctsound_sb03_11_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/sanctsound/products/detections/sb03/sanctsound_sb03_12_northatlanticrightwhale_1d",
    "gs://noaa-passive-bioacoustic/dclde/2013/nefsc_sbnms_200903_nopp6_ch10/",
    "gs://noaa-passive-bioacoustic/nefsc/products/detections/nefsc_baleen/nefsc_sbnms_200712_nopp1_ch10_20080127_narw/",
    "gs://noaa-passive-bioacoustic/nefsc/products/detections/nefsc_baleen/nefsc_ne-offshore_201406_rwm1_20140926_narw/",
    "gs://noaa-passive-bioacoustic/nefsc/products/detections/nefsc_baleen/nefsc_nc_201712_hatteras_ch2_20180121_narw/",
    "gs://noaa-passive-bioacoustic/nefsc/products/detections/nefsc_baleen/nefsc_ma-ri_201512_ch4_20160218_narw/",
    "gs://noaa-passive-bioacoustic/nefsc/products/detections/nefsc_baleen/nefsc_georges-bank_201806_wat-hz_narw/",
    "gs://noaa-passive-bioacoustic/nefsc/products/detections/nefsc_baleen/nefsc_ga_201510_ch1_20160108_narw/"
]


def download_datasets():
    print(f"Creating directory: {DOWNLOAD_DIR}")
    
    for url in gcs_urls:
        print(f"Downloading {url}...")
        # This triggers your newly installed gcloud CLI
        result = subprocess.run(["gcloud", "storage", "cp", "-r", url, DOWNLOAD_DIR], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error downloading {url}: {result.stderr}")
    print("--- Download Complete ---")

def process_acoustic_data():
    all_whale_data = []
    
    print("Starting processing...")
    for folder_name in os.listdir(DOWNLOAD_DIR):
        folder_path = os.path.join(DOWNLOAD_DIR, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            
        json_file = None
        csv_file = None
        
        # Check standard SanctSound structure (metadata/ and data/ folders)
        metadata_dir = os.path.join(folder_path, "metadata")
        data_dir = os.path.join(folder_path, "data")
        
        # Handle cases where CSV and JSON are directly in the folder (like older nefsc data)
        if os.path.exists(metadata_dir) and os.path.exists(data_dir):
            for f in os.listdir(metadata_dir):
                if f.endswith('.json'): json_file = os.path.join(metadata_dir, f)
            for f in os.listdir(data_dir):
                if f.endswith('.csv'): csv_file = os.path.join(data_dir, f)
        else:
            for f in os.listdir(folder_path):
                if f.endswith('.json'): json_file = os.path.join(folder_path, f)
                if f.endswith('.csv'): csv_file = os.path.join(folder_path, f)
                
        if json_file and csv_file:
            try:
                # 1. Extract Lat/Lon
                with open(json_file, 'r') as file:
                    metadata = json.load(file)
                    # Handle different JSON keys just in case
                    shape_str = metadata.get("SHAPE", metadata.get("shape", ""))
                    
                    match = re.search(r'POINT \(([^ ]+) ([^ ]+)\)', str(shape_str))
                    if match:
                        lon = float(match.group(1))
                        lat = float(match.group(2))
                    else:
                        print(f"Skipping {folder_name}: Could not find Lat/Lon coordinates in JSON.")
                        continue
                
                # 2. Extract Data
                df = pd.read_csv(csv_file)
                
                # We only want days where the whale was present (> 0)
                if 'Presence' in df.columns:
                    whales_present_df = df[df['Presence'] > 0].copy()
                else:
                    # Fallback if column is named differently in older datasets
                    whales_present_df = df.copy() 
                
                # 3. Combine it
                whales_present_df['LAT'] = lat
                whales_present_df['LON'] = lon
                whales_present_df['SOURCE'] = 'Acoustic'
                
                all_whale_data.append(whales_present_df)
                print(f"Success! {folder_name}: Found {len(whales_present_df)} days with whales.")
            except Exception as e:
                print(f"Error processing {folder_name}: {e}")

    # Final Merge
    if all_whale_data:
        master_df = pd.concat(all_whale_data, ignore_index=True)
        # Standardize timestamp column name
        if 'ISOStartTime' in master_df.columns:
            master_df.rename(columns={'ISOStartTime': 'SIGHTINGDATE'}, inplace=True)
        return master_df
    else:
        print("Failed to compile master dataset.")
        return pd.DataFrame()

# --- EXECUTE EVERYTHING ---
# (It will take some time to download everything on the first run!)
download_datasets() 

final_data = process_acoustic_data()

if not final_data.empty:
    final_data.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nBOOM! Master dataset successfully saved to: {OUTPUT_CSV_PATH}")