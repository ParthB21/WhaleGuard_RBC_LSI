import pandas as pd

base_dir = r"data\runs\run_20260331_190615_05b7b819"
paper_path = f"{base_dir}\\whale_dataset_paper.csv"
parth_path = r"c:\Users\sujay\Downloads\Parth Version (No Sightings) - REAL_Whale_NoSightings_10000_paper_from_cache.csv"
output_path = f"{base_dir}\\whale_dataset_final.csv"

# Read the CSV files
paper_df = pd.read_csv(paper_path, dtype=str)
parth_df = pd.read_csv(parth_path, dtype=str)

# Use paper CSV header order as target schema
target_columns = list(paper_df.columns)

# Normalize Parth columns to the paper schema
parth_columns = set(parth_df.columns)
missing_in_parth = [col for col in target_columns if col not in parth_columns]
extra_in_parth = [col for col in parth_df.columns if col not in target_columns]

if missing_in_parth:
    print(f"Warning: Parth file missing target columns: {missing_in_parth}")

# Fill missing target columns in Parth file with empty strings
for col in missing_in_parth:
    parth_df[col] = ""

# Drop any extra columns from Parth file that are not part of paper schema
if extra_in_parth:
    print(f"Dropping extra Parth columns: {extra_in_parth}")
    parth_df = parth_df[[col for col in parth_df.columns if col in target_columns]]

# Force all Parth rows to Presence=0
parth_df["Presence"] = "0"

# Align columns exactly to the paper schema order
parth_df = parth_df[target_columns]

# Ensure the paper dataset has the same ordered columns
target_paper_cols = [col for col in target_columns if col in paper_df.columns]
for col in target_columns:
    if col not in paper_df.columns:
        paper_df[col] = ""
paper_df = paper_df[target_columns]

# Concatenate paper and Parth datasets
combined_df = pd.concat([paper_df, parth_df], ignore_index=True)

# Sort by date and write output
combined_df["SIGHTINGDATE"] = pd.to_datetime(combined_df["SIGHTINGDATE"], errors="coerce")
combined_df = combined_df.sort_values(by="SIGHTINGDATE").reset_index(drop=True)
combined_df.to_csv(output_path, index=False)

print(f"Paper rows: {len(paper_df)}")
print(f"Parth non-sighting rows: {len(parth_df)}")
print(f"Combined rows: {len(combined_df)}")
print(f"Saved merged file to: {output_path}")
