#!/usr/bin/env python3
"""
Combined script to convert and clean the Biodiversity‐Exploratories grassland management data.

Data source
-----------
This dataset comes from the article
“Eleven years’ data of grassland management in Germany”
by Juliane Vogt et al., Biodiversity Data Journal 2020.
See:
  • Article page: https://bdj.pensoft.net/article/36387/element/5/5234383//
  • Supplementary TXT: https://bdj.pensoft.net/article/download/suppl/5234383/

The raw TXT (tab‑separated, UTF‑16) contains 116 management variables
for 150 grassland plots surveyed annually (2006–2016).

Script overview
---------------
1. **convert_txt_to_csv**
   Reads `DATA/oo_338774.txt` (tab‑separated, UTF‑16)
   Writes `DATA/oo_338774.csv`.

2. **clean_and_extract**
   Reads `DATA/oo_338774.csv`,
   Fixes `PlotID` (adds leading zero to single‐digit numeric suffix),
   Selects a predefined subset of 25 columns,
   Writes `Grassland_AllPlots.csv`.

Usage
-----
    python this_script.py
"""

import os
import pandas as pd
import re

# ---------------------
# Step 1: TXT → CSV
# ---------------------
def convert_txt_to_csv(
    input_folder="DATA",
    input_filename="oo_338774.txt",
    output_filename="oo_338774.csv"
):
    input_path = os.path.join(input_folder, input_filename)
    output_path = os.path.join(input_folder, output_filename)
    try:
        # Read the TXT file assuming tab separation and UTF‑16 encoding
        df = pd.read_csv(input_path, sep='\t', dtype=str, encoding='utf-16')
        # Save as CSV
        df.to_csv(output_path, index=False)
        print(f"Successfully converted {input_filename} → {output_filename}")
    except Exception as e:
        print(f"Error converting TXT to CSV: {e}")
        raise


# ---------------------
# Step 2: Clean & Extract
# ---------------------
def fix_plot_id(plot_id):
    """
    Add a leading zero to the numeric part of PlotID if it's a single digit.
    E.g. AEG1 → AEG01, AEG10 → AEG10.
    """
    match = re.match(r"^([A-Za-z]+)(\d+)$", str(plot_id))
    if not match:
        return plot_id
    prefix, num = match.groups()
    if len(num) == 1:
        num = f"0{num}"
    return f"{prefix}{num}"

def clean_and_extract(
    input_csv="./DATA/oo_338774.csv",
    output_csv="Grassland_AllPlots.csv"
):
    # Read the newly created CSV
    df = pd.read_csv(input_csv)

    # Fix PlotID formatting
    df['PlotID'] = df['PlotID'].apply(fix_plot_id)

    # Columns to keep
    columns_to_keep = [
        "StudyRegion",
        "Year",
        "PlotID",
        "Drainage",
        "WaterLogging",
        "Mowing",
        "MowingMaschine",
        "CutWidth_m",
        "CutHeight_cm",
        "DriveSpeed_kmh",
        "MowingConditioner",
        "Fertilization",
        "NbFertilization",
        "Manure_tha",
        "Slurry_m3ha",
        "orgNitrogen_kgNha",
        "minNitrogen_kgNha",
        "totalNitrogen_kgNha",
        "minPhosphorus_kgPha",
        "minPotassium_kgKha",
        "Sulphur_kgSha",
        "Maintenance",
        "Leveling",
        "Mulching",
        "Seeds"
    ]

    # Extract subset
    df_subset = df[columns_to_keep]

    # Save to CSV
    df_subset.to_csv(output_csv, index=False)
    print(f"Data cleaned and saved to '{output_csv}'.")


# ---------------------
# Entry point
# ---------------------
if __name__ == "__main__":
    # 1) Convert TXT to CSV
    convert_txt_to_csv()

    # 2) Clean the new CSV and extract columns
    clean_and_extract()
