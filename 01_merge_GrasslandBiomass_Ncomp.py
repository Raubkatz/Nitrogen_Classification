#!/usr/bin/env python3
"""
Script to merge two CSV files (Grassland_AllPlots.csv and Biomass_AllPlots.csv) on PlotID and Year.
Only records with matching PlotID and Year in both datasets are kept. Additionally, rows are discarded
if the biomass value is missing, non-numeric, or not a positive float.

Missing Value Processing Options:
    Option 0: No missing value compensation (keep as is).
    Option 1: Replace all "-1" values with empty strings ("").
    Option 2: Replace all "-1" values and empty strings ("") with np.nan.
    Option 3: Replace "-1" and empty strings with np.nan and then drop any row with missing values.
    Option 4: Replace "-1" and empty strings with np.nan, then fill each numeric column (except the nitrogen columns)
              with its median. Additionally, if FERTILIZER_ZERO_FIX is True, for fertilizer-related columns (those containing
              'nitrogen', 'potassium', 'sulph', 'phosphor', or 'slurry') replace zeros with the column median.

Additionally, if SEPARATE_FERTILIZATION is True, the script will immediately create two CSVs from the merged data:
    one containing only records where a fertilization-related column (i.e. a column whose name contains "fertil")
    indicates "yes", and one for all others.

Right after merging (and removing biomass ≤ 0), the script handles three nitrogen columns:
    1) orgNitrogen_kgNha: Remove all values ≤ 0 (and any -1) → NaN, then fill missing with the average of remaining.
    2) minNitrogen_kgNha: Remove all values ≤ 0 (and any -1) → NaN, then compute the average from remaining;
       then, for each row, if "Fertilization" is "yes", fill missing with the average; if "no", fill missing with 0.
    3) totalNitrogen_kgNha: Remove all values ≤ 0 (and any -1) → NaN, then for each row, if missing or not equal to
       (orgNitrogen_kgNha + minNitrogen_kgNha), set its value to that sum.
    4) Final check: For each row, if totalNitrogen_kgNha is not equal (within tolerance) to orgNitrogen_kgNha + minNitrogen_kgNha,
       replace it with the sum.

After that, in the later missing-value processing steps (Options 1–4), the three nitrogen columns are excluded.
Then, for a set of extra columns:
    ["Manure_tha", "Slurry_m3ha", "minPhosphorus_kgPha", "minPotassium_kgKha", "Sulphur_kgSha"]
any zeros are treated as missing according to the selected MISSING_VALUE_OPTION:
    - Option 1: Zeros → "" (empty string)
    - Option 2/3: Zeros → np.nan
    - Option 4: Zeros are replaced with the median (of nonzero values).

Usage:
    python merge_grassland_biomass.py
"""

import pandas as pd
import numpy as np

# ---------------------
# Configuration Block
# ---------------------
MISSING_VALUE_OPTION = 1  # Set to 0, 1, 2, 3, or 4.
FERTILIZER_ZERO_FIX = True
SEPARATE_FERTILIZATION = True

# Columns in which zeros should be treated as missing in the final processing:
EXTRA_COLS = ["Manure_tha", "Slurry_m3ha", "minPhosphorus_kgPha", "minPotassium_kgKha", "Sulphur_kgSha"]

def main():
    # 1. Read input CSV files.
    df_xxx = pd.read_csv("./biomass_31448_9.csv")

    # Rename the columns
    df_xxx.rename(columns={
        "Useful_EP_PlotID": "PlotID",
        "Year": "Year",
        "biomass": "Biomass"
    }, inplace=True)

    # (Optional) Save the result — do it if you actually want
    df_xxx.to_csv("Biomass_31448.csv", index=False)


    grassland_csv = "Grassland_AllPlots.csv"
    #biomass_csv = "Biomass_AllPlots.csv"
    biomass_csv = "Biomass_31448.csv"

    df_grassland = pd.read_csv(grassland_csv)
    df_biomass = pd.read_csv(biomass_csv)

    # 2. Drop unwanted column "StudyRegion" (if present).
    df_grassland.drop(columns=["StudyRegion"], inplace=True, errors="ignore")

    # 3. Merge on PlotID and Year.
    df_merged = pd.merge(
        df_grassland,
        df_biomass[["PlotID", "Year", "Biomass"]],
        on=["PlotID", "Year"],
        how="inner"
    )

    # 4. Convert Biomass to numeric and retain rows with positive Biomass.
    df_merged["Biomass"] = pd.to_numeric(df_merged["Biomass"], errors="coerce")
    df_merged = df_merged[df_merged["Biomass"] > 0]

    # Save original merged file.
    output_csv = "Grassland_Biomass_Merged.csv"
    df_merged.to_csv(output_csv, index=False)
    print(f"[INFO] Merged data saved to '{output_csv}'.")

    # --- Separate data by Fertilization if required ---
    if SEPARATE_FERTILIZATION:
        fert_cols = [col for col in df_merged.columns if "fertil" in col.lower()]
        if fert_cols:
            fert_col = fert_cols[0]
            df_fert_yes = df_merged[df_merged[fert_col].astype(str).str.lower().str.contains("yes")]
            df_fert_no = df_merged[~df_merged[fert_col].astype(str).str.lower().str.contains("yes")]
            output_fert_yes = "Grassland_Biomass_Merged_fertilization_yes.csv"
            output_fert_no = "Grassland_Biomass_Merged_fertilization_no.csv"
            df_fert_yes.to_csv(output_fert_yes, index=False)
            df_fert_no.to_csv(output_fert_no, index=False)
            print(f"[INFO] Fertilization separation done: saved '{output_fert_yes}' and '{output_fert_no}'.")
        else:
            print("[WARN] No fertilization-related column found; skipping fertilization separation.")

    # --- Nitrogen Processing Section ---
    # Process three nitrogen columns, if they exist.
    # (a) orgNitrogen_kgNha: remove ≤ 0, then fill missing with average.
    if "orgNitrogen_kgNha" in df_merged.columns:
        df_merged["orgNitrogen_kgNha"] = pd.to_numeric(df_merged["orgNitrogen_kgNha"], errors="coerce")
        df_merged.loc[df_merged["orgNitrogen_kgNha"] <= 0, "orgNitrogen_kgNha"] = np.nan
        org_avg = df_merged["orgNitrogen_kgNha"].mean(skipna=True)
        df_merged["orgNitrogen_kgNha"].fillna(org_avg, inplace=True)
        print(f"[INFO] orgNitrogen_kgNha: missing replaced with average {org_avg:.2f}.")

    # (b) minNitrogen_kgNha: remove ≤ 0, then compute average; for each row, if Fertilization == yes fill missing with avg, else set missing to 0.
    if "minNitrogen_kgNha" in df_merged.columns:
        df_merged["minNitrogen_kgNha"] = pd.to_numeric(df_merged["minNitrogen_kgNha"], errors="coerce")
        df_merged.loc[df_merged["minNitrogen_kgNha"] <= 0, "minNitrogen_kgNha"] = np.nan
        min_avg = df_merged["minNitrogen_kgNha"].mean(skipna=True)
        fert_cols = [col for col in df_merged.columns if "fertil" in col.lower()]
        if fert_cols:
            fert_col = fert_cols[0]
            yes_mask = df_merged[fert_col].astype(str).str.lower().str.contains("yes")
            df_merged.loc[yes_mask, "minNitrogen_kgNha"] = df_merged.loc[yes_mask, "minNitrogen_kgNha"].fillna(min_avg)
            df_merged.loc[~yes_mask, "minNitrogen_kgNha"] = df_merged.loc[~yes_mask, "minNitrogen_kgNha"].fillna(0)
            print(f"[INFO] minNitrogen_kgNha: fertilized missing filled with average {min_avg:.2f}, non-fertilized set to 0.")
        else:
            df_merged["minNitrogen_kgNha"].fillna(min_avg, inplace=True)
            print(f"[INFO] minNitrogen_kgNha (no fertil info): missing filled with average {min_avg:.2f}.")

    # (c) totalNitrogen_kgNha: remove ≤ 0, then for each row, if missing or if not equal to (org + min), replace with (org + min).
    if "totalNitrogen_kgNha" in df_merged.columns:
        df_merged["totalNitrogen_kgNha"] = pd.to_numeric(df_merged["totalNitrogen_kgNha"], errors="coerce")
        df_merged.loc[df_merged["totalNitrogen_kgNha"] <= 0, "totalNitrogen_kgNha"] = np.nan
        def fix_total(row):
            total_calc = row.get("orgNitrogen_kgNha", 0) + row.get("minNitrogen_kgNha", 0)
            if pd.isna(row["totalNitrogen_kgNha"]) or not np.isclose(row["totalNitrogen_kgNha"], total_calc):
                return total_calc
            return row["totalNitrogen_kgNha"]
        df_merged["totalNitrogen_kgNha"] = df_merged.apply(fix_total, axis=1)
        print("[INFO] totalNitrogen_kgNha: corrected to equal sum(orgNitrogen, minNitrogen) where needed.")

    # Save the dataset after nitrogen fixes.
    output_fixed = "Grassland_Biomass_Merged_NitrogenFixed.csv"
    df_merged.to_csv(output_fixed, index=False)
    print(f"[INFO] Nitrogen columns fixed. Data saved to '{output_fixed}'.")

    # --- Missing Value Processing for Non-Nitrogen Columns ---
    # Exclude the nitrogen columns from further processing.
    nitrogen_cols = ["orgNitrogen_kgNha", "minNitrogen_kgNha", "totalNitrogen_kgNha"]
    # Also, store the columns that need extra zero-removal later.
    extra_cols = ["Manure_tha", "Slurry_m3ha", "minPhosphorus_kgPha", "minPotassium_kgKha", "Sulphur_kgSha"]

    if MISSING_VALUE_OPTION == 0:
        df_final = df_merged.copy()
    elif MISSING_VALUE_OPTION == 1:
        df_final = df_merged.copy()
        non_nitro = [col for col in df_final.columns if col not in nitrogen_cols]
        df_final[non_nitro] = df_final[non_nitro].replace(-1, "").replace("-1", "")
        if FERTILIZER_ZERO_FIX:
            fert_keywords = ["nitrogen", "potassium", "sulph", "phosphor", "slurry"]
            for col in non_nitro:
                if any(kw in col.lower() for kw in fert_keywords):
                    if pd.api.types.is_numeric_dtype(df_final[col]):
                        df_final.loc[df_final[col] == 0, col] = np.nan
        # --- Extra processing for specified columns: treat zeros as missing.
        for col in extra_cols:
            if col in df_final.columns:
                df_final.loc[df_final[col] == 0, col] = ""
        output_mv = "Grassland_Biomass_Merged_mv.csv"
        df_final.to_csv(output_mv, index=False)
        print(f"[INFO] Option 1 applied: processed non-nitrogen columns. Output saved to '{output_mv}'.")
    elif MISSING_VALUE_OPTION == 2:
        df_final = df_merged.copy()
        non_nitro = [col for col in df_final.columns if col not in nitrogen_cols]
        df_final[non_nitro] = df_final[non_nitro].replace(-1, np.nan).replace("-1", np.nan).replace("", np.nan)
        if FERTILIZER_ZERO_FIX:
            fert_keywords = ["nitrogen", "potassium", "sulph", "phosphor", "slurry"]
            for col in non_nitro:
                if any(kw in col.lower() for kw in fert_keywords):
                    if pd.api.types.is_numeric_dtype(df_final[col]):
                        df_final.loc[df_final[col] == 0, col] = np.nan
        for col in extra_cols:
            if col in df_final.columns:
                df_final.loc[df_final[col] == 0, col] = np.nan
        output_nan = "Grassland_Biomass_Merged_nan.csv"
        df_final.to_csv(output_nan, index=False)
        print(f"[INFO] Option 2 applied: processed non-nitrogen columns. Output saved to '{output_nan}'.")
    elif MISSING_VALUE_OPTION == 3:
        df_final = df_merged.copy()
        non_nitro = [col for col in df_final.columns if col not in nitrogen_cols]
        df_final[non_nitro] = df_final[non_nitro].replace(-1, np.nan).replace("-1", np.nan).replace("", np.nan)
        if FERTILIZER_ZERO_FIX:
            fert_keywords = ["nitrogen", "potassium", "sulph", "phosphor", "slurry"]
            for col in non_nitro:
                if any(kw in col.lower() for kw in fert_keywords):
                    if pd.api.types.is_numeric_dtype(df_final[col]):
                        df_final.loc[df_final[col] == 0, col] = np.nan
        for col in extra_cols:
            if col in df_final.columns:
                df_final.loc[df_final[col] == 0, col] = np.nan
        df_final.dropna(inplace=True)
        output_dmv = "Grassland_Biomass_Merged_dmv.csv"
        df_final.to_csv(output_dmv, index=False)
        print(f"[INFO] Option 3 applied: processed non-nitrogen columns and dropped rows with missing values. Output saved to '{output_dmv}'.")
    elif MISSING_VALUE_OPTION == 4:
        df_final = df_merged.copy()
        non_nitro = [col for col in df_final.columns if col not in nitrogen_cols]
        df_final[non_nitro] = df_final[non_nitro].replace(-1, np.nan).replace("-1", np.nan).replace("", np.nan)
        for col in non_nitro:
            if pd.api.types.is_numeric_dtype(df_final[col]):
                med_val = df_final[col].median(skipna=True)
                df_final[col].fillna(med_val, inplace=True)
        if FERTILIZER_ZERO_FIX:
            fert_keywords = ["nitrogen", "potassium", "sulph", "phosphor", "slurry"]
            for col in non_nitro:
                if any(kw in col.lower() for kw in fert_keywords):
                    if pd.api.types.is_numeric_dtype(df_final[col]):
                        zero_mask = (df_final[col] == 0)
                        if zero_mask.any():
                            med_nonzero = df_final.loc[~zero_mask, col].median(skipna=True)
                            df_final.loc[zero_mask, col] = med_nonzero
        for col in extra_cols:
            if col in df_final.columns:
                # Replace zeros in the extra columns with the median (for numeric columns)
                # or with empty strings if not numeric.
                if pd.api.types.is_numeric_dtype(df_final[col]):
                    zero_mask = (df_final[col] == 0)
                    if zero_mask.any():
                        med_nonzero = df_final.loc[~zero_mask, col].median(skipna=True)
                        df_final.loc[zero_mask, col] = med_nonzero
                else:
                    df_final.loc[df_final[col] == 0, col] = ""
        output_median = "Grassland_Biomass_Merged_median.csv"
        df_final.to_csv(output_median, index=False)
        print(f"[INFO] Option 4 applied: processed non-nitrogen columns; filled missing with median. Output saved to '{output_median}'.")

if __name__ == "__main__":
    main()
