#!/usr/bin/env python3
"""
Script to merge two CSV files (Grassland_AllPlots.csv and Biomass_AllPlots.csv) on PlotID and Year.
[docstring truncated – original text kept unchanged …]
"""

import pandas as pd
import numpy as np

# ─────────────────────────── Configuration ───────────────────────────
MISSING_VALUE_OPTION             = 1   # 0,1,2,3,4
FERTILIZER_ZERO_FIX              = False
ZERO_AS_EMPTY_IF_NO_FERTILIZATION = True     # ← NEW FLAG
SEPARATE_FERTILIZATION           = True

EXTRA_COLS = ["Manure_tha", "Slurry_m3ha", "minPhosphorus_kgPha",
              "minPotassium_kgKha", "Sulphur_kgSha"]

# full list of fertiliser-related columns for the new rule
FERTILIZER_VARS = ["NbFertilization", "Manure_tha", "Slurry_m3ha",
                   "orgNitrogen_kgNha", "minNitrogen_kgNha", "totalNitrogen_kgNha",
                   "minPhosphorus_kgPha", "minPotassium_kgKha", "Sulphur_kgSha"]

# ────────────────────────────── Main ─────────────────────────────────
def main():
    # 1) load raw CSVs -------------------------------------------------
    df_xxx = pd.read_csv("./biomass_31448_9.csv")
    df_xxx.rename(columns={"Useful_EP_PlotID": "PlotID",
                           "Year": "Year",
                           "biomass": "Biomass"}, inplace=True)
    df_xxx.to_csv("Biomass_31448.csv", index=False)

    df_grassland = pd.read_csv("Grassland_AllPlots.csv")
    df_biomass   = pd.read_csv("Biomass_31448.csv")

    df_grassland.drop(columns=["StudyRegion"], inplace=True, errors="ignore")

    # 2) merge & basic biomass filter ---------------------------------
    df_merged = pd.merge(df_grassland,
                         df_biomass[["PlotID", "Year", "Biomass"]],
                         on=["PlotID", "Year"], how="inner")
    df_merged["Biomass"] = pd.to_numeric(df_merged["Biomass"], errors="coerce")
    df_merged = df_merged[df_merged["Biomass"] > 0]

    # 3) NEW zero→"" rule for non-fertilised rows ---------------------
    if ZERO_AS_EMPTY_IF_NO_FERTILIZATION and not FERTILIZER_ZERO_FIX:
        fert_cols = [c for c in df_merged.columns if "fertil" in c.lower()]
        if fert_cols:
            fert_col = fert_cols[0]
            no_mask  = df_merged[fert_col].astype(str).str.lower().str.contains("no")
            for col in FERTILIZER_VARS:
                if col in df_merged.columns:
                    idx = (df_merged[col] == 0) & no_mask
                    if idx.any():
                        df_merged.loc[idx, col] = ""      # ← empty string
            print("[INFO] Zero→'' conversion applied to fertiliser fields for non-fertilised rows.")
        else:
            print("[WARN] ZERO_AS_EMPTY_IF_NO_FERTILIZATION set but no fertilisation column found.")

    # 4) save merged pre-processing snapshot --------------------------
    df_merged.to_csv("Grassland_Biomass_Merged.csv", index=False)
    print("[INFO] Merged data saved to 'Grassland_Biomass_Merged.csv'.")

    # 5) optional split by fertilisation ------------------------------
    if SEPARATE_FERTILIZATION:
        fert_cols = [c for c in df_merged.columns if "fertil" in c.lower()]
        if fert_cols:
            fert_col   = fert_cols[0]
            yes_mask   = df_merged[fert_col].astype(str).str.lower().str.contains("yes")
            df_merged[yes_mask].to_csv("Grassland_Biomass_Merged_fertilization_yes.csv", index=False)
            df_merged[~yes_mask].to_csv("Grassland_Biomass_Merged_fertilization_no.csv",  index=False)
            print("[INFO] Fertilisation split saved.")
        else:
            print("[WARN] No fertilisation column found; skipping split.")

    # 6) nitrogen-fix block  (original code unchanged) ----------------
    # -----------------------------------------------------------------
    if "orgNitrogen_kgNha" in df_merged.columns:
        df_merged["orgNitrogen_kgNha"] = pd.to_numeric(df_merged["orgNitrogen_kgNha"], errors="coerce")
        df_merged.loc[df_merged["orgNitrogen_kgNha"] <= 0, "orgNitrogen_kgNha"] = np.nan
        df_merged["orgNitrogen_kgNha"].fillna(df_merged["orgNitrogen_kgNha"].mean(skipna=True), inplace=True)

    if "minNitrogen_kgNha" in df_merged.columns:
        df_merged["minNitrogen_kgNha"] = pd.to_numeric(df_merged["minNitrogen_kgNha"], errors="coerce")
        df_merged.loc[df_merged["minNitrogen_kgNha"] <= 0, "minNitrogen_kgNha"] = np.nan
        min_avg = df_merged["minNitrogen_kgNha"].mean(skipna=True)
        fert_cols = [c for c in df_merged.columns if "fertil" in c.lower()]
        if fert_cols:
            fert_col = fert_cols[0]
            yes_m = df_merged[fert_col].astype(str).str.lower().str.contains("yes")
            df_merged.loc[yes_m,  "minNitrogen_kgNha"] = df_merged.loc[yes_m,  "minNitrogen_kgNha"].fillna(min_avg)
            df_merged.loc[~yes_m, "minNitrogen_kgNha"] = df_merged.loc[~yes_m, "minNitrogen_kgNha"].fillna(0)
        else:
            df_merged["minNitrogen_kgNha"].fillna(min_avg, inplace=True)

    if "totalNitrogen_kgNha" in df_merged.columns:
        df_merged["totalNitrogen_kgNha"] = pd.to_numeric(df_merged["totalNitrogen_kgNha"], errors="coerce")
        df_merged.loc[df_merged["totalNitrogen_kgNha"] <= 0, "totalNitrogen_kgNha"] = np.nan
        df_merged["totalNitrogen_kgNha"] = df_merged.apply(
            lambda r: r.get("orgNitrogen_kgNha", 0) + r.get("minNitrogen_kgNha", 0)
                      if pd.isna(r["totalNitrogen_kgNha"]) or
                         not np.isclose(r["totalNitrogen_kgNha"],
                                        r.get("orgNitrogen_kgNha", 0)+r.get("minNitrogen_kgNha", 0))
                      else r["totalNitrogen_kgNha"], axis=1)

    df_merged.to_csv("Grassland_Biomass_Merged_NitrogenFixed.csv", index=False)

    # 7) missing-value processing options 0-4 (original logic intact) --
    nitrogen_cols = ["orgNitrogen_kgNha", "minNitrogen_kgNha", "totalNitrogen_kgNha"]

    if MISSING_VALUE_OPTION == 0:
        df_final = df_merged.copy()

    elif MISSING_VALUE_OPTION == 1:
        df_final = df_merged.copy()
        non_nit = [c for c in df_final.columns if c not in nitrogen_cols]
        df_final[non_nit] = df_final[non_nit].replace([-1, "-1"], "")
        if FERTILIZER_ZERO_FIX:
            for col in non_nit:
                if any(k in col.lower() for k in ["nitrogen","potassium","sulph","phosphor","slurry"]):
                    if pd.api.types.is_numeric_dtype(df_final[col]):
                        df_final.loc[df_final[col] == 0, col] = np.nan
        for col in EXTRA_COLS:
            if col in df_final.columns:
                df_final.loc[df_final[col] == 0, col] = ""

    elif MISSING_VALUE_OPTION == 2:
        df_final = df_merged.replace([-1, "-1", ""], np.nan)
        if FERTILIZER_ZERO_FIX:
            for col in df_final.columns:
                if any(k in col.lower() for k in ["nitrogen","potassium","sulph","phosphor","slurry"]):
                    if pd.api.types.is_numeric_dtype(df_final[col]):
                        df_final.loc[df_final[col] == 0, col] = np.nan
        for col in EXTRA_COLS:
            if col in df_final.columns:
                df_final.loc[df_final[col] == 0, col] = np.nan

    elif MISSING_VALUE_OPTION == 3:
        df_final = df_merged.replace([-1,"-1",""], np.nan)
        if FERTILIZER_ZERO_FIX:
            for col in df_final.columns:
                if any(k in col.lower() for k in ["nitrogen","potassium","sulph","phosphor","slurry"]):
                    if pd.api.types.is_numeric_dtype(df_final[col]):
                        df_final.loc[df_final[col] == 0, col] = np.nan
        for col in EXTRA_COLS:
            if col in df_final.columns:
                df_final.loc[df_final[col] == 0, col] = np.nan
        df_final.dropna(inplace=True)

    elif MISSING_VALUE_OPTION == 4:
        df_final = df_merged.replace([-1,"-1",""], np.nan)
        non_nit = [c for c in df_final.columns if c not in nitrogen_cols]
        for col in non_nit:
            if pd.api.types.is_numeric_dtype(df_final[col]):
                df_final[col].fillna(df_final[col].median(skipna=True), inplace=True)
        if FERTILIZER_ZERO_FIX:
            for col in non_nit:
                if any(k in col.lower() for k in ["nitrogen","potassium","sulph","phosphor","slurry"]):
                    if pd.api.types.is_numeric_dtype(df_final[col]):
                        z = df_final[col] == 0
                        if z.any():
                            df_final.loc[z, col] = df_final.loc[~z, col].median(skipna=True)
        for col in EXTRA_COLS:
            if col in df_final.columns:
                if pd.api.types.is_numeric_dtype(df_final[col]):
                    z = df_final[col] == 0
                    if z.any():
                        df_final.loc[z, col] = df_final.loc[~z, col].median(skipna=True)
                else:
                    df_final.loc[df_final[col] == 0, col] = ""

    # 8) write final output -------------------------------------------
    out_name = {
        0: "Grassland_Biomass_Merged_raw.csv",
        1: "Grassland_Biomass_Merged_mv_final.csv",
        2: "Grassland_Biomass_Merged_nan.csv",
        3: "Grassland_Biomass_Merged_dmv.csv",
        4: "Grassland_Biomass_Merged_median.csv"
    }[MISSING_VALUE_OPTION]
    df_final.to_csv(out_name, index=False)
    print(f"[INFO] Final dataset written to '{out_name}'.")

if __name__ == "__main__":
    main()
