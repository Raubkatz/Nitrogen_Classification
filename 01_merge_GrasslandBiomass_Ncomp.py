#!/usr/bin/env python3
"""
Minimal merge-and-clean script for the fixed workflow:

1. Merge Grassland_AllPlots.csv & Biomass_31448_9.csv (on PlotID + Year).
2. Keep rows with Biomass > 0.
3. For rows whose Fertilization column says “no”, set any *0* in the
   fertiliser-related variables to an empty string ("").
4. Basic nitrogen fixes (org / min / total).
5. Replace all –1 (numeric or string) with empty strings ("") in every
   non-nitrogen column.
6. In EXTRA_COLS, convert zeros to empty strings.
7. Optional fertilisation split saved as “…_yes.csv” & “…_no.csv”.
8. Final cleaned CSV: Grassland_Biomass_Merged_mv_final.csv
"""

import pandas as pd
import numpy as np

# ─── constants ───────────────────────────────────────────────────────
ZERO_AS_EMPTY_IF_NO_FERTILIZATION = True
SEPARATE_FERTILIZATION            = True

EXTRA_COLS = ["Manure_tha", "Slurry_m3ha",
              "minPhosphorus_kgPha", "minPotassium_kgKha", "Sulphur_kgSha"]

FERTILIZER_VARS = ["NbFertilization", "Manure_tha", "Slurry_m3ha",
                   "orgNitrogen_kgNha", "minNitrogen_kgNha", "totalNitrogen_kgNha",
                   "minPhosphorus_kgPha", "minPotassium_kgKha", "Sulphur_kgSha"]

NITROGEN_COLS = ["orgNitrogen_kgNha", "minNitrogen_kgNha", "totalNitrogen_kgNha"]

# ─── helper ──────────────────────────────────────────────────────────
def mean_or_nan(s):
    return s.mean(skipna=True) if s.notna().any() else np.nan

# ─── main ────────────────────────────────────────────────────────────
def main():
    # 1) read and rename biomass file
    bio = (pd.read_csv("biomass_31448_9.csv")
             .rename(columns={"Useful_EP_PlotID": "PlotID",
                              "Year": "Year",
                              "biomass": "Biomass"}))
    bio.to_csv("Biomass_31448.csv", index=False)

    grass = pd.read_csv("Grassland_AllPlots.csv").drop(columns=["StudyRegion"],
                                                       errors="ignore")

    # 2) merge & basic biomass filter
    df = grass.merge(
        bio[["PlotID", "Year", "Biomass"]],
        on=["PlotID", "Year"], how="inner")

    df["Biomass"] = pd.to_numeric(df["Biomass"], errors="coerce")
    df = df[df["Biomass"] > 0]

    # 3) convert zeros to "" for non-fertilised rows
    fert_cols = [c for c in df.columns if "fertil" in c.lower()]
    if ZERO_AS_EMPTY_IF_NO_FERTILIZATION and fert_cols:
        fert_flag = fert_cols[0]
        no_mask   = df[fert_flag].astype(str).str.lower().str.contains("no")
        for col in FERTILIZER_VARS:
            if col in df.columns:
                df.loc[(df[col] == 0) & no_mask, col] = ""

    # 4) nitrogen fixes ------------------------------------------------
    if "orgNitrogen_kgNha" in df:
        df["orgNitrogen_kgNha"] = pd.to_numeric(df["orgNitrogen_kgNha"],
                                                errors="coerce")
        df.loc[df["orgNitrogen_kgNha"] <= 0, "orgNitrogen_kgNha"] = np.nan
        df["orgNitrogen_kgNha"].fillna(mean_or_nan(df["orgNitrogen_kgNha"]),
                                       inplace=True)

    if "minNitrogen_kgNha" in df:
        df["minNitrogen_kgNha"] = pd.to_numeric(df["minNitrogen_kgNha"],
                                                errors="coerce")
        df.loc[df["minNitrogen_kgNha"] <= 0, "minNitrogen_kgNha"] = np.nan
        min_avg = mean_or_nan(df["minNitrogen_kgNha"])
        if fert_cols:
            yes_mask = df[fert_flag].astype(str).str.lower().str.contains("yes")
            df.loc[yes_mask,  "minNitrogen_kgNha"] = (
                df.loc[yes_mask,  "minNitrogen_kgNha"].fillna(min_avg))
            df.loc[~yes_mask, "minNitrogen_kgNha"] = (
                df.loc[~yes_mask, "minNitrogen_kgNha"].fillna(0))
        else:
            df["minNitrogen_kgNha"].fillna(min_avg, inplace=True)

    if "totalNitrogen_kgNha" in df:
        df["totalNitrogen_kgNha"] = pd.to_numeric(df["totalNitrogen_kgNha"],
                                                  errors="coerce")
        df.loc[df["totalNitrogen_kgNha"] <= 0, "totalNitrogen_kgNha"] = np.nan
        df["totalNitrogen_kgNha"] = df.apply(
            lambda r: r["orgNitrogen_kgNha"] + r["minNitrogen_kgNha"]
                      if pd.isna(r["totalNitrogen_kgNha"]) or
                         not np.isclose(r["totalNitrogen_kgNha"],
                                        r["orgNitrogen_kgNha"]+r["minNitrogen_kgNha"])
                      else r["totalNitrogen_kgNha"], axis=1)

    df.to_csv("Grassland_Biomass_Merged_NitrogenFixed.csv", index=False)

    # 5) simple “option 1” missing-value processing --------------------
    non_nit = [c for c in df.columns if c not in NITROGEN_COLS]
    df[non_nit] = df[non_nit].replace([-1, "-1"], "")
    for col in EXTRA_COLS:
        if col in df.columns:
            df.loc[df[col] == 0, col] = ""

    # 6) optional fertilisation split ---------------------------------
    if SEPARATE_FERTILIZATION and fert_cols:
        yes = df[fert_flag].astype(str).str.lower().str.contains("yes")
        df[yes].to_csv("Grassland_Biomass_Merged_fertilization_yes.csv",
                       index=False)
        df[~yes].to_csv("Grassland_Biomass_Merged_fertilization_no.csv",
                        index=False)

    # 7) final output --------------------------------------------------
    out_file = "Grassland_Biomass_Merged_mv_final.csv"
    df.to_csv(out_file, index=False)
    print(f"[INFO] Final dataset written to '{out_file}'.")

if __name__ == "__main__":
    main()
