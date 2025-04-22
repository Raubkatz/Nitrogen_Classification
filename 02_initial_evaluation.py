#!/usr/bin/env python3
"""
Initial evaluation script that:
  1. Loads 'Grassland_Biomass_Merged_mv.csv'.
  2. Drops unwanted columns (PlotID, Year, etc.).
  3. Quantile-based categorization of biomass into 2, 3, 4, 5, or 33 bins.
  4. Creates boxplots for each numeric feature by the newly assigned category.
  5. Generates a comprehensive report:
     - Missing values count per feature
     - Basic descriptive stats for numeric features
     - Value counts for categorical features
     - Category distribution
  6. Saves all outputs to an 'initial_evaluation' folder.

Usage:
    Adjust NUM_CATEGORIES as desired, then run the script.

Note:
    This script does NOT do any scaling or model training. It's purely for
    an initial data exploration of the original dataset with user-defined
    quantile splits for biomass.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.colors import ListedColormap

# User settings
NUM_CATEGORIES = 2  # Choose among {2, 3, 4, 5, 33} for the categorization
DATA_CSV = 'Grassland_Biomass_Merged_mv.csv'
UNWANTED_COLS = ['PlotID', 'Year', 'DriveSpeed_kmh', 'CutHeight_cm', 'CutWidth_m']

# Font size control
BASE_FONT_SIZE = 18
TITLE_FONT_SIZE = BASE_FONT_SIZE + 2

# Create a timestamped folder "initial_evaluation_<timestamp>"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
eval_folder = f"initial_evaluation_{timestamp}_nk{NUM_CATEGORIES}"
os.makedirs(eval_folder, exist_ok=True)

# Define color palette
custom_palette = ["#188FA7", "#769FB6", "#9DBBAE", "#D5D6AA", "#E2DBBE"]


def main():
    # -----------------------------------------------------
    # 1) Load Data
    # -----------------------------------------------------
    print("[INFO] Loading data:", DATA_CSV)
    df = pd.read_csv(DATA_CSV)
    if 'Biomass' not in df.columns:
        raise KeyError("Biomass column not found in dataset!")

    # -----------------------------------------------------
    # 2) Drop Unwanted Columns
    # -----------------------------------------------------
    drop_cols = [col for col in UNWANTED_COLS if col in df.columns]
    if drop_cols:
        print("[INFO] Dropping columns:", drop_cols)
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Ensure 'Biomass' is numeric and drop rows with missing Biomass
    df['Biomass'] = pd.to_numeric(df['Biomass'], errors='coerce')
    df.dropna(subset=['Biomass'], inplace=True)

    # -----------------------------------------------------
    # 3) Quantile-based Categorization
    # -----------------------------------------------------
    biomass_values = df['Biomass'].values

    def categorize_biomass(n_cat, y):
        """Return category assignment and category labels (in ascending order)."""
        if n_cat == 2:
            median_val = np.nanquantile(y, 0.50)

            def cat_func(v):
                return "low" if v <= median_val else "high"

            labels = ["low", "high"]
            return np.array([cat_func(v) for v in y]), labels
        elif n_cat == 3:
            q1, q2 = np.nanquantile(y, [0.33, 0.66])

            def cat_func(v):
                if v <= q1:
                    return "low"
                elif v <= q2:
                    return "medium"
                else:
                    return "high"

            labels = ["low", "medium", "high"]
            return np.array([cat_func(v) for v in y]), labels
        elif n_cat == 4:
            q1, q2, q3 = np.nanquantile(y, [0.25, 0.50, 0.75])

            def cat_func(v):
                if v <= q1:
                    return "very_low"
                elif v <= q2:
                    return "low"
                elif v <= q3:
                    return "high"
                else:
                    return "very_high"

            labels = ["very_low", "low", "high", "very_high"]
            return np.array([cat_func(v) for v in y]), labels
        elif n_cat == 5:
            q1, q2, q3, q4 = np.nanquantile(y, [0.20, 0.40, 0.60, 0.80])

            def cat_func(v):
                if v <= q1:
                    return "very_low"
                elif v <= q2:
                    return "low"
                elif v <= q3:
                    return "medium"
                elif v <= q4:
                    return "high"
                else:
                    return "very_high"

            labels = ["very_low", "low", "medium", "high", "very_high"]
            return np.array([cat_func(v) for v in y]), labels
        elif n_cat == 33:
            q_low, q_high = np.nanquantile(y, [0.20, 0.80])

            def cat_func(v):
                if v <= q_low:
                    return "very_low"
                elif v <= q_high:
                    return "mid"
                else:
                    return "very_high"

            labels = ["very_low", "mid", "very_high"]
            return np.array([cat_func(v) for v in y]), labels
        else:
            raise ValueError(f"NUM_CATEGORIES={n_cat} is not supported.")

    # Obtain category assignments and initial labels.
    cat_assignments, cat_labels = categorize_biomass(NUM_CATEGORIES, biomass_values)
    df['BiomassCategory'] = cat_assignments

    # Convert BiomassCategory to a categorical variable with the desired descending order.
    if NUM_CATEGORIES == 2:
        sorted_order = ["high", "low"]
    elif NUM_CATEGORIES == 3:
        sorted_order = ["high", "medium", "low"]
    elif NUM_CATEGORIES == 4:
        sorted_order = ["very_high", "high", "low", "very_low"]
    elif NUM_CATEGORIES == 5:
        sorted_order = ["very_high", "high", "medium", "low", "very_low"]
    elif NUM_CATEGORIES == 33:
        sorted_order = ["very_high", "high", "low", "very_low"]
    else:
        sorted_order = cat_labels  # fallback to the original order

    df['BiomassCategory'] = pd.Categorical(df['BiomassCategory'], categories=sorted_order, ordered=True)

    # Display distribution.
    print("[INFO] Category distribution (sorted descending):")
    print(df['BiomassCategory'].value_counts())

    # -----------------------------------------------------
    # 4) Boxplots for Each Numeric Feature by Category
    # -----------------------------------------------------
    boxplot_folder = os.path.join(eval_folder, "boxplots")
    os.makedirs(boxplot_folder, exist_ok=True)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "Biomass"]
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))

        # Use the proper sorted categorical order
        class_order = df['BiomassCategory'].cat.categories

        sns.boxplot(
            x="BiomassCategory", y=col, data=df,
            palette=custom_palette, showfliers=False, order=class_order
        )
        plt.title(f"{col} by {NUM_CATEGORIES}-Category Biomass", fontsize=TITLE_FONT_SIZE)
        plt.xlabel("Biomass Category", fontsize=BASE_FONT_SIZE)
        plt.ylabel(col, fontsize=BASE_FONT_SIZE)
        plt.tight_layout()
        png_path = os.path.join(boxplot_folder, f"boxplot_{col}.png")
        eps_path = os.path.join(boxplot_folder, f"boxplot_{col}.eps")
        plt.savefig(png_path, dpi=300)
        plt.savefig(eps_path, format='eps')
        plt.clf()
        plt.close()
        print(f"[INFO] Saved boxplot for {col}: {png_path}")

    # -----------------------------------------------------
    # 5) Full Report: Missing Values, Descriptive Stats, Value Counts, and Category Distribution
    # -----------------------------------------------------
    report_lines = []
    report_lines.append("==== Initial Data Evaluation Report ====\n")

    # Missing values.
    report_lines.append("** Missing Values per Column **")
    na_counts = df.isna().sum()
    for c in df.columns:
        report_lines.append(f"{c}: {na_counts[c]} missing")
    report_lines.append("")

    # Numeric stats.
    report_lines.append("** Numeric Column Stats **")
    for col in numeric_cols:
        desc = df[col].describe()
        report_lines.append(f"--- {col} ---\n{desc.to_string()}")
    report_lines.append("")

    # Categorical value counts.
    report_lines.append("** Categorical Column Value Counts **")
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    for c in cat_cols:
        report_lines.append(f"--- {c} ---")
        vc = df[c].value_counts(dropna=False)
        report_lines.append(vc.to_string())
        report_lines.append("")
    report_lines.append("")

    # Category distribution.
    report_lines.append("** Biomass Category Distribution (Sorted) **")
    cat_dist = df["BiomassCategory"].value_counts()
    report_lines.append(cat_dist.to_string())
    report_lines.append("")

    report_path = os.path.join(eval_folder, "InitialEvaluationReport.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"[INFO] Full data evaluation report saved to: {report_path}")
    print("[INFO] Initial evaluation completed.")

    # -----------------------------------------------------
    # Additional Boxplots with Whiskers from Min to Max
    # -----------------------------------------------------
    for col in numeric_cols:
        plt.figure(figsize=(8, 10))

        # Instead of df["BiomassCategory"].unique(), use the correct categorical order
        class_order = df["BiomassCategory"].cat.categories

        data_by_class = [
            df[df["BiomassCategory"] == cls][col].dropna() for cls in class_order
        ]

        # Draw boxplot with whiskers from min to max
        box = plt.boxplot(data_by_class, patch_artist=True, whis=[0, 100])

        # Apply custom colors
        for patch, color in zip(box['boxes'], custom_palette):
            patch.set_facecolor(color)

        # Axis labels and ticks with our specified font sizes
        plt.xticks(ticks=range(1, len(class_order) + 1), labels=class_order, fontsize=BASE_FONT_SIZE+8)
        plt.yticks(fontsize=BASE_FONT_SIZE+8)
        plt.xlabel("Biomass Category", fontsize=BASE_FONT_SIZE+8)
        plt.ylabel(col, fontsize=BASE_FONT_SIZE+8)
        plt.title(f"{col}", fontsize=TITLE_FONT_SIZE+8)
        plt.tight_layout()

        plot_png = os.path.join(boxplot_folder, f"{col}_boxplot_minmax.png")
        plot_eps = os.path.join(boxplot_folder, f"{col}_boxplot_minmax.eps")
        plt.savefig(plot_png, dpi=300)
        plt.savefig(plot_eps, format="eps")
        plt.clf()
        plt.close()
        print(f"[INFO] Boxplot (min-max) for {col} saved to {plot_png} and {plot_eps}.")


if __name__ == "__main__":
    main()
