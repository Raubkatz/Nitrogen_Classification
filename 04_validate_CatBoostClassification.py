#!/usr/bin/env python3
"""
Evaluation script for a CatBoost classifier trained with ADASYN oversampling and Bayesian optimization.
It loads the model, scaler, quantiles, and the column information from a specified experiment folder,
then reconstructs and evaluates the same data structure for predictions.

Steps:
  1. Load all training artifacts (model, scaler, quantiles, catboost training columns, numeric columns).
  2. Reindex the validation/test data to the exact catboost training columns (same order).
  3. Convert categorical features to codes if needed.
  4. (Optionally) Subset the features.
  5. Predict with the model.
  6. Generate confusion matrices and classification reports.
  7. Plot feature importance.
  8. Produce distribution boxplots and summaries.

All output diagrams are generated with the class labels sorted in descending order. For example:
    - For 5 categories: ["very_high", "high", "medium", "low", "very_low"]
    - For 4 categories: ["very_high", "high", "low", "very_low"]
    - For 3 categories: ["high", "medium", "low"]
    - For 2 categories: ["high", "low"]

All outputs (plots, CSVs, logs) are saved in subfolders of the experiment folder.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from matplotlib.colors import ListedColormap

# ---------------------
# Plot Configuration
# ---------------------
BASE_FONT_SIZE = 18
TITLE_FONT_SIZE = BASE_FONT_SIZE + 1

import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = BASE_FONT_SIZE
mpl.rcParams['ytick.labelsize'] = BASE_FONT_SIZE

# ---------------------
# User Settings
# ---------------------
NUM_CATEGORIES = 2
#EXPERIMENT_FOLDER = "Experiment_numcat5_rs42_ts20250414085315"
#EXPERIMENT_FOLDER = "Experiment_numcat4_rs42_ts20250413115713"
#EXPERIMENT_FOLDER = "Experiment_numcat3_rs42_ts20250413110726"
#EXPERIMENT_FOLDER = "Experiment_numcat2_rs42_ts20250413103409"

#EXPERIMENT_FOLDER = "Experiment_numcat5_rs42_ts20250415125547"
#EXPERIMENT_FOLDER = "Experiment_numcat4_rs42_ts20250415120536"
#EXPERIMENT_FOLDER = "Experiment_numcat3_rs42_ts20250415113946"
EXPERIMENT_FOLDER = "Experiment_numcat2_rs42_ts20250415111959"

SELECTED_FEATURES = None  # If None, all columns are used.

CUSTOM_PALETTE = [
    "#E2DBBE",
    "#D5D6AA",
    "#9DBBAE",
    "#769FB6",
    "#188FA7",
]

# ---------------------
# Utility Functions
# ---------------------
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def convert_categorical_to_codes(df):
    """Convert columns of object or categorical dtype to numeric codes."""
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("category"):
            df[col] = df[col].astype("category").cat.codes
    return df

def plot_confusion_matrix(cm, class_labels, title, save_path_png, save_path_eps):
    cmap = ListedColormap(CUSTOM_PALETTE)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=class_labels, yticklabels=class_labels,
        annot_kws={"size": BASE_FONT_SIZE}
    )
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Predicted Labels", fontsize=BASE_FONT_SIZE)
    plt.ylabel("True Labels", fontsize=BASE_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(save_path_png)
    plt.savefig(save_path_eps, format='eps')
    plt.clf()
    plt.close()

def plot_relative_confusion_matrix(cm, class_labels, title, save_path_png, save_path_eps):
    relative_cm = cm / cm.sum(axis=1, keepdims=True)
    cmap = ListedColormap(CUSTOM_PALETTE)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        relative_cm, annot=True, fmt=".2f", cmap=cmap,
        xticklabels=class_labels, yticklabels=class_labels,
        annot_kws={"size": BASE_FONT_SIZE}
    )
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Predicted Labels", fontsize=BASE_FONT_SIZE)
    plt.ylabel("True Labels", fontsize=BASE_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(save_path_png)
    plt.savefig(save_path_eps, format='eps')
    plt.clf()
    plt.close()

def generate_distribution_summary(df, group_col, output_file):
    """
    For each feature in df (excluding group_col):
      - If numeric, compute .describe() grouped by group_col
      - If categorical, compute value_counts grouped by group_col
    """
    summary_lines = []
    features = [col for col in df.columns if col != group_col]
    grouped = df.groupby(group_col)
    for feat in features:
        summary_lines.append(f"Feature: {feat}")
        if pd.api.types.is_numeric_dtype(df[feat]):
            stats = grouped[feat].describe()
            summary_lines.append(stats.to_string())
        else:
            for grp_val, subdf in grouped:
                summary_lines.append(f"Class: {grp_val}")
                counts = subdf[feat].value_counts()
                uniques = subdf[feat].unique()
                summary_lines.append(counts.to_string())
                summary_lines.append(f"Unique values: {uniques}")
        summary_lines.append("-" * 60)
    with open(output_file, "w") as f:
        f.write("\n".join(summary_lines))

# ---------------------
# Main Validation Code
# ---------------------
def main():
    models_folder = os.path.join(EXPERIMENT_FOLDER, "models")

    # Load CatBoost model.
    model_files = sorted(f for f in os.listdir(models_folder) if f.startswith("catboost_classifier"))
    if not model_files:
        raise FileNotFoundError("No CatBoost model file found.")
    best_model = load_pickle(os.path.join(models_folder, model_files[0] + ".pkl"))

    # Load scaler and column info.
    scaler = load_pickle(os.path.join(models_folder, sorted(f for f in os.listdir(models_folder) if f.startswith("scaler_"))[0]))
    numeric_cols_trained = load_pickle(os.path.join(models_folder, sorted(f for f in os.listdir(models_folder) if f.startswith("scaler_columns_"))[0]))
    catboost_training_cols = load_pickle(os.path.join(models_folder, sorted(f for f in os.listdir(models_folder) if f.startswith("catboost_training_columns_"))[0]))
    quantiles = load_pickle(os.path.join(models_folder, sorted(f for f in os.listdir(models_folder) if f.startswith("quantiles_"))[0]))

    print("[INFO] Successfully loaded model, scaler, column info, and quantiles.")

    # Load data CSVs.
    data_folder = os.path.join(EXPERIMENT_FOLDER, "data")
    X_train_orig = pd.read_csv(os.path.join(data_folder, "X_train_original.csv"))
    y_train_orig = pd.read_csv(os.path.join(data_folder, "y_train_original.csv")).squeeze()
    X_test = pd.read_csv(os.path.join(data_folder, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_folder, "y_test.csv")).squeeze()
    X_train_adasyn = pd.read_csv(os.path.join(data_folder, "X_train_adasyn.csv"))
    y_train_adasyn = pd.read_csv(os.path.join(data_folder, "y_train_adasyn.csv")).squeeze()

    # Prepare feature frames.
    X_train_eval = convert_categorical_to_codes(X_train_orig.copy())
    X_test_eval = convert_categorical_to_codes(X_test.copy())
    X_train_adasyn_eval = convert_categorical_to_codes(X_train_adasyn.copy())

    if SELECTED_FEATURES:
        X_train_eval = X_train_eval[SELECTED_FEATURES]
        X_test_eval = X_test_eval[SELECTED_FEATURES]
        X_train_adasyn_eval = X_train_adasyn_eval[SELECTED_FEATURES]

    # Align columns for CatBoost.
    X_train_eval = X_train_eval.reindex(columns=catboost_training_cols)
    X_test_eval = X_test_eval.reindex(columns=catboost_training_cols)
    X_train_adasyn_eval = X_train_adasyn_eval.reindex(columns=catboost_training_cols)

    print("[INFO] Making predictions on test, original train, and ADASYN train sets...")
    y_pred_test = best_model.predict(X_test_eval)
    y_pred_train_orig = best_model.predict(X_train_eval)
    y_pred_train_adasyn = best_model.predict(X_train_adasyn_eval)

    # Class labels in descending order.
    if NUM_CATEGORIES == 2:
        class_labels = ["high", "low"]
    elif NUM_CATEGORIES == 3:
        class_labels = ["high", "medium", "low"]
    elif NUM_CATEGORIES == 4:
        class_labels = ["very_high", "high", "low", "very_low"]
    elif NUM_CATEGORIES == 5:
        class_labels = ["very_high", "high", "medium", "low", "very_low"]
    else:
        class_labels = []

    # Prepare folders.
    eval_folder = os.path.join(EXPERIMENT_FOLDER, "evaluation")
    for sub in ["confusion_matrices", "feature_importance", "reports", "distributions"]:
        os.makedirs(os.path.join(eval_folder, sub), exist_ok=True)

    # Open single report file.
    report_path = os.path.join(eval_folder, "reports", "evaluation_report.txt")
    with open(report_path, "w") as report:
        report.write(f"Evaluation Report — {datetime.now().isoformat()}\n")
        report.write(f"Experiment folder: {EXPERIMENT_FOLDER}\n")
        report.write("="*80 + "\n\n")

        # Evaluate datasets.
        datasets = {
            "Test":      (X_test_eval,      y_test,      y_pred_test),
            "Train_Orig":(X_train_eval,     y_train_orig,y_pred_train_orig),
            "Train_ADASYN":(X_train_adasyn_eval, y_train_adasyn, y_pred_train_adasyn)
        }

        for name, (X_data, y_true, y_pred) in datasets.items():
            cm = confusion_matrix(y_true, y_pred, labels=class_labels)
            cm_rel = (cm / cm.sum(axis=1, keepdims=True)).round(2)
            acc = accuracy_score(y_true, y_pred)
            crep = classification_report(y_true, y_pred, labels=class_labels)

            # Write section header
            report.write(f"--- {name} ---\n")
            report.write(f"Accuracy: {acc:.4f}\n\n")
            report.write("Confusion Matrix (absolute):\n")
            report.write(np.array2string(cm) + "\n\n")
            report.write("Confusion Matrix (relative):\n")
            report.write(np.array2string(cm_rel) + "\n\n")
            report.write("Classification Report:\n")
            report.write(crep + "\n")
            report.write("-"*80 + "\n\n")

            # Also print to console
            print(f"[INFO] {name} Accuracy: {acc:.4f}")
            print("Confusion Matrix:\n", cm)
            print("Relative Confusion Matrix:\n", cm_rel)
            print("Classification Report:\n", crep)

        # Feature importance
        feature_importances = best_model.get_feature_importance()
        feature_names = X_train_orig.columns
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": (feature_importances / feature_importances.sum()).round(4)
        }).sort_values(by="Importance", ascending=False)

        # Plot
        fi_png = os.path.join(eval_folder, "feature_importance", "feature_importance.png")
        fi_eps = os.path.join(eval_folder, "feature_importance", "feature_importance.eps")
        plt.figure(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=importance_df, palette=CUSTOM_PALETTE)
        plt.title("Feature Importance", fontsize=TITLE_FONT_SIZE)
        plt.xlabel("Relative Importance", fontsize=BASE_FONT_SIZE)
        plt.ylabel("Feature", fontsize=BASE_FONT_SIZE)
        plt.tight_layout()
        plt.savefig(fi_png, dpi=300)
        plt.savefig(fi_eps, format="eps")
        plt.clf()
        plt.close()
        print(f"[INFO] Feature importance plots saved to {fi_png} and {fi_eps}.")

        # Write importance table
        report.write("=== Feature Importance ===\n")
        report.write(importance_df.to_string(index=False))
        report.write("\n")

    print(f"[INFO] Single combined report written to {report_path}")
    print("[INFO] Validation complete.")

if __name__ == "__main__":
    main()
