#!/usr/bin/env python3
"""
Script to classify biomass into either 2, 3, 4, 5, or 33 categories based on quantile thresholds.
The data is loaded from 'Grassland_Biomass_Merged.csv'. The script removes 'PlotID', 'Year', and other unwanted columns
from the features and uses the remaining variables to predict the biomass category.
ADASYN is applied on the training data to generate synthetic samples for all classes so that each class reaches a total
count equal to OVERSAMPLE_FACTOR times the original majority class count.
First, a default CatBoost classifier (out-of-the-box) is trained on the oversampled data.
Then, Bayesian optimization (via BayesSearchCV) is used to tune the hyperparameters of a CatBoost classifier.
Both models are evaluated on the validation set, and the better model is selected.
All outputs—including the synthesized training data, non-synthesized training/test data, the saved model,
the scaler, the quantile thresholds, and detailed logs (classification reports, training logs, plots)—
are saved into a global experiment folder.

Usage:
    Simply run the script.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from matplotlib.colors import ListedColormap
from imblearn.over_sampling import ADASYN
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# ---------------------
# Configuration Block
# ---------------------
NUM_CATEGORIES = 5  # choose 2, 3, 4, 5, or 33
# We only used 2,3,4,5

# OVERSAMPLE_FACTOR: each class will be upsampled to OVERSAMPLE_FACTOR times the original majority count.
OVERSAMPLE_FACTOR = 7
ADASYN_N_NEIGHBORS = 5
RANDOM_STATE = 42

# OVERSAMPLE_FACTOR = 3 #3, best so far for num_Categories 2
# OVERSAMPLE_FACTOR = 4.5 #4 actually #4.5, best so far for num_Categories 4
# OVERSAMPLE_FACTOR = 2. #2.0 best so far num_Categories 3
# OVERSAMPLE_FACTOR = 7 #7.0 best so far num_Categories 5

# Bayesian optimization configuration
N_BAYES_ITER = 20  # number of Bayesian search iterations
CV_FOLDS = 3      # cross-validation folds
SCORING = "accuracy"

# Bayesian Search Space for CatBoostClassifier
bayesian_search_spaces = {
    "depth": Integer(4, 12),
    "iterations": Integer(500, 5000),
    "l2_leaf_reg": Integer(1, 12),
    "border_count": Integer(32, 156)
}

# ---------------------
# Global Experiment Folder Setup
# ---------------------
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
experiment_folder = f"Experiment_numcat{NUM_CATEGORIES}_rs{RANDOM_STATE}_ts{timestamp}"
for sub in ["models", "result_txts", "feature_importance", "confusion_matrices", "data"]:
    os.makedirs(os.path.join(experiment_folder, sub), exist_ok=True)

# ---------------------
# Define Custom Color Palette
# ---------------------
custom_palette = ["#188FA7", "#769FB6", "#9DBBAE", "#D5D6AA", "#E2DBBE"]


def main():
    n_categories = NUM_CATEGORIES

    # ---------------------
    # Data Loading & Cleaning
    # ---------------------
    print("[INFO] Loading data from Grassland_Biomass_Merged_mv.csv...")
    data = pd.read_csv('./Grassland_Biomass_Merged_mv_final.csv')
    #data = pd.read_csv('./Grassland_Biomass_Merged_median.csv')
    target_col = 'Biomass'
    if target_col not in data.columns:
        raise KeyError(f"Target variable '{target_col}' not found in dataset!")

    # Drop unwanted columns (e.g., PlotID, Year, etc.)
    cols_to_drop = [col for col in ['PlotID', 'Year', "DriveSpeed_kmh", "CutHeight_cm", "CutWidth_m", "Mulching", "Seeds"] if
                    col in data.columns]
    if cols_to_drop:
        print(f"[INFO] Dropping columns {cols_to_drop} from features...")
    data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
    data.dropna(subset=[target_col], inplace=True)
    X = data.drop(columns=[target_col] + cols_to_drop, errors='ignore')
    y = data[target_col]

    # Identify numeric and categorical columns
    def is_numeric_series(series):
        try:
            pd.to_numeric(series, errors='raise')
            return True
        except Exception:
            return False

    numeric_cols = [col for col in X.columns if is_numeric_series(X[col])]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X[categorical_cols] = X[categorical_cols].fillna("Unknown")
    for col in numeric_cols:
        X[col].fillna(X[col].median(), inplace=True)

    # ---------------------
    # Define Biomass Categorization & Compute Quantiles
    # ---------------------
    if n_categories == 2:
        median = y.quantile(0.50)
        print(f"[INFO] Median threshold for biomass: {median:.2f}")

        def categorize_biomass(value):
            return 'low' if value <= median else 'high'

        class_labels = ['low', 'high']
    elif n_categories == 3:
        q1, q2 = y.quantile([0.33, 0.66]).values
        print(f"[INFO] 33% and 66% thresholds for biomass: {q1:.2f}, {q2:.2f}")

        def categorize_biomass(value):
            if value <= q1:
                return 'low'
            elif value <= q2:
                return 'medium'
            else:
                return 'high'

        class_labels = ['low', 'medium', 'high']
    elif n_categories == 4:
        q1, q2, q3 = y.quantile([0.25, 0.50, 0.75]).values
        print(f"[INFO] 25%, 50%, and 75% thresholds for biomass: {q1:.2f}, {q2:.2f}, {q3:.2f}")

        def categorize_biomass(value):
            if value <= q1:
                return 'very_low'
            elif value <= q2:
                return 'low'
            elif value <= q3:
                return 'high'
            else:
                return 'very_high'

        class_labels = ['very_low', 'low', 'high', 'very_high']
    elif n_categories == 5:
        q1, q2, q3, q4 = y.quantile([0.20, 0.40, 0.60, 0.80]).values
        print(f"[INFO] Quintile thresholds for biomass: {q1:.2f}, {q2:.2f}, {q3:.2f}, {q4:.2f}")

        def categorize_biomass(value):
            if value <= q1:
                return 'very_low'
            elif value <= q2:
                return 'low'
            elif value <= q3:
                return 'medium'
            elif value <= q4:
                return 'high'
            else:
                return 'very_high'

        class_labels = ['very_low', 'low', 'medium', 'high', 'very_high']
    elif n_categories == 33:
        q_low, q_high = y.quantile([0.20, 0.80]).values
        print(f"[INFO] 20% and 80% thresholds for biomass: {q_low:.2f}, {q_high:.2f}")

        def categorize_biomass(value):
            if value <= q_low:
                return 'very_low'
            elif value <= q_high:
                return 'mid'
            else:
                return 'very_high'

        class_labels = ['very_low', 'mid', 'very_high']

    y_class = y.apply(categorize_biomass)

    # Save quantile thresholds in a dictionary for later use.
    if n_categories == 2:
        quantiles = {"median": median}
    elif n_categories == 3:
        quantiles = {"q1": q1, "q2": q2}
    elif n_categories == 4:
        quantiles = {"q1": q1, "q2": q2, "q3": q3}
    elif n_categories == 5:
        quantiles = {"q1": q1, "q2": q2, "q3": q3, "q4": q4}
    elif n_categories == 33:
        quantiles = {"q_low": q_low, "q_high": q_high}

    print("[INFO] Overall target category distribution:")
    print(y_class.value_counts())

    # ---------------------
    # Train/Test Split & Scaling
    # ---------------------
    X_train, X_val, y_train, y_val = train_test_split(X, y_class, test_size=0.1, random_state=RANDOM_STATE)
    print("[INFO] Distribution in training data before ADASYN:")
    print(y_train.value_counts())

    # Scale numeric columns
    scaler = StandardScaler()
    X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val.loc[:, numeric_cols] = scaler.transform(X_val[numeric_cols])

    # Convert categorical columns to numeric codes for ADASYN (CatBoost can handle categories directly,
    # but ADASYN requires numeric arrays)
    X_train_adasyn = X_train.copy()
    X_val_transformed = X_val.copy()
    for col in categorical_cols:
        X_train_adasyn[col] = X_train_adasyn[col].astype('category').cat.codes
        X_val_transformed[col] = X_val_transformed[col].astype('category').cat.codes

    # ---------------------
    # Save Non-Synthesized Data as CSVs
    # ---------------------
    data_folder = os.path.join(experiment_folder, "data")
    X_train.to_csv(os.path.join(data_folder, "X_train_original.csv"), index=False)
    pd.Series(y_train, name="Target").to_csv(os.path.join(data_folder, "y_train_original.csv"), index=False)
    X_val.to_csv(os.path.join(data_folder, "X_test.csv"), index=False)
    pd.Series(y_val, name="Target").to_csv(os.path.join(data_folder, "y_test.csv"), index=False)

    # ---------------------
    # Apply ADASYN Oversampling
    # ---------------------
    print("[INFO] Applying ADASYN oversampling on training data...")
    counts = y_train.value_counts()
    majority_count = counts.max()
    desired_count = int(OVERSAMPLE_FACTOR * majority_count)
    sampling_strategy_dict = {cls: desired_count for cls in counts.index}
    print("[INFO] ADASYN sampling strategy dictionary (target count for each class):")
    print(sampling_strategy_dict)

    adasyn = ADASYN(n_neighbors=ADASYN_N_NEIGHBORS,
                    sampling_strategy=sampling_strategy_dict,
                    random_state=RANDOM_STATE)
    X_train_res, y_train_res = adasyn.fit_resample(X_train_adasyn, y_train)
    print(f"[INFO] Training data size after ADASYN: {X_train_res.shape[0]} samples")
    print("[INFO] Distribution in training data after ADASYN:")
    print(pd.Series(y_train_res).value_counts())

    # Save the oversampled training data
    pd.DataFrame(X_train_res, columns=X_train_adasyn.columns).to_csv(
        os.path.join(data_folder, "X_train_adasyn.csv"),
        index=False
    )
    pd.Series(y_train_res, name="Target").to_csv(os.path.join(data_folder, "y_train_adasyn.csv"), index=False)

    # ---------------------
    # Train Models
    # ---------------------
    print("[INFO] Training default (out-of-the-box) CatBoost classifier on oversampled data...")
    base_model = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
    start_time_base = time.time()
    base_model.fit(X_train_res, y_train_res)
    base_training_time = time.time() - start_time_base

    y_pred_base = base_model.predict(X_val_transformed)
    base_accuracy = accuracy_score(y_val, y_pred_base)
    base_report = classification_report(y_val, y_pred_base)
    print(f"[RESULT - Base] Accuracy: {base_accuracy:.4f}")
    print("[RESULT - Base] Classification Report:\n", base_report)

    # ---------------------
    # Bayesian Optimization
    # ---------------------
    print("[INFO] Starting Bayesian Optimization for CatBoost hyperparameters...")
    base_estimator = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
    bayes_search = BayesSearchCV(
        estimator=base_estimator,
        search_spaces=bayesian_search_spaces,
        n_iter=N_BAYES_ITER,
        cv=CV_FOLDS,
        scoring=SCORING,
        random_state=RANDOM_STATE,
        verbose=100
    )
    start_time_bayes = time.time()
    bayes_search.fit(X_train_res, y_train_res)
    bayes_training_time = time.time() - start_time_bayes
    bayes_model = bayes_search.best_estimator_

    print("[INFO] Best hyperparameters found:")
    print(bayes_search.best_params_)

    y_pred_bayes = bayes_model.predict(X_val_transformed)
    bayes_accuracy = accuracy_score(y_val, y_pred_bayes)
    bayes_report = classification_report(y_val, y_pred_bayes)
    print(f"[RESULT - Bayes] Accuracy: {bayes_accuracy:.4f}")
    print("[RESULT - Bayes] Classification Report:\n", bayes_report)

    # ---------------------
    # Compare & Finalize
    # ---------------------
    if base_accuracy >= bayes_accuracy:
        best_model_final = base_model
        winner = "Base (Out-of-the-box)"
        final_accuracy = base_accuracy
        final_report = base_report
        final_training_time = base_training_time
    else:
        best_model_final = bayes_model
        winner = "Bayesian Optimized"
        final_accuracy = bayes_accuracy
        final_report = bayes_report
        final_training_time = bayes_training_time

    print(f"[INFO] Final chosen model: {winner} with accuracy {final_accuracy:.4f}")

    # ---------------------
    # Confusion Matrix of Final Best Model
    # ---------------------
    y_pred_final = best_model_final.predict(X_val_transformed)
    cm = confusion_matrix(y_val, y_pred_final, labels=class_labels)
    custom_cmap = ListedColormap(custom_palette)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=custom_cmap,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Final Best Model Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.tight_layout()

    cm_png_path = os.path.join(
        experiment_folder, "confusion_matrices",
        f"final_confusion_matrix_{timestamp}_cat{n_categories}.png"
    )
    cm_eps_path = os.path.join(
        experiment_folder, "confusion_matrices",
        f"final_confusion_matrix_{timestamp}_cat{n_categories}.eps"
    )
    plt.savefig(cm_png_path)
    plt.savefig(cm_eps_path, format='eps')
    plt.clf()
    plt.close()

    print(f"[INFO] Final confusion matrix saved as {cm_png_path} and {cm_eps_path}.")

    # ---------------------
    # Feature Importance
    # ---------------------
    feature_importances = best_model_final.get_feature_importance()
    feature_names = X.columns  # original feature names
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances / feature_importances.sum()
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette=custom_palette)
    plt.title("Final Model Feature Importance", fontsize=16)
    plt.xlabel("Relative Importance", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.tight_layout()

    fi_png_path = os.path.join(
        experiment_folder, "feature_importance",
        f"final_feature_importance_{timestamp}_cat{n_categories}.png"
    )
    fi_eps_path = os.path.join(
        experiment_folder, "feature_importance",
        f"final_feature_importance_{timestamp}_cat{n_categories}.eps"
    )
    plt.savefig(fi_png_path, dpi=300)
    plt.savefig(fi_eps_path, format='eps')
    plt.clf()
    plt.close()

    print(f"[INFO] Final feature importance plots saved as {fi_png_path} and {fi_eps_path}.")

    # ---------------------
    # Save Model, Scaler, Column Info, and Quantiles
    # ---------------------
    model_save_name = f"catboost_classifier_{timestamp}_cat{n_categories}_{winner.replace(' ', '_')}"
    model_path = os.path.join(experiment_folder, "models", model_save_name + ".cbm")
    best_model_final.save_model(model_path)

    # Also save the model in pickle format
    with open(model_path + ".pkl", "wb") as f:
        pickle.dump(best_model_final, f)

    scaler_path = os.path.join(
        experiment_folder, "models",
        f"scaler_{timestamp}_cat{n_categories}_{winner.replace(' ', '_')}.pkl"
    )
    with open(scaler_path, "wb") as sf:
        pickle.dump(scaler, sf)

    # Save numeric columns
    scaler_cols_path = os.path.join(
        experiment_folder, "models",
        f"scaler_columns_{timestamp}_cat{n_categories}_{winner.replace(' ', '_')}.pkl"
    )
    with open(scaler_cols_path, "wb") as cf:
        pickle.dump(numeric_cols, cf)

    # NEW: Save the entire feature order used by the final CatBoost training.
    # For example, we used X_train_res to train. That means the final model expects these columns in this exact order:
    catboost_training_cols_path = os.path.join(
        experiment_folder, "models",
        f"catboost_training_columns_{timestamp}_cat{n_categories}_{winner.replace(' ', '_')}.pkl"
    )
    # The model was trained on X_train_res columns, so let's store that order:
    with open(catboost_training_cols_path, "wb") as cf2:
        pickle.dump(list(X_train_res.columns), cf2)

    quantiles_path = os.path.join(
        experiment_folder, "models",
        f"quantiles_{timestamp}_cat{n_categories}_{winner.replace(' ', '_')}.pkl"
    )
    with open(quantiles_path, "wb") as qf:
        pickle.dump(quantiles, qf)

    # ---------------------
    # Save Logs
    # ---------------------
    if n_categories == 2:
        thresh_info = f"Median: {median:.2f}"
    elif n_categories == 3:
        thresh_info = f"33%: {q1:.2f}, 66%: {q2:.2f}"
    elif n_categories == 4:
        thresh_info = f"25%: {q1:.2f}, 50%: {q2:.2f}, 75%: {q3:.2f}"
    elif n_categories == 33:
        thresh_info = f"20%: {q_low:.2f}, 80%: {q_high:.2f}"
    else:
        thresh_info = f"20%: {q1:.2f}, 40%: {q2:.2f}, 60%: {q3:.2f}, 80%: {q4:.2f}"

    results_content = (
        f"=== Out-of-the-box Model ===\n"
        f"Accuracy: {base_accuracy:.4f}\n"
        f"Training Time: {base_training_time:.2f} seconds\n"
        f"Classification Report:\n{base_report}\n\n"
        f"=== Bayesian Optimized Model ===\n"
        f"Accuracy: {bayes_accuracy:.4f}\n"
        f"Training Time: {bayes_training_time:.2f} seconds\n"
        f"Best Hyperparameters:\n{bayes_search.best_params_}\n"
        f"Classification Report:\n{bayes_report}\n\n"
        f"=== Final Best Model ===\n"
        f"Winner: {winner}\n"
        f"Accuracy: {final_accuracy:.4f}\n"
        f"Training Time: {final_training_time:.2f} seconds\n"
        f"Classification Report:\n{final_report}\n"
        f"Thresholds used: {thresh_info}\n"
        f"Number of categories: {n_categories}\n"
    )

    results_path = os.path.join(
        experiment_folder, "result_txts",
        f"results_{timestamp}_cat{n_categories}.txt"
    )
    with open(results_path, 'w') as f:
        f.write(results_content)

    print("[INFO] Experiment completed. Final results:")
    print(results_content)
    print("[INFO] All outputs have been saved in folder:")
    print(experiment_folder)


if __name__ == "__main__":
    main()
