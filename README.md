# Grassland Biomass Classification Pipeline

## Overview
This repository provides a complete pipeline for processing, exploring, and classifying grassland management data from the Biodiversity‑Exploratories project (2006–2016). Starting from the raw TXT survey data, it:
1. Converts and cleans the raw data.  
2. Merges grassland management with biomass measurements.  
3. Performs initial exploratory data analysis.  
4. Trains CatBoost classifiers—both out‑of‑the‑box and Bayesian‑optimized—using ADASYN for class balancing.  
5. Validates the final models, generating reports, confusion matrices, and feature‑importance visualizations.  

## Features
- **Data Conversion & Cleaning:** `00_extract_grassland.py`  
- **Merge & Missing‑Value Handling:** `01_merge_GrasslandBiomass_Ncomp.py` with five configurable strategies  
- **Initial EDA:** `02_initial_evaluation.py` — quantile‑based biomass categorization, boxplots, full textual report  
- **Model Training:** `03_train_CatBoostBayesianOptOotB.py` — ADASYN oversampling + CatBoost + Bayesian hyperparameter search  
- **Model Validation:** `04_validate_CatBoostClassification.py` — loads experiment artifacts, reindexes features, prints & saves all reports  

## Repository Structure
 Grassland_Biomass_Classification

├── 00_extract_grassland.py # TXT→CSV + clean/extract subset

├── 01_merge_GrasslandBiomass_Ncomp.py # Merge & missing‑value options

├── 02_initial_evaluation.py # Exploratory data analysis

├── 03_train_CatBoostBayesianOptOotB.py# ADASYN + CatBoost + Bayesian search

├── 04_validate_CatBoostClassification.py # Load & evaluate experiments

└── README.md # Project documentation
