# Analyzing the Impact of Nitrogen Fertilization on Grassland Biomass: A Machine Learning Approach

Author: Dr. techn. Sebastian Raubitzek

## Overview

This repository implements a multi‑stage machine learning pipeline to predict grassland biomass yield from detailed management records collected by the Biodiversity Exploratories (Germany, 2009–2016). We combine plot‑level dry‐matter measurements with 116 management variables—including mowing frequency, grazing pressure, and mineral & organic nitrogen inputs—to train CatBoost classifiers optimized via Bayesian search and balanced with ADASYN oversampling.

The grassland data is obtained from the supplementary material of https://bdj.pensoft.net/article/36387/element/4/5226558// . The .txt file can be download directly from the homepage of the publication.
The bioamss data can be obtained from https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/1365-2745.14288 and the data can be found here: https://www.bexis.uni-jena.de/ddm/data/Showdata/31448 .

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
