# Exploratory Data Analysis & Preprocessing

**Notebook:** `notebooks/dataset_analysis.ipynb`

## Overview
This notebook establishes the foundational preprocessing pipeline for the FunaDB dataset. Since this is a clinical/educational dataset measuring task scores and response times, careful consideration is given to missing values, sentinels, and class-specific distributions before the data is passed to the ML models.

## Key Steps

### 1. Data Cleaning & Univariate Analysis
- **Sentinel Values:** Replaces known invalid sentinels with NaNs.
- **Numeric Safety:** Enforces strict numeric safety; task scores cannot be negative.
- **Distributions:** Maps out histograms and kernel density estimates to understand the skewness and kurtosis of each raw feature (NC, DM, NS, ADD, SUB, CA).

### 2. Missingness Analysis & Imputation
- **Informative Missingness:** Missing scores in educational datasets are rarely random. An absent student or a timed-out task is a diagnostic signal. The notebook analyzes whether missingness is disproportionately concentrated in the At-Risk class.
- **Indicator Flags:** To preserve the diagnostic signal of absence, binary flags (`{feature}_incomplete`) are generated before any imputation occurs.
- **Per-Class Median Imputation:** NaNs are imputed dynamically using the median *of the respective class*. This group-aware imputation prevents bleeding signals across the At-Risk and Typical boundaries.

### 3. Outlier Handling (Per-Class IQR Clipping)
- Outliers are clipped using the Interquartile Range (IQR) method. 
- Crucially, this clipping is done **within each class separately**. Global clipping would corrupt the genuinely different, extreme score ranges exhibited by At-Risk students.

### 4. Correlation & Multivariate Interactions
- **Correlation Matrix:** Quantifies collinearity among features, specifically highlighting the division between Number Processing constructs (NC, DM) and Arithmetic Fluency constructs (NS, ADD, SUB, CA).
- **Pairplots:** Visualizes bivariate relationships with class-specific color mappings to identify clear decision boundaries prior to normalization.

## Output
Produces the `datasets/processed/cleaned_dataset.csv` artifact, ready for synthetic data generation and modeling.
