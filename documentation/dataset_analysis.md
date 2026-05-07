# Exploratory Data Analysis and Preprocessing

**Notebook:** `notebooks/dataset_analysis.ipynb`  
**Primary output:** `datasets/processed/cleaned_dataset.csv`  
**Missing-rate output:** `outputs/logs_and_metrics/missing_rates.json`

## Overview

The dataset-analysis notebook prepares the labeled FunaDB dataset for synthetic generation and C4.5 evaluation. It cleans invalid values, preserves informative missingness through flags, imputes task scores, clips outliers by class, and exports a cleaned research dataset.

The current cleaned dataset has 358 rows:

| Class | Rows |
|---|---:|
| Typical (`0`) | 220 |
| At-risk (`1`) | 138 |

## Cleaned Dataset Columns

`cleaned_dataset.csv` currently contains the six raw task features, the target label, and six incomplete flags:

```text
NC, DM, NS, ADD, SUB, CA, Label,
NC_incomplete, DM_incomplete, NS_incomplete,
ADD_incomplete, SUB_incomplete, CA_incomplete
```

Derived diagnostic features are added later by `notebooks/synthetic_data_generation.ipynb`.

## Key Steps

### 1. Cleaning and Type Safety

The notebook replaces known invalid sentinel values with missing values, coerces task columns to numeric values, and prevents impossible negative task scores from entering downstream modeling.

### 2. Missingness Analysis

Missing task values are treated as potentially informative. Before imputation, the notebook creates one binary incomplete flag per raw feature:

| Raw Feature | Flag |
|---|---|
| `NC` | `NC_incomplete` |
| `DM` | `DM_incomplete` |
| `NS` | `NS_incomplete` |
| `ADD` | `ADD_incomplete` |
| `SUB` | `SUB_incomplete` |
| `CA` | `CA_incomplete` |

The empirical missing rates are exported to `outputs/logs_and_metrics/missing_rates.json` and reused by the synthetic generation notebook.

### 3. Per-Class Median Imputation

Missing values are imputed using the median for the corresponding class. This keeps imputation from blending typical and at-risk distributions.

### 4. Per-Class IQR Clipping

Outliers are clipped using class-specific IQR bounds. This preserves genuine differences between typical and at-risk score ranges while limiting extreme values that could dominate GAN training or tree splits.

### 5. Distribution and Correlation Checks

The notebook visualizes univariate distributions, class-separated patterns, and multivariate relationships across the six raw task features. These checks inform the later synthetic-data validation strategy.

## Downstream Use

`cleaned_dataset.csv` is the input to `notebooks/synthetic_data_generation.ipynb`, which creates the train/validation/test split, synthetic at-risk rows, derived diagnostic features, and deployment-ready CSVs.
