# Synthetic Minority Oversampling

**Notebook:** `notebooks/synthetic_data_generation.ipynb`  
**Primary outputs:** `datasets/processed/train.csv`, `val.csv`, `test.csv`, `s_train.csv`, and deployment CSV counterparts

## Overview

The synthetic-data notebook creates minority-class at-risk samples for the TSTR pipeline. It keeps validation and test data fully real, generates synthetic rows from the training minority cohort only, and exports both research CSVs and deployment CSVs.

The current run produces 58 synthetic class-1 rows, balancing the real training split from 154 typical / 96 at-risk to 154 typical / 154 at-risk in TSTR mode.

## Pipeline

### 1. Stratified Real Split

The cleaned dataset is split 70/15/15 into real train, validation, and test partitions:

| Split | Rows | Class Distribution |
|---|---:|---|
| Train | 250 | 154 typical / 96 at-risk |
| Validation | 54 | 33 typical / 21 at-risk |
| Test | 54 | 33 typical / 21 at-risk |

Synthetic generation is restricted to the training split to avoid leakage.

### 2. GAN Candidate Generation

The notebook trains both `CopulaGANSynthesizer` and `CTGANSynthesizer` candidates using SDV. Candidate generation uses raw task features plus `BC` so the generator can better model the ADD/SUB/CA relationship.

The search explores multiplier, epoch, batch-size, generator learning-rate, and discriminator learning-rate combinations. Candidate pools are filtered so the final selected samples are statistically close to the real at-risk training distribution.

### 3. Statistical Validation

Synthetic candidates are checked feature-by-feature against the real minority cohort using:

| Check | Purpose |
|---|---|
| Two-sample KS test | Rejects distributions that significantly differ from real at-risk data |
| Jensen-Shannon Divergence | Bounds histogram-level distribution drift |
| Mean / median / std deltas | Controls central tendency and spread |
| Skewness / kurtosis deltas | Controls distribution shape |
| Correlation diagnostics | Checks multivariate structure |

The notebook also generates visual diagnostics such as overlaid histograms, Q-Q plots, and correlation heatmaps.

### 4. Derived Features

Derived diagnostic features are recomputed deterministically after synthetic raw features are selected:

| Feature | Formula |
|---|---|
| `NP` | `NC + DM` |
| `SN` | `NC - DM` |
| `AF` | `(ADD + SUB + CA) / 3` |
| `BC` | `CA - AF` |
| `AS` | `ADD - SUB` |
| `PF` | `AF / NP` |

This keeps the deployment feature math exact instead of asking the GAN to learn deterministic relationships.

### 5. Missingness Injection

The notebook reads empirical per-class missing rates from `outputs/logs_and_metrics/missing_rates.json`. For synthetic class-1 rows, it applies pseudo-random missing masks that mirror the at-risk cohort, imputes affected values with real minority medians, and sets `{feature}_incomplete` flags.

Current at-risk missing rates:

| Feature | Missing Rate |
|---|---:|
| NC | 0.014 |
| DM | 0.014 |
| NS | 0.036 |
| ADD | 0.029 |
| SUB | 0.029 |
| CA | 0.087 |

The current synthetic run reports 9 synthetic rows with at least one injected incomplete flag.

## Exported Datasets

| File | Columns | Purpose |
|---|---|---|
| `train.csv`, `val.csv`, `test.csv` | Raw features, label, derived features, incomplete flags, `is_synthetic` where applicable | Research/evaluation |
| `s_train.csv` | Synthetic at-risk rows with research columns | TSTR augmentation |
| `train_deployment.csv`, `val_deployment.csv`, `test_deployment.csv` | Raw features, label, derived features | Deployment training/evaluation |
| `s_train_deployment.csv` | Synthetic at-risk rows with deployment columns | TSTR deployment training |

Deployment CSVs drop incomplete flags because the TSTR evaluation found that flags had 0 split importance in the final tree.
