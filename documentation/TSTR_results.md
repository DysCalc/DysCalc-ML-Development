# TSTR vs. TRTR Evaluation Report

**Model:** Custom C4.5 Decision Tree  
**Dataset:** FunaDB Dyscalculia Screening  
**Evaluation notebook:** `notebooks/tstr_vs_trtr.ipynb`  
**Split:** 70/15/15 stratified real-data split  
**CV:** 5-fold stratified CV on real rows, with synthetic rows appended only to TSTR training folds

---

## Overview

This report summarizes the current Train-on-Synthetic Test-on-Real (TSTR) evaluation against a Train-on-Real Test-on-Real (TRTR) baseline. The benchmark uses raw tree probabilities directly, without probability calibration, and selects model settings through a validation grid search followed by CV tie-breaking.

The current repo writes the full search artifacts to `outputs/grid_search/` and visualization summaries to `outputs/figures/`.

---

## Dataset Composition

| Split | Rows | Class Distribution | Composition |
|---|---:|---|---|
| TRTR train | 250 | 154 typical / 96 at-risk | Real only |
| TSTR train | 308 | 154 typical / 154 at-risk | 250 real + 58 synthetic at-risk |
| Validation | 54 | 33 typical / 21 at-risk | Real only |
| Test | 54 | 33 typical / 21 at-risk | Real only |

The 58 synthetic rows are class-1 at-risk samples. During TSTR cross-validation, they are pinned to every training fold and never placed in validation folds.

---

## Hyperparameter and Threshold Search

The notebook searches:

| Parameter | Values |
|---|---|
| `conf_fact` | 0.10 to 0.50 in 0.05 increments |
| `min_samples_leaf` | 10 to 50 |
| `max_depth` | 5 to 15 |
| `threshold` | 0.35 to 0.75 in 0.05 increments |

Each condition evaluates 4,059 hyperparameter combinations across 9 thresholds, producing 36,531 validation evaluations per condition.

### Best Validation Results

| Condition | F2 | Recall | Precision | Threshold | Params |
|---|---:|---:|---:|---:|---|
| TRTR | 0.8108 | 0.8571 | 0.6667 | 0.35 | `conf_fact=0.25`, `min_samples_leaf=10`, `max_depth=5` |
| TSTR | 0.8491 | 0.8571 | 0.8182 | 0.35-0.60 tied | validation-best ties include several settings |

Because many TSTR candidates tie on validation F2, the notebook resolves tied candidates using cross-validation.

### CV Tie Resolution

| Condition | Selected Threshold | Selected Params | Tie Candidates |
|---|---:|---|---:|
| TRTR | 0.35 | `conf_fact=0.50`, `min_samples_leaf=10`, `max_depth=6` | 42 |
| TSTR | 0.40 | `conf_fact=0.40`, `min_samples_leaf=11`, `max_depth=7` | 948 |

The deployment script currently locks TSTR to the same tie-resolved settings. For TRTR mode, `scripts/train.py` uses the validation-best settings (`conf_fact=0.25`, `min_samples_leaf=10`, `max_depth=5`, threshold `0.35`).

---

## Cross-Validation Results

| Metric | TRTR mean +/- std | TSTR mean +/- std | Delta |
|---|---:|---:|---:|
| Recall | 0.6568 +/- 0.1416 | 0.6663 +/- 0.0985 | +0.0095 |
| Precision | 0.6238 +/- 0.0516 | 0.5738 +/- 0.0662 | -0.0500 |
| F1-Score | 0.6325 +/- 0.0822 | 0.6091 +/- 0.0480 | -0.0234 |
| F2-Score | 0.6452 +/- 0.1162 | 0.6402 +/- 0.0725 | -0.0050 |
| Accuracy | 0.7160 +/- 0.0388 | 0.6720 +/- 0.0601 | -0.0440 |

TSTR slightly improves mean recall and lowers recall variance, but the CV mean F2 is essentially tied with TRTR and slightly lower by 0.0050.

---

## Fold Variance

| Fold | TRTR Recall | TSTR Recall |
|---:|---:|---:|
| 1 | 0.6000 | 0.7000 |
| 2 | 0.6842 | 0.7895 |
| 3 | 0.7368 | 0.5263 |
| 4 | 0.4211 | 0.5789 |
| 5 | 0.8421 | 0.7368 |
| Range | 0.4211 - 0.8421 | 0.5263 - 0.7895 |

TSTR raises the worst-fold recall from 0.4211 to 0.5263 and reduces fold-to-fold spread.

---

## Held-Out Test Results

| Metric | TRTR | TSTR | Delta |
|---|---:|---:|---:|
| Recall | 0.5714 | 0.6667 | +0.0952 |
| Precision | 0.7059 | 0.6087 | -0.0972 |
| F1-Score | 0.6316 | 0.6364 | +0.0048 |
| F2-Score | 0.5941 | 0.6542 | +0.0601 |
| Accuracy | 0.7407 | 0.7037 | -0.0370 |

### CV-to-Test Drift

| Condition | Recall Drift | F2 Drift |
|---|---:|---:|
| TRTR | -0.0854 | -0.0511 |
| TSTR | +0.0004 | +0.0140 |

On the held-out real test set, TSTR improves recall and F2 while trading off precision and accuracy. This is the expected screening-oriented behavior: the model catches more at-risk students while allowing more false positives.

---

## Global Feature Importance

The final TSTR tree uses only raw task features as split drivers. Incomplete flags were evaluated during research, but their split importance was 0 and they are dropped from deployment CSVs.

| Feature | Importance | Domain |
|---|---:|---|
| NC | 0.4928 | Number Comparison |
| NS | 0.2584 | Number Series |
| ADD | 0.1876 | Single-Digit Addition |
| SUB | 0.0612 | Single-Digit Subtraction |
| DM | 0.0000 | Digit-Dot Matching |
| CA | 0.0000 | Multi-Digit Addition and Subtraction |

---

## Generated Artifacts

| Artifact | Purpose |
|---|---|
| `outputs/grid_search/trtr_grid_search_results.csv` | Full TRTR validation grid search |
| `outputs/grid_search/tstr_grid_search_results.csv` | Full TSTR validation grid search |
| `outputs/grid_search/trtr_validation_tie_cv_results.csv` | CV results for tied TRTR validation candidates |
| `outputs/grid_search/tstr_validation_tie_cv_results.csv` | CV results for tied TSTR validation candidates |
| `outputs/figures/analysis_report.txt` | Compact summary generated by `scripts/plot_tstr_vs_trtr.py` |
| `outputs/figures/*.png` | Performance, CV distribution, and hyperparameter trend charts |
