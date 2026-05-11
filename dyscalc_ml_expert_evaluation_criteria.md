# DysCalc ML and Data Science Expert Evaluation Criteria

**Repository under review:** `DysCalc/DysCalc-ML-Development`  
**Thesis proposal basis:** *DysCalc: An AI-Assisted Learning Platform for Screening and Targeted Learning for Students with Potential Dyscalculia in the Philippine Educational Context*  
**Evaluation purpose:** independent expert validation of the repository's data cleaning, exploratory data analysis, machine learning implementation, model evaluation, and reproducibility.  
**Recommended evaluator profile:** data scientist, machine learning researcher/practitioner, educational data mining researcher, psychometrics-informed ML reviewer, or applied AI reviewer with experience in tabular classification, imbalanced data, model validation, and reproducible analysis.

---

## 1. Evaluation Context

DysCalc uses the public FUNA-DB dataset to build a proof-of-concept machine learning pipeline for early risk screening of potential dyscalculia indicators. The repository covers dataset preprocessing, RMAT-based screening label derivation, exploratory data analysis, synthetic minority augmentation, C4.5-style decision tree development, model evaluation, and saved model artifacts.

This evaluation must treat the repository as a **machine learning screening pipeline**, not as a clinical diagnostic system. The expert reviewer should assess whether the data cleaning and analysis are technically sound, whether the label construction is defensible, whether the model implementation is correct and reproducible, and whether the reported performance claims are supported by the evidence in the repository.

---

## 2. Review Scope

### Included in this Expert Review

- FUNA-DB dataset provenance, cleaning, preprocessing, and documentation.
- Exploratory data analysis, descriptive statistics, missingness analysis, and class balance analysis.
- RMAT-based `At-Risk` / `Typical` label derivation.
- Feature engineering and leakage control.
- C4.5 decision tree implementation and model packaging.
- Train/validation/test design, cross-validation, hyperparameter tuning, and threshold selection.
- Metric choice and interpretation for screening-oriented classification.
- Synthetic minority augmentation and synthetic-augmented-versus-real-only evaluation.
- Model interpretability outputs, including confidence, decision paths, feature importance, task importance, and domain severity scores.
- Reproducibility, auditability, code quality, and repository organization.

### Outside Primary Scope

- Clinical validation of dyscalculia diagnosis.
- Local learner pilot testing in Philippine schools.
- Full web application usability evaluation.
- Pedagogical intervention or classroom activity review.
- Security, database, and production infrastructure review.

---

## 3. Proposal Traceability Matrix

The reviewer should check that the repository supports the thesis proposal objectives relevant to data science and ML.

| Proposal Objective | Repository Evidence to Review | Related Rubric Sections |
|---|---|---|
| Pre-process and curate FUNA-DB features aligned with dyscalculia indicators. | Raw and processed datasets, dataset analysis notebook, cleaning documentation, missing-rate logs. | B, C |
| Construct `At-Risk` and `Typical` labels from RMAT using percentile/psychometric thresholds. | `scripts/RMAT_Labeling.py`, labeled dataset, README, label documentation. | D |
| Train and evaluate an ML model for potential dyscalculia risk classification. | `src/C45DecisionTree.py`, `scripts/train.py`, grid-search outputs, metrics logs, saved model. | F, G, H |
| Provide interpretable, feature-based numeracy insights. | Diagnostic dataclasses, decision path outputs, feature importance, severity/task-importance logic, documentation. | E, I |
| Explore class-balanced training through synthetic augmentation. | Synthetic generation notebook, synthetic-data documentation, synthetic-augmented evaluation report, generated CSVs. | J |
| Ensure reproducible data science workflow and repository organization. | Requirements, scripts, notebooks, outputs, model artifacts, documentation. | K |

---

## 4. Rating Scale

Use the following 0-4 scale for each criterion.

| Score | Interpretation |
|---:|---|
| 0 | Not present, incorrect, unverifiable, or contradicted by repository evidence |
| 1 | Present but weak, incomplete, poorly justified, or methodologically questionable |
| 2 | Adequate for a prototype, but requires correction, clearer evidence, or stronger justification |
| 3 | Good and mostly defensible for thesis-level proof-of-concept validation |
| 4 | Strong, reproducible, well-justified, and technically defensible for the stated thesis scope |

Each criterion has an assigned weight. The weighted score is computed as:

```text
Weighted Score = (Criterion Score / 4) x Criterion Weight
```

Total possible score: **100 points**.

---

## 5. Minimum Passing Gates

These checks are non-negotiable. If any gate fails, the reviewer should mark the repository as **Not Yet Validated**, even if the weighted score is acceptable.

| Gate | Required Condition | Pass/Fail | Evidence / Comments |
|---|---|---|---|
| G1. RMAT leakage prevention | RMAT or RMAT-derived values must be used only for label construction and must not be used directly or indirectly as model input features. |  |  |
| G2. Real held-out evaluation | Validation and test sets must contain only real FUNA-DB rows; synthetic rows must not appear in validation or test evaluation. |  |  |
| G3. Fixed split and tuning policy | Train/validation/test splits must be documented and reproducible; threshold and hyperparameter choices must not be selected on the held-out test set. |  |  |
| G4. Screening-only claim | Repository language must describe outputs as screening-risk indicators, not clinical dyscalculia diagnosis. |  |  |
| G5. Synthetic augmentation honesty | Claims about synthetic augmentation must report both benefits and trade-offs, especially recall/F2 gains versus precision, accuracy, AUC, or RMSE losses. |  |  |
| G6. Synthetic-training terminology accuracy | If the implementation uses real + synthetic training, it must be described as synthetic-augmented training rather than pure TSTR unless the distinction is explicitly explained. |  |  |
| G7. Reproducible model selection | The final selected model, threshold, and hyperparameters must be traceable to validation and cross-validation evidence. |  |  |
| G8. Non-local dataset limitation | The repository must acknowledge that FUNA-DB is non-local secondary data and that deployment to Philippine learners requires future local validation. |  |  |

---

## 6. Weighted Evaluation Rubric

### A. Research Alignment and Scope Control - 5 pts

| Criterion | Weight | Score | Evidence / Comments |
|---|---:|---:|---|
| The repository clearly supports the thesis proposal's ML purpose: early, risk-based screening using FUNA-DB-derived numeracy features. | 2 |  |  |
| The repository consistently frames the model as a screening aid rather than a diagnostic authority. | 2 |  |  |
| Claims about Filipino learners, classroom use, and deployment are limited to proof-of-concept or future-validation language. | 1 |  |  |

**Reviewer notes:** Check whether words such as "diagnosis," "dyscalculic," "confirmed," or "validated for deployment" are used carefully. The proposal permits screening-risk classification, not clinical diagnosis.

---

### B. Dataset Provenance and Data Management - 8 pts

| Criterion | Weight | Score | Evidence / Comments |
|---|---:|---:|---|
| The raw FUNA-DB source file is identified, preserved, and separated from processed datasets. | 2 |  |  |
| The dataset-selection rationale matches the proposal: public, anonymized, psychometrically validated, and appropriate for secondary data analysis. | 2 |  |  |
| Dataset versions are traceable from raw data to labeled, cleaned, split, synthetic, and deployment-ready CSVs. | 2 |  |  |
| Data artifacts are named and organized clearly enough for an external reviewer to audit the workflow. | 1 |  |  |
| Processed datasets contain only intended columns for their stated purpose. | 1 |  |  |

**Expected evidence:**  
`datasets/raw/`, `datasets/processed/`, `README.md`, `documentation/dataset_analysis.md`

---

### C. Data Cleaning, Missingness Handling, and EDA - 14 pts

| Criterion | Weight | Score | Evidence / Comments |
|---|---:|---:|---|
| Cleaning rules are documented, including handling of `-99` sentinel values, impossible values, invalid values, and type coercion. | 2 |  |  |
| Missingness handling is justified and traceable, including incomplete flags, imputation policy, and missing-rate artifacts. | 3 |  |  |
| Outlier handling is documented and justified, including any class-specific clipping or distribution-preserving decisions. | 2 |  |  |
| Exploratory data analysis reports descriptive statistics for the raw task features and final modeling features. | 2 |  |  |
| EDA includes class balance checks before and after labeling, splitting, and synthetic augmentation. | 2 |  |  |
| Distributional and correlation analyses are sufficient to support modeling and synthetic-data decisions. | 2 |  |  |
| The analysis identifies data limitations that may affect model validity or generalizability. | 1 |  |  |

**Expected evidence:**  
`notebooks/dataset_analysis.ipynb`, `documentation/dataset_analysis.md`, `outputs/logs_and_metrics/missing_rates.json`, `outputs/figures/analysis_report.txt`

---

### D. Label Construction Validity and Leakage Control - 12 pts

| Criterion | Weight | Score | Evidence / Comments |
|---|---:|---:|---|
| The `At-Risk` / `Typical` label derivation from RMAT is clearly documented and matches the proposal's percentile-based screening rationale. | 3 |  |  |
| The RMAT threshold is justified as a screening threshold for elevated risk, not as a clinical diagnostic cutoff. | 2 |  |  |
| RMAT and RMAT-derived values are removed from all predictive feature sets after label creation. | 3 |  |  |
| Label construction is reproducible through a script or notebook and produces the documented class distribution. | 2 |  |  |
| Documentation discusses how the chosen threshold affects class balance, sensitivity, precision, and interpretation. | 2 |  |  |

**Reviewer notes:** Inspect `scripts/RMAT_Labeling.py` and the labeled dataset. Confirm whether the documented 35th-percentile or z-score rule is implemented exactly and whether any proposal/documentation mismatch is explained.

---

### E. Feature Engineering and Construct Validity - 10 pts

| Criterion | Weight | Score | Evidence / Comments |
|---|---:|---:|---|
| Raw predictive features are limited to intended FUNA-DB T1 task measures such as `NC`, `DM`, `NS`, `ADD`, `SUB`, and `CA`. | 2 |  |  |
| Derived features such as `NP`, `SN`, `AF`, `BC`, `AS`, and `PF` are mathematically defined, reproducible, and consistently implemented. | 2 |  |  |
| The repository clearly separates features used for tree splitting from features used only for interpretation, if those sets differ. | 2 |  |  |
| Feature transformations, imputation, clipping, and scaling are fitted or computed without validation/test leakage. | 2 |  |  |
| Feature choices are linked to numeracy constructs from the proposal, including number processing, symbolic/non-symbolic processing, arithmetic fluency, and operation-specific weaknesses. | 2 |  |  |

**Reviewer notes:** Verify feature formulas in the README, notebooks, scripts, and generated CSVs. Any mismatch with the proposal formulas should be documented and justified.

---

### F. C4.5 Model Implementation Quality - 12 pts

| Criterion | Weight | Score | Evidence / Comments |
|---|---:|---:|---|
| The custom decision tree correctly applies gain-ratio-based splitting for continuous tabular features. | 2 |  |  |
| Threshold candidate generation and split selection are technically sound and documented. | 2 |  |  |
| Pruning behavior is implemented and documented, including the meaning and effect of `conf_fact`. | 2 |  |  |
| Probability estimation is clearly defined, including Laplace smoothing and leaf-distribution behavior. | 2 |  |  |
| The implementation handles edge cases such as empty splits, single-class leaves, unseen values, missing values, shallow/deep trees, and invalid thresholds. | 2 |  |  |
| The model can be saved, loaded, and reused reproducibly for inference. | 1 |  |  |
| The code is modular enough for expert inspection, testing, and future comparison with baseline models. | 1 |  |  |

**Expected evidence:**  
`src/C45DecisionTree.py`, `src/Dataclasses.py`, `scripts/train.py`, `documentation/C45_decision_tree.md`, `models/v1.pkl`

---

### G. Training, Validation, Testing, and Model Selection - 14 pts

| Criterion | Weight | Score | Evidence / Comments |
|---|---:|---:|---|
| Train/validation/test splits are stratified, fixed before model selection, and reproducible through saved datasets or seeded code. | 2 |  |  |
| Validation data are used for hyperparameter and threshold selection, while the held-out test set is reserved for final evaluation only. | 3 |  |  |
| Cross-validation is stratified and reports mean plus standard deviation for relevant metrics. | 2 |  |  |
| The hyperparameter search space is documented and reasonable for the 358-row dataset and the C4.5 model. | 2 |  |  |
| Thresholded and non-thresholded evaluations are distinguished clearly, including what probability source is thresholded. | 2 |  |  |
| Final deployment settings match the documented selected model settings, threshold, and feature set. | 2 |  |  |
| Logs, grid-search outputs, figures, and saved artifacts are sufficient to audit how the final model was selected. | 1 |  |  |

**Reviewer notes:** Flag any test-set-driven thresholding or hyperparameter selection. Also check whether any stated minimum recall target is achieved; if not, repository claims should acknowledge the gap instead of implying that the target was met.

---

### H. Metric Selection and Performance Interpretation - 10 pts

| Criterion | Weight | Score | Evidence / Comments |
|---|---:|---:|---|
| Recall is prioritized appropriately because false negatives may delay support or further assessment. | 2 |  |  |
| Precision is interpreted as a safeguard against excessive over-identification. | 2 |  |  |
| F1 and F2 are reported and interpreted correctly, especially the stronger recall emphasis of F2. | 2 |  |  |
| Accuracy is not overemphasized under class imbalance. | 1 |  |  |
| AUC-ROC is interpreted as threshold-independent discrimination rather than proof of deployment readiness. | 1 |  |  |
| RMSE is interpreted cautiously as a probability/calibration-style diagnostic metric, not as the main classifier metric. | 1 |  |  |
| Confusion matrices or equivalent TP/FP/TN/FN analysis are available for final held-out test results. | 1 |  |  |

**Reviewer notes:** The reviewer should assess whether the final narrative matches the results. Synthetic augmentation should be credited mainly when it improves recall/F2 or recall stability, not presented as uniformly better across all metrics.

---

### I. Model Interpretability and Diagnostic Outputs - 5 pts

| Criterion | Weight | Score | Evidence / Comments |
|---|---:|---:|---|
| Per-sample outputs include predicted class, confidence, decision path, readable path explanation, and leaf distribution. | 1 |  |  |
| Domain severity scores and task-importance scores are explainable and traceable to model paths, feature values, or documented calculations. | 2 |  |  |
| Confidence scores are explained as model/leaf probabilities or estimates, not as clinical certainty. | 1 |  |  |
| Global feature importance is documented and not confused with causal importance. | 1 |  |  |

**Reviewer notes:** Interpretability should be evaluated as part of the ML model's technical output. It should not be treated as clinical explanation or causal evidence.

---

### J. Synthetic Data Generation and Synthetic-Augmented Evaluation - 6 pts

| Criterion | Weight | Score | Evidence / Comments |
|---|---:|---:|---|
| Synthetic generation is restricted to the minority training class and does not use validation/test rows. | 2 |  |  |
| Synthetic data quality is evaluated through distributional and multivariate checks such as KS tests, JSD, summary-statistic deltas, ECDF/QQ plots, histograms, or correlation diagnostics. | 1 |  |  |
| Synthetic rows are checked for plausibility, near-duplicate risk, and memorization/leakage concerns. | 1 |  |  |
| Synthetic-augmented training is compared against real-only training using identical real validation/test splits where possible. | 2 |  |  |

**Expected evidence:**  
`notebooks/synthetic_data_generation.ipynb`, `documentation/synthetic_data_generation.md`, `notebooks/tstr_vs_trtr.ipynb`, `notebooks/tstr_vs_trtr_no_thresholding.ipynb`, `documentation/TSTR_results.md`, `outputs/grid_search/`

---

### K. Reproducibility, Code Quality, and Repository Hygiene - 4 pts

| Criterion | Weight | Score | Evidence / Comments |
|---|---:|---:|---|
| A reviewer can install dependencies and run the main scripts/notebooks from the project root with documented commands. | 1 |  |  |
| Requirements are pinned or specified well enough to recreate the environment. | 1 |  |  |
| Scripts and notebooks use random seeds or saved outputs where needed for reproducibility. | 1 |  |  |
| Outputs, figures, grid-search results, logs, and model artifacts are organized and traceable to the final conclusions. | 1 |  |  |

**Reviewer notes:** The expert should attempt a clean run if feasible and record whether reproduced metrics match documented values within reasonable tolerance.

---

## 7. Overall Scoring Summary

| Section | Weight | Score / 4 | Weighted Score |
|---|---:|---:|---:|
| A. Research Alignment and Scope Control | 5 |  |  |
| B. Dataset Provenance and Data Management | 8 |  |  |
| C. Data Cleaning, Missingness Handling, and EDA | 14 |  |  |
| D. Label Construction Validity and Leakage Control | 12 |  |  |
| E. Feature Engineering and Construct Validity | 10 |  |  |
| F. C4.5 Model Implementation Quality | 12 |  |  |
| G. Training, Validation, Testing, and Model Selection | 14 |  |  |
| H. Metric Selection and Performance Interpretation | 10 |  |  |
| I. Model Interpretability and Diagnostic Outputs | 5 |  |  |
| J. Synthetic Data Generation and Synthetic-Augmented Evaluation | 6 |  |  |
| K. Reproducibility, Code Quality, and Repository Hygiene | 4 |  |  |
| **Total** | **100** |  |  |

---

## 8. Suggested Interpretation of Final Score

| Total Score | Interpretation |
|---:|---|
| 90-100 | Strongly validated for thesis-level ML/data science claims, subject to stated proof-of-concept limitations |
| 80-89 | Validated with minor revisions |
| 70-79 | Conditionally validated; several revisions required |
| 60-69 | Weak validation; major methodological or documentation improvements required |
| Below 60 | Not validated; substantial rework required |

Passing recommendation: **at least 80/100 and all minimum passing gates satisfied**.

---

## 9. Required Reviewer Deliverables

The expert reviewer should provide:

1. Completed weighted rubric with scores and comments.
2. Pass/fail decision for each minimum passing gate.
3. Short technical validation statement suitable for thesis appendix inclusion.
4. List of required revisions separated into:
   - Critical revisions
   - Recommended revisions
   - Optional improvements
5. Final validation decision, one of:
   - Validated
   - Validated with minor revisions
   - Conditionally validated with major revisions
   - Not yet validated

---

## 10. Suggested Expert Validation Statement Template

> I reviewed the DysCalc ML Development repository, including its FUNA-DB preprocessing workflow, exploratory data analysis, RMAT-based screening label construction, feature engineering, C4.5 model implementation, training and validation procedures, synthetic augmentation analysis, model performance results, and diagnostic output design. Based on the submitted materials, I find that the repository is [validated / validated with revisions / not yet validated] for use as a thesis-level proof-of-concept ML screening pipeline, subject to the limitations that the model is not a clinical diagnostic tool, uses non-local secondary data, and requires further local validation before real educational deployment.

---

## 11. Specific Questions for the Expert Reviewer

The reviewer should explicitly answer the following:

1. Is the FUNA-DB dataset appropriate for the proposal's proof-of-concept ML screening objective?
2. Are the data cleaning, missingness handling, and exploratory analyses technically sound?
3. Is the RMAT-based label derivation acceptable and reproducible for a screening study?
4. Is there any evidence of data leakage between label construction, feature engineering, validation, testing, and synthetic augmentation?
5. Are the train/validation/test split policy and cross-validation procedures methodologically acceptable for the sample size?
6. Is the custom C4.5 implementation technically sound enough for thesis-level use?
7. Are the selected metrics appropriate for imbalanced, screening-oriented classification?
8. Does the repository honestly interpret unmet targets, such as any stated recall target?
9. Is the interpretation of synthetic-augmented training versus real-only training fair, especially given precision/recall trade-offs?
10. Are confidence scores, decision paths, domain severity scores, task-importance scores, and feature-importance scores clear enough for expert inspection?
11. Are limitations sufficiently stated to prevent diagnostic, cultural, or deployment overclaiming?
12. What must be revised before the ML work can be considered technically validated?

---

## 12. Evidence Checklist

Before the evaluation is considered complete, verify that the reviewer inspected the following components.

| Component | Reviewed? | Notes |
|---|---|---|
| `README.md` |  |  |
| `requirements.txt` |  |  |
| `scripts/RMAT_Labeling.py` |  |  |
| `scripts/train.py` |  |  |
| `src/C45DecisionTree.py` |  |  |
| `src/Dataclasses.py` |  |  |
| `notebooks/dataset_analysis.ipynb` |  |  |
| `notebooks/synthetic_data_generation.ipynb` |  |  |
| `notebooks/tstr_vs_trtr.ipynb` |  |  |
| `notebooks/tstr_vs_trtr_no_thresholding.ipynb` |  |  |
| `documentation/dataset_analysis.md` |  |  |
| `documentation/synthetic_data_generation.md` |  |  |
| `documentation/TSTR_results.md` |  |  |
| `documentation/C45_decision_tree.md` |  |  |
| `datasets/raw/FUNADB_rawdata_SUPPL.csv` |  |  |
| `datasets/processed/` |  |  |
| `outputs/grid_search/` |  |  |
| `outputs/logs_and_metrics/` |  |  |
| `outputs/figures/` |  |  |
| `models/v1.pkl` or current saved model |  |  |

---

## 13. Reviewer Metadata

| Field | Response |
|---|---|
| Reviewer name |  |
| Affiliation |  |
| Role / expertise |  |
| Highest degree or relevant certification |  |
| Date of review |  |
| Repository commit hash reviewed |  |
| Final decision |  |
| Signature |  |

---

## 14. Recommended Thesis Appendix Format

```text
Appendix X. Expert Evaluation of Data Cleaning, Data Analysis, and ML Implementation

A. Expert Reviewer Profile
B. Repository and Commit Reviewed
C. Evaluation Scope
D. Minimum Passing Gates
E. Completed Weighted Rubric
F. Reviewer Comments and Evidence Notes
G. Required Revisions
H. Final Technical Validation Statement
I. Reviewer Name, Signature, and Date
```
