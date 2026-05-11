# Synthetic-Augmented vs. Real-Only C4.5 Evaluation Report

**Model:** Custom C4.5 Decision Tree  
**Dataset:** FUNA-DB Dyscalculia Screening  
**Split:** 70/15/15 stratified split on real data  
**Cross-validation:** 5-fold stratified CV over real rows; synthetic rows are appended only to synthetic-augmented training folds and are never used as validation/test rows.

---

## 1. Purpose of the Comparison

This report summarizes two experiments comparing a **real-only C4.5 baseline** against a **synthetic-augmented C4.5 model**.

The term **TSTR** is used in the notebooks for convenience, but the implemented setup is more precisely described as:

> **Synthetic-augmented training:** real training data + synthetic At-Risk samples, evaluated only on real validation/test data.

This is not pure TSTR in the strict sense, because the model is not trained on synthetic data alone. The real-only baseline corresponds to standard train-on-real/test-on-real evaluation.

Two prediction strategies were evaluated:

| Aspect | Experiment 1: With Thresholding | Experiment 2: Without Thresholding |
|---|---|---|
| Notebook | `tstr_vs_trtr.ipynb` | `tstr_vs_trtr_no_thresholding.ipynb` |
| Prediction rule | `predict_proba()` returns P(At-Risk) from the leaf distribution, then thresholded | Native tree class prediction via `predict()` is used directly; `predict_proba()` provides probabilities for AUC/RMSE scoring only |
| Threshold behavior | Threshold is selected on validation data | No explicit threshold search; equivalent to native majority-leaf behavior |
| Search space | 4,059 C4.5 parameter combinations × 9 thresholds = 36,531 evaluations | 4,059 C4.5 parameter combinations |
| Threshold range | 0.35 to 0.75 in increments of 0.05 | Not applicable |
| Probability source | `C45DecisionTree.predict_proba(X, positive_class=1)` — Laplace-smoothed leaf probability | Same method used for AUC-ROC and RMSE computation |

The two experiments answer different questions:

1. **With thresholding:** Does validation-based post-training threshold selection improve screening-oriented performance?
2. **Without thresholding:** Does synthetic augmentation still help when the C4.5 tree is used directly without an additional threshold-selection step?

---

## 2. Dataset Composition

| Split / Training Condition | Rows | Class Distribution | Composition |
|---|---:|---|---|
| Real-only train | 250 | 154 Typical / 96 At-Risk | Real training rows only |
| Synthetic-augmented train | 308 | 154 Typical / 154 At-Risk | 250 real rows + 58 synthetic At-Risk rows |
| Validation | 54 | 33 Typical / 21 At-Risk | Real rows only |
| Test | 54 | 33 Typical / 21 At-Risk | Real rows only |

The synthetic rows contain only class-1 At-Risk samples. During cross-validation, they are treated strictly as augmentation data: they are appended to the training portion of each fold and are never placed in validation folds.

---

## 3. Hyperparameter Search

Both experiments use the same C4.5 hyperparameter grid:

| Parameter | Values |
|---|---|
| `conf_fact` | 0.10 to 0.50 in increments of 0.05 |
| `min_samples_leaf` | 10 to 50 |
| `max_depth` | 5 to 15 |
| `threshold` | 0.35 to 0.75 in increments of 0.05; thresholded experiment only |

In addition to F2-score (the primary selection target), the grid search now records **AUC-ROC** and **RMSE** for each configuration using probabilities from `predict_proba()`. These are retained as secondary diagnostic metrics and are not used for hyperparameter selection.

The primary optimization target is **F2-score**, because the screening task prioritizes recall more strongly than precision. Recall, precision, F1-score, and accuracy are retained as secondary metrics to expose the trade-off between missed At-Risk cases and false positives.

---

# 4. Experiment 1: With Probability Thresholding

## 4.1 Validation Results

| Condition | Best Validation F2 | Recall | Precision | Threshold | Validation-Best Parameters |
|---|---:|---:|---:|---:|---|
| Real-only | 0.8108 | 0.8571 | 0.6667 | 0.35 | `cf=0.25`, `msl=10`, `md=5` |
| Synthetic-augmented | 0.8491 | 0.8571 | 0.8182 | 0.35–0.60 tied | multiple tied candidates |

The thresholded experiment produced the strongest validation result for synthetic augmentation. The synthetic-augmented model achieved the same recall as the real-only model but with substantially higher precision, increasing validation F2 from **0.8108** to **0.8491**.

## 4.2 Tie Resolution by Cross-Validation

Because multiple candidates achieved the same or near-equivalent validation performance, tied candidates were resolved using cross-validation stability.

| Condition | Selected Threshold | Selected Parameters | Number of Tie Candidates |
|---|---:|---|---:|
| Real-only | 0.35 | `cf=0.50`, `msl=10`, `md=6` | 42 |
| Synthetic-augmented | 0.40 | `cf=0.40`, `msl=11`, `md=7` | 948 |

The selected synthetic-augmented threshold is **0.40**. This means the final classifier predicts At-Risk when the tree-derived `P(At-Risk)` is at least 0.40.

## 4.3 Cross-Validation Results

| Metric | Real-only mean ± std | Synthetic-augmented mean ± std | Difference |
|---|---:|---:|---:|
| Recall | 0.6568 ± 0.1416 | 0.6663 ± 0.0985 | +0.0095 |
| Precision | 0.6238 ± 0.0516 | 0.5738 ± 0.0662 | −0.0500 |
| F1-score | 0.6325 ± 0.0822 | 0.6091 ± 0.0480 | −0.0234 |
| F2-score | 0.6452 ± 0.1162 | 0.6402 ± 0.0725 | −0.0050 |
| Accuracy | 0.7160 ± 0.0388 | 0.6720 ± 0.0601 | −0.0440 |
| AUC-ROC | 0.7383 ± 0.0829 | 0.7124 ± 0.0426 | −0.0259 |
| RMSE | 0.4418 ± 0.0349 | 0.4694 ± 0.0270 | +0.0276 |

Cross-validation shows a mixed result. Synthetic augmentation slightly improves mean recall and reduces recall variance, but it lowers precision, F1, F2, and accuracy. Therefore, the thresholded experiment should not be described as a general CV-performance improvement. Its better interpretation is:

> Thresholded synthetic augmentation improves validation-best precision/F2 and produces more stable recall, but its average CV F2 is approximately comparable to the real-only baseline.

## 4.4 Fold-Level Recall Stability

| Fold | Real-only Recall | Synthetic-augmented Recall |
|---:|---:|---:|
| 1 | 0.6000 | 0.7000 |
| 2 | 0.6842 | 0.7895 |
| 3 | 0.7368 | 0.5263 |
| 4 | 0.4211 | 0.5789 |
| 5 | 0.8421 | 0.7368 |
| **Range** | **0.4211–0.8421** | **0.5263–0.7895** |

The synthetic-augmented model narrows the recall range. Its worst fold is higher than the real-only worst fold, suggesting reduced risk of very poor At-Risk detection in some partitions.

## 4.5 Held-Out Test Results

| Metric | Real-only | Synthetic-augmented | Difference |
|---|---:|---:|---:|
| Recall | 0.5714 | 0.6667 | +0.0952 |
| Precision | 0.7059 | 0.6087 | −0.0972 |
| F1-score | 0.6316 | 0.6364 | +0.0048 |
| F2-score | 0.5941 | 0.6542 | +0.0601 |
| Accuracy | 0.7407 | 0.7037 | −0.0370 |
| AUC-ROC | 0.7641 | 0.7561 | −0.0079 |
| RMSE | 0.4317 | 0.4492 | +0.0175 |

On the held-out test set, thresholded synthetic augmentation improves recall and F2-score. The trade-off is lower precision and accuracy. This supports the synthetic-augmented model for **recall-oriented screening**, not for maximizing overall accuracy.

## 4.6 CV-to-Test Drift

| Condition | Recall Drift | F2 Drift |
|---|---:|---:|
| Real-only | −0.0854 | −0.0511 |
| Synthetic-augmented | +0.0004 | +0.0140 |

The synthetic-augmented model has nearly zero or slightly positive drift, while the real-only model drops from CV to test. This suggests that the synthetic-augmented model generalizes more consistently to the held-out test set under the thresholded setup.

---

# 5. Experiment 2: Without Thresholding

## 5.1 Validation Results

| Condition | Best Validation F2 | Recall | Precision | Validation-Best Parameters |
|---|---:|---:|---:|---|
| Real-only | 0.7798 | 0.8095 | 0.6800 | `cf=0.25`, `msl=10`, `md=5` |
| Synthetic-augmented | 0.8491 | 0.8571 | 0.8182 | `cf=0.10`, `msl=10`, `md=6` |

Even without explicit thresholding, synthetic augmentation produces a stronger validation result than the real-only model. It improves validation F2, recall, and precision.

## 5.2 Tie Resolution by Cross-Validation

| Condition | Selected Parameters | Number of Tie Candidates |
|---|---|---:|
| Real-only | `cf=0.50`, `msl=10`, `md=6` | 21 |
| Synthetic-augmented | `cf=0.25`, `msl=11`, `md=7` | 156 |

The no-threshold experiment selects a synthetic-augmented configuration with the same `min_samples_leaf` and `max_depth` as the thresholded experiment but with a lower `conf_fact`.

## 5.3 Cross-Validation Results

| Metric | Real-only mean ± std | Synthetic-augmented mean ± std | Difference |
|---|---:|---:|---:|
| Recall | 0.5942 ± 0.1410 | 0.6458 ± 0.1123 | +0.0516 |
| Precision | 0.6311 ± 0.0370 | 0.5797 ± 0.0431 | −0.0514 |
| F1-score | 0.6052 ± 0.0791 | 0.6041 ± 0.0521 | −0.0011 |
| F2-score | 0.5969 ± 0.1149 | 0.6269 ± 0.0863 | +0.0300 |
| Accuracy | 0.7120 ± 0.0325 | 0.6800 ± 0.0358 | −0.0320 |
| AUC-ROC | 0.7383 ± 0.0829 | 0.7115 ± 0.0436 | −0.0268 |
| RMSE | 0.4418 ± 0.0349 | 0.4652 ± 0.0230 | +0.0235 |

Without thresholding, synthetic augmentation improves mean recall and F2-score in cross-validation and also reduces their variability. Precision and accuracy remain lower, which is expected because the synthetic-augmented model is more sensitive to At-Risk cases.

## 5.4 Fold-Level Recall Stability

| Fold | Real-only Recall | Synthetic-augmented Recall |
|---:|---:|---:|
| 1 | 0.5500 | 0.6500 |
| 2 | 0.5263 | 0.7895 |
| 3 | 0.6316 | 0.4737 |
| 4 | 0.4211 | 0.5789 |
| 5 | 0.8421 | 0.7368 |
| **Range** | **0.4211–0.8421** | **0.4737–0.7895** |

Synthetic augmentation again narrows the recall range relative to the real-only baseline. However, the worst-fold recall is lower than in the thresholded synthetic-augmented experiment.

## 5.5 Held-Out Test Results

| Metric | Real-only | Synthetic-augmented | Difference |
|---|---:|---:|---:|
| Recall | 0.5238 | 0.6667 | +0.1429 |
| Precision | 0.6875 | 0.6087 | −0.0788 |
| F1-score | 0.5946 | 0.6364 | +0.0418 |
| F2-score | 0.5500 | 0.6542 | +0.1042 |
| Accuracy | 0.7222 | 0.7037 | −0.0185 |
| AUC-ROC | 0.7641 | 0.7561 | −0.0079 |
| RMSE | 0.4317 | 0.4492 | +0.0175 |

The no-threshold experiment confirms the main finding: synthetic augmentation improves held-out recall and F2-score even when no explicit threshold optimization is applied.

## 5.6 CV-to-Test Drift

| Condition | Recall Drift | F2 Drift |
|---|---:|---:|
| Real-only | −0.0704 | −0.0469 |
| Synthetic-augmented | +0.0209 | +0.0273 |

The synthetic-augmented model again shows better CV-to-test consistency than the real-only model.

---

# 6. Cross-Experiment Comparison

## 6.1 Synthetic-Augmented Model: Thresholded vs. Non-Thresholded

| Metric | With Thresholding, θ=0.40 | Without Thresholding | Difference |
|---|---:|---:|---:|
| CV Recall | 0.6663 ± 0.0985 | 0.6458 ± 0.1123 | +0.0205 |
| CV F2-score | 0.6402 ± 0.0725 | 0.6269 ± 0.0863 | +0.0133 |
| CV Precision | 0.5738 ± 0.0662 | 0.5797 ± 0.0431 | −0.0059 |
| CV AUC-ROC | 0.7124 ± 0.0426 | 0.7115 ± 0.0436 | +0.0009 |
| CV RMSE | 0.4694 ± 0.0270 | 0.4652 ± 0.0230 | +0.0042 |
| Test Recall | 0.6667 | 0.6667 | 0.0000 |
| Test F2-score | 0.6542 | 0.6542 | 0.0000 |
| Test Precision | 0.6087 | 0.6087 | 0.0000 |
| Test Accuracy | 0.7037 | 0.7037 | 0.0000 |
| Test AUC-ROC | 0.7561 | 0.7561 | 0.0000 |
| Test RMSE | 0.4492 | 0.4492 | 0.0000 |

For the synthetic-augmented model, thresholding gives slightly better cross-validation recall and F2, but it does not change the held-out test predictions in this run. The same test samples are classified identically under the selected thresholded and non-thresholded synthetic-augmented models.

This means thresholding is not necessary to obtain the synthetic-augmented model's held-out test improvement in this experiment. However, thresholding still gives a more explicit and controllable screening rule.

## 6.2 Real-Only Model: Thresholded vs. Non-Thresholded

| Metric | With Thresholding, θ=0.35 | Without Thresholding | Difference |
|---|---:|---:|---:|
| CV Recall | 0.6568 ± 0.1416 | 0.5942 ± 0.1410 | +0.0626 |
| CV F2-score | 0.6452 ± 0.1162 | 0.5969 ± 0.1149 | +0.0483 |
| CV Precision | 0.6238 ± 0.0516 | 0.6311 ± 0.0370 | −0.0073 |
| CV AUC-ROC | 0.7383 ± 0.0829 | 0.7383 ± 0.0829 | 0.0000 |
| CV RMSE | 0.4418 ± 0.0349 | 0.4418 ± 0.0349 | 0.0000 |
| Test Recall | 0.5714 | 0.5238 | +0.0476 |
| Test F2-score | 0.5941 | 0.5500 | +0.0441 |
| Test Precision | 0.7059 | 0.6875 | +0.0184 |
| Test Accuracy | 0.7407 | 0.7222 | +0.0185 |
| Test AUC-ROC | 0.7641 | 0.7641 | 0.0000 |
| Test RMSE | 0.4317 | 0.4317 | 0.0000 |

Thresholding benefits the real-only model more clearly than the synthetic-augmented model. This likely occurs because the real-only training set remains class-imbalanced, while synthetic augmentation already balances the At-Risk and Typical classes in the training data.

Note that AUC-ROC and RMSE are identical across thresholded and non-thresholded variants for the same condition, because both metrics are computed from `predict_proba()` output, which is independent of the classification threshold. Thresholding only affects the discrete class predictions (and therefore recall, precision, F1, F2, accuracy).

---

# 7. Main Findings

## Finding 1: Synthetic augmentation improves At-Risk detection in both experiments

Across both thresholded and non-thresholded experiments, the synthetic-augmented model improves held-out recall and F2-score compared with the real-only baseline.

| Experiment | Test Recall Improvement | Test F2 Improvement |
|---|---:|---:|
| With thresholding | +0.0952 | +0.0601 |
| Without thresholding | +0.1429 | +0.1042 |

This supports the use of synthetic At-Risk augmentation as a screening-oriented strategy.

## Finding 2: Synthetic augmentation trades precision/accuracy for recall/F2

The synthetic-augmented model consistently lowers precision and accuracy compared with the real-only model. This means it flags more students as At-Risk. For an early-warning screening system, this trade-off is acceptable if false negatives are considered more costly than false positives.

## Finding 3: Thresholding is useful but not essential for the synthetic-augmented test result

For the synthetic-augmented model, the held-out test metrics are identical with and without thresholding:

| Metric | Thresholded | Non-thresholded |
|---|---:|---:|
| Test Recall | 0.6667 | 0.6667 |
| Test F2-score | 0.6542 | 0.6542 |
| Test Precision | 0.6087 | 0.6087 |
| Test Accuracy | 0.7037 | 0.7037 |

Thus, the observed held-out test improvement is not dependent on threshold optimization. This strengthens the claim that the improvement comes primarily from synthetic augmentation rather than from the threshold search.

## Finding 4: Thresholding improves the real-only baseline

The real-only model benefits from a lower threshold because the real-only training data remains imbalanced. Thresholding improves real-only test recall from 0.5238 to 0.5714 and F2-score from 0.5500 to 0.5941.

## Finding 5: Synthetic augmentation improves stability

Synthetic augmentation reduces recall variability in both experiments:

| Experiment | Real-only Recall Range | Synthetic-Augmented Recall Range |
|---|---|---|
| With thresholding | 0.4211–0.8421 | 0.5263–0.7895 |
| Without thresholding | 0.4211–0.8421 | 0.4737–0.7895 |

The thresholded synthetic-augmented model has the better worst-fold recall, while the non-thresholded synthetic-augmented model still narrows the range relative to real-only training.

## Finding 6: Synthetic-augmented models show better CV-to-test consistency

In both experiments, real-only models show negative recall/F2 drift from CV to test, while synthetic-augmented models show near-zero or positive drift. This suggests that synthetic augmentation improves generalization consistency under this split.

---

# 8. Selected Model Parameters

## 8.1 Thresholded Configuration

| Setting | Real-only | Synthetic-augmented |
|---|---|---|
| `conf_fact` | 0.50 after CV tie resolution | 0.40 after CV tie resolution |
| `min_samples_leaf` | 10 | 11 |
| `max_depth` | 6 after CV tie resolution | 7 after CV tie resolution |
| `threshold` | 0.35 | 0.40 |

The thresholded synthetic-augmented deployment configuration is:

```python
BEST_PARAMS = {
    "conf_fact": 0.40,
    "min_samples_leaf": 11,
    "max_depth": 7,
}
LOCKED_THRESHOLD = 0.40
```

## 8.2 Non-Thresholded Configuration

| Setting | Real-only | Synthetic-augmented |
|---|---|---|
| `conf_fact` | 0.50 after CV tie resolution | 0.25 after CV tie resolution |
| `min_samples_leaf` | 10 | 11 |
| `max_depth` | 6 after CV tie resolution | 7 after CV tie resolution |
| `threshold` | N/A | N/A |

The non-thresholded synthetic-augmented configuration is:

```python
BEST_PARAMS = {
    "conf_fact": 0.25,
    "min_samples_leaf": 11,
    "max_depth": 7,
}
```

## 8.3 Recommended Deployment Choice

The recommended deployment choice depends on how strictly the final thesis methodology should avoid threshold optimization.

| Priority | Recommended Setup | Reason |
|---|---|---|
| Maximum methodological simplicity | Non-thresholded synthetic-augmented model | No additional threshold-selection step; still improves test recall and F2 |
| Explicit screening control | Thresholded synthetic-augmented model | Allows the screening cutoff to be locked and reported; slightly better CV recall/F2 |
| Defense-safe compromise | Report both, deploy the thresholded version only if threshold selection is documented in the methodology | Shows that the synthetic effect remains even without thresholding |

If threshold optimization is added to the manuscript, the thresholded synthetic-augmented model is preferable because it provides a transparent screening cutoff. If the manuscript must remain closer to the original proposal, the non-thresholded synthetic-augmented model is simpler and still supports the main conclusion.

---

# 9. Feature Importance

Both experiments selected the same final synthetic-augmented tree structure and feature importance profile:

| Feature | Importance | Domain |
|---|---:|---|
| NC | 0.4928 | Number Comparison |
| NS | 0.2584 | Number Series |
| ADD | 0.1876 | Single-Digit Addition |
| SUB | 0.0612 | Single-Digit Subtraction |
| DM | 0.0000 | Digit-Dot Matching |
| CA | 0.0000 | Multi-Digit Addition and Subtraction |

The model relies mainly on Number Comparison, Number Series, and Single-Digit Addition. Digit-Dot Matching and Complex Arithmetic were available but were not selected as split features in the final synthetic-augmented tree.

If incomplete flags were included in experimental feature sets but not selected as split criteria, they can be reported as non-contributing in this trained tree. However, they should only be removed from deployment data if the deployed model and feature list are fixed to exclude them.

---

# 10. Final Interpretation

The two experiments support the same core conclusion:

> Synthetic At-Risk augmentation improves recall-oriented screening performance compared with real-only C4.5 training.

The evidence is strongest for held-out recall and F2-score. The synthetic-augmented model consistently detects more At-Risk students but produces more false positives, resulting in lower precision and accuracy. This is an acceptable trade-off for a teacher-guided early-warning system, where missed At-Risk learners are more costly than false alarms.

The thresholded experiment provides a more configurable screening rule and slightly better CV behavior for the synthetic-augmented model. The non-thresholded experiment is methodologically simpler and shows that the synthetic-augmented model's held-out test improvement does not depend on threshold optimization.

Therefore, the safest thesis claim is:

> Compared with real-only training, synthetic augmentation improved At-Risk recall and F2-score on the held-out real test set in both thresholded and non-thresholded experiments. Thresholding improved control over the screening decision boundary, but the synthetic-augmented test result remained unchanged without thresholding, suggesting that the observed improvement is primarily attributable to synthetic At-Risk augmentation rather than threshold tuning alone.

---

# 11. Suggested Thesis Wording

Use this wording in the methodology/results section:

> Two variants of the C4.5 screening model were evaluated. The first used the native class prediction of the decision tree, while the second applied a validation-selected post-training threshold to the model-derived At-Risk probability. In both variants, the synthetic-augmented model was trained using real training data plus synthetic At-Risk samples, while validation and test evaluation were performed exclusively on real data. The thresholded variant was included to examine whether a screening-specific cutoff improves recall-oriented performance, while the non-thresholded variant was used to verify that the synthetic augmentation effect persists without explicit threshold optimization.

Use this wording in the results/conclusion section:

> Across both variants, synthetic augmentation improved held-out At-Risk recall and F2-score relative to the real-only baseline, at the cost of lower precision and accuracy. The thresholded synthetic-augmented and non-thresholded synthetic-augmented models produced identical held-out test metrics in this run, indicating that the synthetic augmentation benefit did not rely solely on threshold selection. These results support the use of synthetic At-Risk augmentation for recall-oriented educational screening, while reinforcing that the model should be interpreted as a teacher-guided early-warning tool rather than a clinical diagnostic instrument.

---

# 12. Generated Artifacts

| Artifact | Purpose |
|---|---|
| `outputs/grid_search/trtr_grid_search_results.csv` | Real-only validation grid search with thresholds |
| `outputs/grid_search/tstr_grid_search_results.csv` | Synthetic-augmented validation grid search with thresholds |
| `outputs/grid_search/trtr_no_threshold_grid_search_results.csv` | Real-only validation grid search without thresholds |
| `outputs/grid_search/tstr_no_threshold_grid_search_results.csv` | Synthetic-augmented validation grid search without thresholds |
| `outputs/grid_search/trtr_validation_tie_cv_results.csv` | CV results for tied real-only candidates with thresholds |
| `outputs/grid_search/tstr_validation_tie_cv_results.csv` | CV results for tied synthetic-augmented candidates with thresholds |
| `outputs/grid_search/trtr_validation_tie_no_threshold_cv_results.csv` | CV results for tied real-only candidates without thresholds |
| `outputs/grid_search/tstr_validation_tie_no_threshold_cv_results.csv` | CV results for tied synthetic-augmented candidates without thresholds |
| `outputs/figures/analysis_report.txt` | Compact summary generated by plotting script |
| `outputs/figures/*.png` | Performance, CV distribution, and hyperparameter trend charts |

