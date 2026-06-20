# DysCalc ML Validation Pipeline

## Evaluation Summary

Two evaluation variants are now documented:

- **Thresholded:** converts tree confidence to `P(At-Risk)` and applies a validation-selected cutoff.
- **Non-thresholded:** uses the native C4.5 class prediction directly, with no post-training cutoff search.

The non-thresholded experiment was added to verify that the synthetic-augmentation effect does not depend only on threshold tuning. The deployment script defaults to the thresholded TRSTR settings, and `--no-threshold` switches to the evaluated native-prediction settings shown below.

Selection process: the notebooks first rank the full validation grid by F2, recall, precision, F1, and accuracy. Candidates tied with the top validation row across those metrics are then re-evaluated with 5-fold stratified CV and sorted by CV F2, recall, precision, F1, and accuracy, with threshold/configuration fields used only as deterministic tie-breakers. The first exported tie-CV row becomes the locked configuration; for the thresholded experiment, that same row also locks the threshold. AUC-ROC and RMSE are reported as diagnostics, not as selection criteria.

### Selected Evaluation Settings

| Variant | Condition | Threshold | Params |
|---|---|---:|---|
| Thresholded | TRTR evaluation | 0.35 | `conf_fact=0.45`, `min_samples_leaf=10`, `max_depth=15` |
| Thresholded | TRSTR evaluation | 0.40 | `conf_fact=0.10`, `min_samples_leaf=15`, `max_depth=15` |
| Non-thresholded | TRTR evaluation | N/A | `conf_fact=0.50`, `min_samples_leaf=10`, `max_depth=15` |
| Non-thresholded | TRSTR evaluation | N/A | `conf_fact=0.10`, `min_samples_leaf=15`, `max_depth=15` |

### Thresholded Cross-Validation

| Metric | TRTR | TRSTR | Delta |
|---|---:|---:|---:|
| Recall | 0.6568 +/- 0.1416 | 0.6663 +/- 0.0985 | +0.0095 |
| Precision | 0.6238 +/- 0.0516 | 0.5738 +/- 0.0662 | -0.0500 |
| F1 | 0.6325 +/- 0.0822 | 0.6091 +/- 0.0480 | -0.0234 |
| F2 | 0.6452 +/- 0.1162 | 0.6402 +/- 0.0725 | -0.0050 |
| Accuracy | 0.7160 +/- 0.0388 | 0.6720 +/- 0.0601 | -0.0440 |
| AUC-ROC | 0.7383 +/- 0.0829 | 0.7124 +/- 0.0426 | -0.0259 |
| RMSE | 0.4418 +/- 0.0349 | 0.4694 +/- 0.0270 | +0.0276 |

### Thresholded Held-Out Test

| Metric | TRTR | TRSTR | Delta |
|---|---:|---:|---:|
| Recall | 0.5714 | 0.6667 | +0.0952 |
| Precision | 0.7059 | 0.6087 | -0.0972 |
| F1 | 0.6316 | 0.6364 | +0.0048 |
| F2 | 0.5941 | 0.6542 | +0.0601 |
| Accuracy | 0.7407 | 0.7037 | -0.0370 |
| AUC-ROC | 0.7641 | 0.7561 | -0.0079 |
| RMSE | 0.4317 | 0.4492 | +0.0175 |

### Non-Thresholded Cross-Validation

| Metric | TRTR | TRSTR | Delta |
|---|---:|---:|---:|
| Recall | 0.5942 +/- 0.1410 | 0.6458 +/- 0.1123 | +0.0516 |
| Precision | 0.6311 +/- 0.0370 | 0.5797 +/- 0.0431 | -0.0514 |
| F1 | 0.6052 +/- 0.0791 | 0.6041 +/- 0.0521 | -0.0011 |
| F2 | 0.5969 +/- 0.1149 | 0.6269 +/- 0.0863 | +0.0300 |
| Accuracy | 0.7120 +/- 0.0325 | 0.6800 +/- 0.0358 | -0.0320 |
| AUC-ROC | 0.7383 +/- 0.0829 | 0.7115 +/- 0.0436 | -0.0268 |
| RMSE | 0.4418 +/- 0.0349 | 0.4652 +/- 0.0230 | +0.0235 |

### Non-Thresholded Held-Out Test

| Metric | TRTR | TRSTR | Delta |
|---|---:|---:|---:|
| Recall | 0.5238 | 0.6667 | +0.1429 |
| Precision | 0.6875 | 0.6087 | -0.0788 |
| F1 | 0.5946 | 0.6364 | +0.0418 |
| F2 | 0.5500 | 0.6542 | +0.1042 |
| Accuracy | 0.7222 | 0.7037 | -0.0185 |
| AUC-ROC | 0.7641 | 0.7561 | -0.0079 |
| RMSE | 0.4317 | 0.4492 | +0.0175 |

Across both variants, TRSTR improves held-out recall and F2, with the expected screening trade-off of lower precision and accuracy. In this run, the thresholded and non-thresholded TRSTR models produce identical held-out test metrics, so the TRSTR improvement is not dependent on threshold optimization alone.

The thresholded and non-thresholded TRSTR tree visualizations can also look structurally identical. Thresholding is a post-training decision rule applied to `predict_proba()` output, so it does not change the learned split structure. The selected TRSTR configurations use different `conf_fact` values (`0.40` with thresholding, `0.25` without thresholding), but those values led to the same pruning decisions for the current data. As a result, `v1_full_tree.svg` and `v1_no_threshold_full_tree.svg` can show the same splits, sample counts, gain ratios, and leaf distributions even though they came from different evaluated prediction regimes.

### TRSTR Feature Importance

| Feature | Importance |
|---|---:|
| `NC` | 0.4928 |
| `NS` | 0.2584 |
| `ADD` | 0.1876 |
| `SUB` | 0.0612 |
| `DM` | 0.0000 |
| `CA` | 0.0000 |

Incomplete flags were tested in the research notebook, had 0 split importance, and are dropped from the deployment CSVs.
