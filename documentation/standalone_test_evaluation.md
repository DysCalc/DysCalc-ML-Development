# Standalone Test Evaluation

This script evaluates the TRTR and TRSTR models on the held-out test set using both the thresholded and non-thresholded prediction methods, applying a bootstrap mechanism to generate confidence intervals.

## Execution Output

```text
--- NO_THRESHOLD TEST EVALUATION WITH CONFIDENCE INTERVALS ---
  [TRTR | no_threshold] Locked params: {'conf_fact': 0.5, 'min_samples_leaf': 10, 'max_depth': 15}
  [TRTR | no_threshold] Test Set Results:
    Recall:    0.8235  [95% CI: 0.5897, 0.9381] (Wilson)
    Precision: 0.7000  [95% CI: 0.4810, 0.8545] (Wilson)
    F1-Score:  0.7568  [95% CI: 0.5714, 0.8889] (Bootstrap)
    F2-Score:  0.7955  [95% CI: 0.6111, 0.9341] (Bootstrap)
    Accuracy:  0.8125  [95% CI: 0.6875, 0.9167] (Bootstrap)
    AUC-ROC:   0.8634  [95% CI: 0.7474, 0.9539] (Bootstrap)
    RMSE:      0.3792  [95% CI: 0.2910, 0.4644] (Bootstrap)

  [TRSTR | no_threshold] Locked params: {'conf_fact': 0.1, 'min_samples_leaf': 15, 'max_depth': 15}
  [TRSTR | no_threshold] Test Set Results:
    Recall:    0.8824  [95% CI: 0.6566, 0.9671] (Wilson)
    Precision: 0.7500  [95% CI: 0.5313, 0.8881] (Wilson)
    F1-Score:  0.8108  [95% CI: 0.6667, 0.9362] (Bootstrap)
    F2-Score:  0.8523  [95% CI: 0.6962, 0.9649] (Bootstrap)
    Accuracy:  0.8542  [95% CI: 0.7500, 0.9375] (Bootstrap)
    AUC-ROC:   0.8994  [95% CI: 0.7998, 0.9742] (Bootstrap)
    RMSE:      0.3638  [95% CI: 0.2757, 0.4420] (Bootstrap)

--- THRESHOLDED TEST EVALUATION WITH CONFIDENCE INTERVALS ---

  [TRTR | thresholded] Locked params: {'conf_fact': 0.45, 'min_samples_leaf': 10, 'max_depth': 15}
  [TRTR | thresholded] Locked threshold: 0.35
  [TRTR | thresholded] Test Set Results:
    Recall:    0.8235  [95% CI: 0.5897, 0.9381] (Wilson)
    Precision: 0.7000  [95% CI: 0.4810, 0.8545] (Wilson)
    F1-Score:  0.7568  [95% CI: 0.5625, 0.8980] (Bootstrap)
    F2-Score:  0.7955  [95% CI: 0.6172, 0.9259] (Bootstrap)
    Accuracy:  0.8125  [95% CI: 0.6875, 0.9167] (Bootstrap)
    AUC-ROC:   0.8776  [95% CI: 0.7764, 0.9611] (Bootstrap)
    RMSE:      0.3720  [95% CI: 0.2916, 0.4441] (Bootstrap)

  [TRSTR | thresholded] Locked params: {'conf_fact': 0.1, 'min_samples_leaf': 15, 'max_depth': 15}
  [TRSTR | thresholded] Locked threshold: 0.40
  [TRSTR | thresholded] Test Set Results:
    Recall:    0.8824  [95% CI: 0.6566, 0.9671] (Wilson)
    Precision: 0.7500  [95% CI: 0.5313, 0.8881] (Wilson)
    F1-Score:  0.8108  [95% CI: 0.6363, 0.9333] (Bootstrap)
    F2-Score:  0.8523  [95% CI: 0.6897, 0.9649] (Bootstrap)
    Accuracy:  0.8542  [95% CI: 0.7500, 0.9583] (Bootstrap)
    AUC-ROC:   0.8994  [95% CI: 0.7954, 0.9727] (Bootstrap)
    RMSE:      0.3638  [95% CI: 0.2792, 0.4418] (Bootstrap)
```
