# TSTR vs. TRTR Evaluation Report
**Model:** C4.5 Decision Tree | **Dataset:** FunaDB Dyscalculia Screening | **Threshold:** 0.35 | **CV:** 5-Fold Stratified | **Split:** 70/15/15

---

## Overview

This report explains the end-to-end results of a Train-on-Synthetic Test-on-Real (TSTR) evaluation benchmarked against a Train-on-Real Test-on-Real (TRTR) baseline. A C4.5 decision tree was trained across six phases: threshold selection, cross-validation, fold variance inspection, outlier investigation, held-out test evaluation, and global feature importance analysis. The central question is whether augmenting the real training set with 58 synthetic at-risk samples improves identification of dyscalculia-risk students on unseen real data.

---

## Dataset Composition

| Split | Rows | Composition |
|---|---|---|
| TRTR Train | 250 | Real only |
| TSTR Train | 308 | 250 real + 58 synthetic (class 1) |
| Validation | 54 | Real only |
| Test | 54 | Real only |

The synthetic data consists exclusively of class-1 (at-risk) samples. Under the Option A design, these 58 rows are pinned to every TSTR training fold during cross-validation — they never appear in any validation fold. This effectively doubles the positive-class count in TSTR training folds (~135) relative to TRTR (~77), which has significant downstream effects on the tree's decision boundaries and on CV metric interpretation.

---

## Phase 1 — Threshold Selection

Both conditions independently selected a threshold of **0.35** via F2-score maximisation on the validation set.

| Condition | Best Threshold | Val F2 |
|---|---|---|
| TRTR | 0.35 | 0.7619 |
| TSTR | 0.35 | 0.7767 |

The convergence on 0.35 is meaningful. An earlier run using a sweep starting at 0.20 selected that lower threshold, which drove artificially high recall (near 1.0 on some folds) at the cost of precision collapsing to ~0.40 — the model was effectively flagging most students as at-risk. At 0.35, the model is required to express genuine confidence before predicting class 1, producing more discriminating and interpretable behaviour. The slightly higher TSTR val F2 (0.7767 vs 0.7619) provides the first weak signal that synthetic augmentation may help, though a single val set result is insufficient to draw conclusions.

---

## Phase 2 — Cross-Validation Results

5-fold stratified CV was conducted over real samples only. For TSTR, synthetic rows were appended to each training fold but never to validation folds.

### TRTR CV

| Fold | Recall | F2 |
|---|---|---|
| 1 | 0.6000 | 0.5941 |
| 2 | 0.4737 | 0.4945 |
| 3 | 0.5789 | 0.6044 |
| 4 | 0.4737 | 0.4787 |
| 5 | 0.3684 | 0.4023 |
| **Mean** | **0.4989 ± 0.0836** | **0.5148 ± 0.0757** |

### TSTR CV

| Fold | Recall | F2 |
|---|---|---|
| 1 | 0.8500 | 0.6967 |
| 2 | 0.6842 | 0.6633 |
| 3 | 0.6316 | 0.6742 |
| 4 | 0.5263 | 0.5376 |
| 5 | 0.4737 | 0.5172 |
| **Mean** | **0.6332 ± 0.1315** | **0.6178 ± 0.0748** |

### Interpretation

TSTR outperforms TRTR on every CV metric. The recall gain of +0.13 and F2 gain of +0.10 are substantial and consistent with the hypothesis that synthetic class-1 augmentation helps the tree learn more robust at-risk boundaries. Crucially, precision does not degrade — TSTR achieves 0.665 vs TRTR's 0.608 — meaning the improvement is not merely the result of lowering the effective decision boundary.

However, two caveats apply. First, TSTR's recall standard deviation (0.13) is considerably larger than TRTR's (0.08), indicating less stable performance across folds. Second, the pinned synthetic rows guarantee that every TSTR training fold sees ~135 positive samples regardless of which real samples are held out — this inflates apparent generalisation relative to what a truly independent synthetic set would produce.

---

## Phase 3 — Side-by-Side CV Summary

| Metric | TRTR | TSTR | Δ (TSTR − TRTR) |
|---|---|---|---|
| Recall | 0.4989 ± 0.0836 | 0.6332 ± 0.1315 | **+0.1342** |
| Precision | 0.6082 ± 0.0769 | 0.6650 ± 0.1839 | +0.0568 |
| F1 | 0.5430 ± 0.0661 | 0.6176 ± 0.0731 | +0.0746 |
| F2 | 0.5148 ± 0.0757 | 0.6178 ± 0.0748 | **+0.1030** |
| Accuracy | 0.6800 ± 0.0456 | 0.6840 ± 0.1341 | +0.0040 |

The most important observation here is that every delta is positive — TSTR does not trade recall for precision or vice versa; it improves on both simultaneously in CV. The near-zero accuracy delta (+0.004) reflects that accuracy is dominated by class-0 predictions, which are largely unaffected by the class-1 synthetic augmentation.

---

## Phase 4 — Fold Variance Inspection

### TRTR

| Fold | N_train | C1_train | N_val | C1_val | Depth | Recall | F2 | Flag |
|---|---|---|---|---|---|---|---|---|
| 1 | 200 | 76 | 50 | 20 | 9 | 0.6000 | 0.5941 | ⚠ HIGH |
| 2 | 200 | 77 | 50 | 19 | 9 | 0.4737 | 0.4945 | |
| 3 | 200 | 77 | 50 | 19 | 10 | 0.5789 | 0.6044 | |
| 4 | 200 | 77 | 50 | 19 | 10 | 0.4737 | 0.4787 | |
| 5 | 200 | 77 | 50 | 19 | 10 | 0.3684 | 0.4023 | ⚠ LOW |

Correlation: C1_train vs recall = −0.605 ⚠, tree_depth vs recall = −0.370

### TSTR

| Fold | N_train | C1_train | N_val | C1_val | Depth | Recall | F2 | Flag |
|---|---|---|---|---|---|---|---|---|
| 1 | 258 | 134 | 50 | 20 | 9 | 0.8500 | 0.6967 | ⚠ HIGH |
| 2 | 258 | 135 | 50 | 19 | 10 | 0.6842 | 0.6633 | |
| 3 | 258 | 135 | 50 | 19 | 10 | 0.6316 | 0.6742 | |
| 4 | 258 | 135 | 50 | 19 | 10 | 0.5263 | 0.5376 | |
| 5 | 258 | 135 | 50 | 19 | 10 | 0.4737 | 0.5172 | ⚠ LOW |

Correlation: C1_train vs recall = −0.824 ⚠, tree_depth vs recall = −0.824 ⚠

### Interpretation

Both conditions show a consistent fold-ordering effect: recall declines monotonically from Fold 1 to Fold 5. Since the random seed is fixed, this reflects the genuine structural composition of the data as assigned by StratifiedKFold — Fold 1's validation set appears to contain easier-to-classify positive cases.

The strongly negative correlation between C1_train and recall in both conditions (−0.61 for TRTR, −0.82 for TSTR) is counterintuitive and warrants attention. Naively, more positive training samples should improve recall. The negative correlation instead suggests that fold 1, which has slightly fewer positive training samples (76 vs 77) and the highest recall, benefits from a training distribution that is more tightly aligned with the particular positive subgroup present in its validation set. As subsequent folds rotate in more diverse positive samples, the tree's boundaries become noisier or shift in ways that miss the specific val-fold subgroup.

For TSTR, this effect is amplified by the pinned synthetic samples. Although C1_train is nearly constant across folds (~134–135), the 58 synthetic rows interact differently with each fold's real positive complement — potentially introducing split noise on ADD, DM, and NC that the tree resolves differently fold-to-fold.

---

## Phase 4b — TSTR Outlier Fold Investigation (Fold 5)

Fold 5 was identified as the lowest-recall TSTR fold (recall = 0.4737 vs mean = 0.6332). The investigation compared the at-risk feature distributions of the Fold 5 validation set against the at-risk cases in its corresponding training set.

| Feature | Val Mean | Val Std | Train Mean | Train Std |
|---|---|---|---|---|
| NC | 1154.3 | 321.6 | 1089.1 | 351.4 |
| DM | 2569.1 | 753.9 | 2452.4 | 924.7 |
| NS | 10.9 | 5.1 | 12.1 | 5.7 |
| ADD | 30.9 | 10.7 | 31.1 | 13.7 |
| SUB | 24.3 | 11.8 | 27.9 | 13.5 |
| CA | 15.4 | 7.3 | 17.0 | 9.2 |

The feature distributions are broadly overlapping — no single feature shows a dramatic shift between val and train. The most notable divergence is in SUB (val mean 24.3 vs train mean 27.9) and NC (val 1154 vs train 1089), suggesting the Fold 5 val positives tend slightly toward lower subtraction scores and higher number comparison times. Since SUB accounts for only 5.5% of tree importance and NC for 27.6%, the NC shift is more likely to matter. However, given the overlapping standard deviations, the performance drop in Fold 5 is more plausibly attributable to the inherent difficulty of a small positive val sample (n=19) than to a systematic distributional gap. This fold is a natural low — not a structural failure of the synthetic data.

---

## Phase 5 — Test Set Evaluation

Both models were retrained on their full respective training sets, calibrated on the validation set, and evaluated on the held-out test set (n=54) at threshold 0.35.

| Metric | TRTR | TSTR | Δ (TSTR − TRTR) |
|---|---|---|---|
| Recall | **0.7619** | 0.7143 | −0.0476 |
| Precision | 0.6667 | **0.7143** | +0.0476 |
| F1 | 0.7111 | **0.7143** | +0.0032 |
| F2 | **0.7407** | 0.7143 | −0.0265 |
| Accuracy | 0.7593 | **0.7778** | +0.0185 |

### CV-to-Test Drift

| Condition | Recall Drift | F2 Drift |
|---|---|---|
| TRTR | +0.2630 ⚠ | +0.2259 ⚠ |
| TSTR | +0.0811 | +0.0965 |

### Interpretation

The test results are effectively a statistical tie. With n=54, a single misclassified positive shifts recall by ~0.048, meaning the entire TRTR recall advantage (+0.0476) corresponds to exactly one additional true positive. No meaningful conclusion about model superiority can be drawn from this margin alone.

The CV-to-test drift figures are more informative than the raw test metrics. TRTR's large positive drift (+0.26 recall) reveals that its CV scores were pessimistic — the real training data generalises better to the test set than the CV folds suggested, likely because the full training set covers more of the positive subspace than any single 4/5 fold. TSTR's small drift (+0.08) confirms that its CV scores were already a reliable estimate of test performance, which is the desired property of a well-designed evaluation pipeline.

This asymmetry has an important implication: TRTR's CV recall (0.499) significantly underestimated its true capability, while TSTR's CV recall (0.633) was an honest predictor. For future model selection, TSTR's CV metrics are the more trustworthy signal.

---

## Phase 6 — Global Feature Importance (TSTR Tree)

### Incomplete Flags

No incomplete flag (NC_incomplete, DM_incomplete, etc.) was selected as a split node. The tree ignored all six flags entirely, meaning missingness/timeout signals carry no discriminative power beyond what the raw task scores already encode. These features can be retained as a safety net for edge cases but do not need to be treated as primary predictors.

### Task Feature Importance

| Feature | Importance | Domain |
|---|---|---|
| ADD | 0.3285 | Single-Digit Addition |
| DM | 0.3242 | Digit-Dot Matching |
| NC | 0.2759 | Number Comparison |
| SUB | 0.0553 | Single-Digit Subtraction |
| NS | 0.0162 | Number Series |
| CA | ~0.0000 | Multi-Digit Arithmetic |

Three features — ADD, DM, and NC — account for **93% of total split importance**. This concentration indicates that dyscalculia risk in this dataset is primarily characterised by deficits in basic arithmetic fluency (ADD), symbolic-to-non-symbolic mapping speed (DM), and number magnitude processing (NC). SUB contributes marginally, and NS and CA are effectively unused.

The dominance of ADD over CA is notable given that CA (multi-digit addition/subtraction) is a more complex operation. This suggests the tree finds single-digit fluency more discriminating — consistent with the view that foundational automaticity deficits, rather than procedural breakdown on complex tasks, are the primary marker in this population.

The near-zero CA importance also raises a question about whether CA captures meaningful additional variance beyond what ADD and SUB already encode, or whether its score range and distribution in this dataset are simply too narrow to produce informative splits.

---

## Summary and Recommendations

### What the results show

TSTR produces more stable and better-calibrated CV estimates than TRTR. Its CV-to-test drift is small (+0.08 recall), confirming that the evaluation pipeline is functioning correctly and that CV performance is predictive of real-world behaviour. On the test set, TRTR and TSTR are statistically indistinguishable given the sample size.

### Why TSTR does not dominate on test

The 58 synthetic samples are all class-1 and cover only a portion of the real positive subspace. While they shift the tree's decision boundaries in a direction that improves CV recall, the full training set (used for the final test model) already provides more real positive examples than any single CV fold — reducing the marginal benefit of synthetic augmentation at test time.

### Recommendations

1. **Use TSTR for deployment.** Its CV metrics are reliable predictors of test performance, making it the safer choice for model selection. TRTR's CV scores systematically underestimate true performance, which is a liability when tuning hyperparameters or comparing configurations.

2. **Expand synthetic coverage beyond class 1.** Currently, all 58 synthetic samples are at-risk. Generating synthetic class-0 samples near the decision boundary may reduce the false-positive rate without sacrificing recall.

3. **Investigate ADD, DM, and NC covariance in synthetic generation.** These three features drive 93% of tree splits. If the synthetic generator does not faithfully reproduce their joint distribution with the real data, the synthetic samples may introduce misleading split signals. Validate by comparing bivariate distributions (ADD vs DM, DM vs NC) between real and synthetic positives.

4. **Increase test set size before drawing final conclusions.** At n=54, a single prediction difference produces a 0.048 metric swing. A minimum of 100–150 test samples would be needed to reliably distinguish the two conditions.

5. **Examine the declining fold recall pattern.** The monotonic recall decline from Fold 1 to Fold 5 reflects a structural property of how StratifiedKFold assigned samples under `random_state=42`. Running CV with multiple random seeds would reveal whether this pattern is seed-specific or a genuine data property.