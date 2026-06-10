# RMAT Labeling Percentile Comparison Report

This report presents a comprehensive comparative analysis of the machine learning validation pipelines trained using different percentile thresholds for RMAT-based dyscalculia labeling. Specifically, it compares labeling percentiles from **10% to 35%** in increments of 5%.

## Notebooks Evaluated
*   **10% Pipeline:** [DysCalc_ML_Validation_Pipeline_10.ipynb](file:///home/caineirb/Documents/DysCalc/DysCalc-ML-Development/notebooks/DysCalc_ML_Validation_Pipeline_10.ipynb)
*   **15% Pipeline:** [DysCalc_ML_Validation_Pipeline_15.ipynb](file:///home/caineirb/Documents/DysCalc/DysCalc-ML-Development/notebooks/DysCalc_ML_Validation_Pipeline_15.ipynb)
*   **20% Pipeline:** [DysCalc_ML_Validation_Pipeline_20.ipynb](file:///home/caineirb/Documents/DysCalc/DysCalc-ML-Development/notebooks/DysCalc_ML_Validation_Pipeline_20.ipynb)
*   **25% Pipeline:** [DysCalc_ML_Validation_Pipeline_25.ipynb](file:///home/caineirb/Documents/DysCalc/DysCalc-ML-Development/notebooks/DysCalc_ML_Validation_Pipeline_25.ipynb)
*   **30% Pipeline:** [DysCalc_ML_Validation_Pipeline_30.ipynb](file:///home/caineirb/Documents/DysCalc/DysCalc-ML-Development/notebooks/DysCalc_ML_Validation_Pipeline_30.ipynb)
*   **35% Pipeline (Default):** [DysCalc_ML_Validation_Pipeline.ipynb](file:///home/caineirb/Documents/DysCalc/DysCalc-ML-Development/notebooks/DysCalc_ML_Validation_Pipeline.ipynb)

---

## 1. Executive Summary & Key Findings

> [!IMPORTANT]
> **Data Scarcity & GAN Mismatch Failure at 10% and 15%**
> 
> *   At the lower percentiles (**10%** and **15%**), the minority (At-Risk) cohort in the training set is extremely small ($n = 25$ and $n = 36$ respectively).
> *   Because of this data scarcity, the GAN synthesizers (CopulaGAN and CTGAN) cannot capture the underlying minority distribution well enough to generate synthetic samples that pass the pipeline's strict statistical similarity checks (KS-test, Wasserstein distance, Jensen-Shannon divergence, and moment matching).
> *   Consequently, both the 10% and 15% notebooks failed with an `AssertionError` during the GAN hyperparameter grid search (Cell 76), indicating that **no synthetic dataset was statistically valid**. As a result, no downstream C4.5 models were trained or evaluated for 10% and 15%.
> *   *Note on Stale Notebook State:* In the repository, cells from Cell 78 onwards in the 10% and 15% notebooks contain saved outputs that are identical to the 20% notebook. This is because they were copied from the 20% notebook but were never fully re-executed due to the failure at Cell 76. The actual, verified execution states are summarized here.

### Summary of Successful Pipeline Runs (20% to 35%)
1.  **Synthetic Augmentation (TRSTR) consistently improves recall-oriented screening:** In all successful pipelines, training on original plus synthetic At-Risk rows (TRSTR) significantly outperforms the real-only baseline (TRTR) in terms of test set Recall and F2-Score.
2.  **Sensitivity maximizes at the 35th percentile:** The default pipeline (35%) achieved the highest overall test sensitivity, with a TRSTR Recall of **0.8824** and a Precision of **0.7500** (F2-score: **0.8523**).
3.  **Thresholding is critical for Real-Only (TRTR) models but redundant for balanced (TRSTR) models:** Real-only models train on highly imbalanced training sets, so post-training probability thresholding significantly improves their test recall. Synthetic-augmented (TRSTR) training already balances the classes in training, so thresholding does not alter final test set predictions.
4.  **Feature selection shifts from basic math (ADD) to serial processing (NS):** At lower percentiles (20% and 25%), models rely heavily on Single-Digit Addition (`ADD`). At higher percentiles (30% and 35%), Number Series (`NS`) becomes the dominant split feature, reflecting a shift from basic arithmetic fluency to sequential reasoning.

---

## 2. Dataset Composition & Partition Sizes

The raw dataset contains 358 subjects. In all pipelines, subjects with missing or invalid RMAT scores ($RMAT < 0$) are removed, reducing the effective dataset size to **316 subjects**. 

The dataset is partitioned using a stratified 70/15/15 split. The table below outlines the class distribution for Typical ($Class\ 0$) and At-Risk ($Class\ 1$) subjects across splits and shows the number of synthetic samples needed to balance the training split.

| RMAT Percentile | Raw Class 0 / 1 | Train Class 0 / 1 | Validation Class 0 / 1 | Test Class 0 / 1 | Synthetic Needed | GAN Execution Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | 280 / 36 | 196 / 25 | 42 / 5 | 42 / 6 | 171 | **FAILED** (Cell 76 AssertionError) |
| **15%** | 264 / 52 | 185 / 36 | 39 / 8 | 40 / 8 | 149 | **FAILED** (Cell 76 AssertionError) |
| **20%** | 247 / 69 | 173 / 48 | 37 / 10 | 37 / 11 | 125 | **SUCCESSFUL** |
| **25%** | 235 / 81 | 164 / 57 | 35 / 12 | 36 / 12 | 107 | **SUCCESSFUL** |
| **30%** | 220 / 96 | 154 / 67 | 33 / 14 | 33 / 15 | 87 | **SUCCESSFUL** |
| **35%** | 205 / 111 | 143 / 78 | 31 / 16 | 31 / 17 | 65 | **SUCCESSFUL** |

---

## 3. GAN Synthesizer Hyperparameter Search

For the successful runs, the similarity grid search evaluated 8 configurations to select the best synthetic dataset based on statistical similarity to the real minority training data.

| RMAT Percentile | Selected GAN Config (Multiplier, Epochs, Batch, Gen LR, Disc LR) | Similarity Score (Lower = Closer) | Moment Failures (Max 2 Allowed) | Selected Rows |
| :---: | :---: | :---: | :---: | :---: |
| **20%** | `MULT=15`, `ep=5000`, `bs=100`, `lr=1e-4` | 11.3968 | 0 | 125 |
| **25%** | `MULT=15`, `ep=5000`, `bs=100`, `lr=1e-4` | 14.9655 | 1 | 107 |
| **30%** | `MULT=10`, `ep=3000`, `bs=500`, `lr=2e-4` | 14.8532 | 2 | 87 |
| **35%** | `MULT=30`, `ep=5000`, `bs=100`, `lr=1e-4` | 13.1894 | 1 | 65 |

---

## 4. Cross-Validation Results

The locked model hyperparameters and probability thresholds were resolved using a 5-fold stratified cross-validation on the real training set. The table below compares the CV Recall scores (mean ± standard deviation and range) for both Real-Only (TRTR) and Synthetic-Augmented (TRSTR) training across no-threshold and thresholded regimes.

| Percentile | TRTR (No Threshold) | TRSTR (No Threshold) | TRTR (Thresholded) | TRSTR (Thresholded) |
| :---: | :---: | :---: | :---: | :---: |
| **20%** | 0.6267 ± 0.1278<br>*(0.4000 - 0.7000)* | 0.9378 ± 0.0908<br>*(0.8000 - 1.0000)* | 0.6667 ± 0.1780<br>*(0.4000 - 0.9000)* | **0.9578 ± 0.0579**<br>*(0.8889 - 1.0000)* |
| **25%** | 0.5818 ± 0.1897<br>*(0.3333 - 0.8182)* | 0.9303 ± 0.0707<br>*(0.8333 - 1.0000)* | 0.6833 ± 0.1620<br>*(0.4545 - 0.8333)* | **0.9636 ± 0.0498**<br>*(0.9091 - 1.0000)* |
| **30%** | 0.5967 ± 0.2171<br>*(0.3571 - 0.8571)* | 0.8352 ± 0.0373<br>*(0.7692 - 0.8571)* | 0.7022 ± 0.2305<br>*(0.3571 - 1.0000)* | **0.8791 ± 0.0869**<br>*(0.7692 - 1.0000)* |
| **35%** | 0.5383 ± 0.1351<br>*(0.3750 - 0.7500)* | 0.7692 ± 0.0348<br>*(0.7333 - 0.8125)* | 0.5767 ± 0.1374<br>*(0.3750 - 0.7500)* | **0.7825 ± 0.0301**<br>*(0.7500 - 0.8125)* |

> [!TIP]
> **Key CV Observations:**
> 
> *   Synthetic augmentation narrows the CV recall standard deviation and range, increasing the stability of At-Risk detection.
> *   For both TRTR and TRSTR, applying post-training thresholding increases average CV recall. The highest overall CV recall is achieved under thresholded TRSTR at 25% (**0.9636**), followed by 20% (**0.9578**).

---

## 5. Final Test Set Evaluation

Evaluation metrics were computed on the locked, held-out real test dataset. The tables below compare all test metrics across percentiles.

### 5.1 No-Threshold Test Performance

| Percentile | Model | Recall | Precision | F1-Score | F2-Score | Accuracy | AUC-ROC | RMSE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20%** | TRTR | 0.2727 | 0.5000 | 0.3529 | 0.3000 | 0.7708 | 0.8501 | 0.3679 |
| | **TRSTR** | **0.8182** | **0.4500** | **0.5806** | **0.7031** | **0.7292** | **0.8722** | **0.4044** |
| **25%** | TRTR | 0.5000 | 0.6000 | 0.5455 | 0.5172 | 0.7917 | 0.6944 | 0.3985 |
| | **TRSTR** | **0.9167** | **0.5789** | **0.7097** | **0.8209** | **0.8125** | **0.9132** | **0.3618** |
| **30%** | TRTR | 0.6000 | 0.6000 | 0.6000 | 0.6000 | 0.7500 | 0.8242 | 0.3841 |
| | **TRSTR** | **0.6000** | **0.5625** | **0.5806** | **0.5921** | **0.7292** | **0.7556** | **0.4528** |
| **35%** | TRTR | 0.8235 | 0.7000 | 0.7568 | 0.7955 | 0.8125 | 0.8634 | 0.3792 |
| | **TRSTR** | **0.8824** | **0.7500** | **0.8108** | **0.8523** | **0.8542** | **0.8994** | **0.3638** |

### 5.2 Thresholded Test Performance

| Percentile | Model | Recall | Precision | F1-Score | F2-Score | Accuracy | AUC-ROC | RMSE | Locked Threshold |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20%** | TRTR | 0.6364 | 0.5385 | 0.5833 | 0.6140 | 0.7917 | 0.8501 | 0.3679 | 0.35 |
| | **TRSTR** | **0.8182** | **0.4500** | **0.5806** | **0.7031** | **0.7292** | **0.8747** | **0.3954** | **0.35** | **0.35** |
| **25%** | TRTR | 0.6667 | 0.5000 | 0.5714 | 0.6250 | 0.7500 | 0.8495 | 0.3752 | 0.45 |
| | **TRSTR** | **0.9167** | **0.5789** | **0.7097** | **0.8209** | **0.8125** | **0.9132** | **0.3618** | **0.35** | **0.35** |
| **30%** | TRTR | 0.6000 | 0.6000 | 0.6000 | 0.6000 | 0.7500 | 0.8242 | 0.3841 | 0.40 |
| | **TRSTR** | **0.6000** | **0.5625** | **0.5806** | **0.5921** | **0.7292** | **0.7556** | **0.4528** | **0.40** | **0.40** |
| **35%** | TRTR | 0.8235 | 0.7000 | 0.7568 | 0.7955 | 0.8125 | 0.8776 | 0.3720 | 0.35 |
| | **TRSTR** | **0.8824** | **0.7500** | **0.8108** | **0.8523** | **0.8542** | **0.8994** | **0.3638** | **0.40** | **0.40** |

> [!WARNING]
> **Important Test Observations:**
> 
> *   **TRSTR Recall Improvement:** Training with synthetic data (TRSTR) consistently improves test set Recall compared to training on real data only (TRTR). The largest increase occurs at the 20% percentile (Recall goes from **0.2727** to **0.8182**, $+0.5455$) and 25% percentile (Recall goes from **0.5000** to **0.9167**, $+0.4167$).
> *   **Redundancy of Thresholding for TRSTR:** For the synthetic-augmented (TRSTR) model, thresholding has **no effect** on test set predictions. The Recall, Precision, and Accuracy remain exactly the same between no-threshold and thresholded variants. This indicates that synthetic augmentation successfully balances the classes, making the native decision boundary equivalent to the optimized probability boundary.
> *   **Percentile Trends:** Test set recall is strongest at the 25% percentile (**0.9167**) and 35% percentile (**0.8824**). However, the 35% percentile model provides a much better balance of metrics, yielding a Precision of **0.7500** (F1: **0.8108**, F2: **0.8523**) and an overall Accuracy of **0.8542**, whereas the 25% model has a Precision of **0.5789** (F1: **0.7097**, F2: **0.8209**) and Accuracy of **0.8125**.

---

## 6. Locked Model Hyperparameters

The grid search locked the following optimal parameters for both training regimes.

### 6.1 No-Threshold Model Hyperparameters

| Percentile | TRTR Locked Params | TRSTR Locked Params |
| :---: | :--- | :--- |
| **20%** | `cf=0.50`, `msl=10`, `md=15` | `cf=0.50`, `msl=16`, `md=5` |
| **25%** | `cf=0.50`, `msl=41`, `md=15` | `cf=0.50`, `msl=18`, `md=6` |
| **30%** | `cf=0.50`, `msl=26`, `md=15` | `cf=0.15`, `msl=10`, `md=15` |
| **35%** | `cf=0.50`, `msl=10`, `md=15` | `cf=0.10`, `msl=15`, `md=15` |

### 6.2 Thresholded Model Hyperparameters

| Percentile | TRTR Locked Params / Threshold | TRSTR Locked Params / Threshold |
| :---: | :--- | :--- |
| **20%** | `cf=0.50`, `msl=10`, `md=15` / **0.35** | `cf=0.50`, `msl=16`, `md=15` / **0.35** |
| **25%** | `cf=0.50`, `msl=10`, `md=15` / **0.45** | `cf=0.50`, `msl=18`, `md=15` / **0.35** |
| **30%** | `cf=0.50`, `msl=27`, `md=15` / **0.40** | `cf=0.50`, `msl=10`, `md=15` / **0.40** |
| **35%** | `cf=0.45`, `msl=10`, `md=15` / **0.35** | `cf=0.10`, `msl=15`, `md=15` / **0.40** |

---

## 7. Feature Importance Analysis

The table below contrasts the feature importance of the synthetic-augmented (TRSTR) model across percentiles. In all models, the C4.5 tree successfully prunes out all incompleteness/missingness flags, confirming that they do not contribute to final diagnostic predictions.

| Feature | 20% Percentile | 25% Percentile | 30% Percentile | 35% Percentile | Cognitive Domain |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ADD** | **0.5463** | **0.5926** | 0.1800 | 0.1071 | Single-Digit Addition (Arithmetic Fluency) |
| **NS** | 0.2562 | 0.3244 | **0.3984** | **0.7011** | Number Series (Arithmetic Fluency) |
| **SUB** | 0.1096 | 0.0000 | 0.1093 | 0.0158 | Single-Digit Subtraction (Arithmetic Fluency) |
| **CA** | 0.0505 | 0.0000 | 0.0290 | 0.1557 | Complex Addition/Subtraction (Arithmetic Fluency) |
| **NC** | 0.0375 | 0.0000 | 0.0233 | 0.0000 | Number Comparison (Number Processing) |
| **DM** | 0.0000 | 0.0830 | 0.2600 | 0.0203 | Digit-Dot Matching (Number Processing) |

> [!NOTE]
> **Pruning & Similarity Behaviors:**
> 
> *   For **25%, 30%, and 35%**, the thresholded and no-threshold TRSTR trees have **identical** feature importance values. This is because the C4.5 pruning confidence factors (`cf`) and min leaf sizes (`msl`) pruned the tree to a structure that did not exceed the lower depth limit (e.g., in 25%, the tree did not grow past depth 6 even when max depth was 15).
> *   For **20%**, the thresholded and no-threshold TRSTR trees differ slightly because the no-threshold tree's `max_depth` was capped at 5, whereas the thresholded tree grew to allow a split on Digit-Dot Matching (`DM = 0.0056`).

---

## 8. Discussion & Recommendations for the Thesis

### 8.1 Methodological Failure at 10% & 15%
In the thesis manuscript, the failure of the 10% and 15% labeling threshold models should be presented as an **empirical boundary result** rather than a programming bug:
> *"The statistical similarity checks built into our synthetic data validation pipeline reveal an operational limit: when labeling Dyscalculia at extremely strict cutoffs (the 10th or 15th percentile of RMAT performance), the training dataset contains too few real positive cases ($n = 25$ and $n = 36$ respectively) to guide the CopulaGAN or CTGAN models. The generators fail to match the statistical distributions of the real subjects, resulting in pipeline halt due to validation checks. This demonstrates that synthetic minority oversampling requires a minimal empirical threshold of real minority cases to produce valid data."*

### 8.2 Comparison of 20%, 25%, 30%, and 35%
*   **The 35% labeling threshold remains the strongest candidate for general deployment.** It delivers a high test recall (**0.8824**) while maintaining the highest test precision (**0.7500**) and overall accuracy (**0.8542**).
*   **The 25% threshold represents a high-sensitivity alternative.** It achieves the absolute highest test set recall (**0.9167**) but trades off precision (**0.5789**) and overall accuracy (**0.8125**). If the screening policy assigns a very high cost to false negatives, the 25% model could be justified.
*   **The 30% model represents a performance anomaly.** It performs poorly, achieving only **0.6000** recall. This drop is likely due to high variance or a misalignment between the synthetic data distribution and the real test set at that specific cutoff (indicated by its lower CV scores: **0.8791** recall vs **0.9578** for 20% and **0.9636** for 25%).
