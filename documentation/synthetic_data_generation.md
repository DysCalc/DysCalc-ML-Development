# Synthetic Minority Oversampling — CopulaGAN & CTGAN

**Notebook:** `notebooks/synthetic_data_generation.ipynb`

## Overview
This notebook implements the Synthetic Minority Oversampling pipeline using Generative Adversarial Networks (GANs). The objective is to synthetically augment the minority "At-Risk" class (class 1) to stabilize the downstream C4.5 Decision Tree, effectively preventing fold-to-fold variance in cross-validation and boosting real-world recall.

## Key Architecture

### 1. Stratified Data Splitting
- The cleaned dataset is partitioned into a strictly isolated 70/15/15 split (Train/Validation/Test).
- **Leakage Prevention:** Synthetic generation targets *only* the training set. The validation and test sets remain 100% real and untouched to ensure evaluations reflect true real-world performance.

### 2. GAN Training (CopulaGAN + CTGAN)
- Two distinct architectures, `CopulaGANSynthesizer` and `CTGANSynthesizer`, are trained concurrently on the real minority data. 
- These networks are tasked with learning the multi-dimensional joint distributions and covariances of the diagnostic metrics.
- **Hyperparameter Search:** An automated search sweeps across epochs and batch sizes, iteratively evaluating each model's statistical fidelity.

### 3. Statistical Validation (Distribution Checker)
Synthetic samples are not accepted blindly. A rigorous statistical validation suite mandates that candidate synthetic datasets sequentially pass:
- Two-Sample Kolmogorov-Smirnov (KS) test
- Jensen-Shannon Divergence (JSD) thresholds
- Strict bounding on deviations in mean, median, standard deviation, skewness, and kurtosis.

Only GAN configurations that yield statistically indistinguishable distributions from the real minority class are accepted.

### 4. Derived Features & Missing Flags Injection
- **Derived Features:** To maintain exact mathematical covariance structures, derived clinical metrics (NP, SN, AF, BC, AS, PF) are deterministically computed *after* the raw features are generated, rather than trusting the GAN to learn the exact math.
- **Empirical Missingness:** The synthetic records are injected with pseudo-random binomial masks to perfectly mirror the empirical missing-data rates of the real 'At-Risk' cohort identified during EDA. Median imputation is then applied to align with the preprocessing pipeline.

### 5. Final Output
- The structurally validated synthetic samples are appended to the real training dataset to create `s_train.csv` and `s_train_deployment.csv`.
- The notebook generates visual diagnostics (Overlaid Histograms, Q-Q Plots, Correlation Heatmaps) proving the indistinguishable quality of the synthetic distribution from the real distribution.
