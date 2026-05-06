# Dyscalculia Risk Screening — C4.5 TSTR Pipeline

A machine learning system for early dyscalculia risk screening using a custom C4.5 decision tree trained with synthetic data augmentation (TSTR). Built on the **FunaDB** dataset, the model classifies students as *At-Risk (1)* or *Typical (0)* based on six numeracy task scores, with per-prediction diagnostic outputs for interpretability.

---

## Project Structure

```
dyscalc-ml-development/
│
├── datasets/
│   ├── raw/                       
│   │   └── FUNADB_rawdata_SUPPL.csv      # Raw FunaDB dataset (source)
│   │
│   └── processed/                 
│       ├── FUNADB_labled.csv             # Labeled dataset (output of RMAT_Labeling.py)
│       ├── cleaned_dataset.csv           # Cleaned dataset after preprocessing
│       ├── train.csv                     # Real training split (TRTR mode)
│       ├── val.csv                       # Validation split
│       ├── test.csv                      # Test split
│       ├── s_train.csv                   # Synthetic training samples (class 1 only)
│       ├── train_deployment.csv          # Deployment training split (real, with derived features)
│       ├── val_deployment.csv            # Deployment validation split
│       ├── test_deployment.csv           # Deployment test split
│       └── s_train_deployment.csv        # Deployment synthetic training split
│
├── documentation/
│   └── TSTR_results.md                   # Full TSTR vs TRTR evaluation report
|
├── models/
│   └── v1.pkl                            # Saved deployment models (.pkl)
│
├── notebooks/
│   ├── dataset_analysis.ipynb            # Exploratory data analysis
│   ├── ML_development.ipynb              # ML Model development and testing
│   ├── synthetic_data_generation.ipynb   # Synthetic data generation experiments
│   └── tstr_vs_trtr.ipynb                # Full TSTR vs TRTR evaluation pipeline
│
├── outputs/
│   ├── figures/                          # Saved tree visualization
│   └── logs_and_metrics/          
│       └── missing_rates.json            # Per-class missing value rates per feature
│   
│
├── scripts/
│   ├── RMAT_Labeling.py                  # Converts raw FunaDB data to labeled dataset
│   └── train.py                          # Deployment training and model saving script
│
└── src/                                  
    ├── __init__.py                       # Module initialization
    ├── C45DecisionTree.py                # Custom C4.5 decision tree implementation
    └── Dataclasses.py                    # Node and DiagnosticOutput dataclasses
```

---

## Background

**Dyscalculia**, a specific learning disorder in mathematics, affects a significant portion of the population worldwide. Early identification is critical but resource-intensive. This project automates screening using a C4.5 decision tree trained on FunaDB — a dataset of numeracy task performance scores — with the goal of flagging at-risk students for further evaluation.

The pipeline uses **TSTR (Train on Synthetic, Test on Real)**: synthetic at-risk samples are generated to augment a small real training set, and the model is validated exclusively on real data.

---

## Dataset

### Source: FunaDB

The **FunaDB** dataset (`datasets/raw/FUNADB_rawdata_SUPPL.csv`) contains raw task scores from numeracy assessments administered to students. After preprocessing, six task score features are retained:

| Feature | Description |
|---|---|
| `NC` | Number Comparison — response time (ms) |
| `DM` | Digit-Dot Matching — response time (ms) |
| `NS` | Number Series — score |
| `ADD` | Single-Digit Addition — score |
| `SUB` | Single-Digit Subtraction — score |
| `CA` | Multi-Digit Addition and Subtraction — score |

### Labeling (`RMAT_Labeling.py`)

Labels are derived from the **RMAT** (Risk for Mathematics) composite score in the raw data:

1. RMAT scores are z-score normalized across the population.
2. Students at or below the **35th percentile** are labeled **At-Risk (1)**; all others are labeled **Typical (0)**.
3. The `RMAT` column is then dropped — it is not used as a model feature.

```
Label = 1  if  z(RMAT) ≤ percentile(35)
Label = 0  otherwise
```

### Deployment Features

For deployment, six **derived diagnostic features** are computed from the raw scores to enhance interpretability. These are used in `predict_with_diagnostics()` but are never split nodes in the tree:

| Derived Feature | Formula | Domain |
|---|---|---|
| `NP` | `NC + DM` | Overall Processing Efficiency |
| `SN` | `NC − DM` | Symbolic vs. Non-Symbolic Processing Difference |
| `AF` | `(ADD + SUB + CA) / 3` | Overall Arithmetic Fluency |
| `BC` | `CA − AF` | Basic vs. Complex Arithmetic Contrast |
| `AS` | `ADD − SUB` | Addition vs. Subtraction Asymmetry |
| `PF` | `AF / NP` | Processing-Fluency Integration |

### Splits

All splits use a **70 / 15 / 15** real-data ratio (stratified). The synthetic data augments the training split only.

| Split | Rows | Composition |
|---|---|---|
| Train (TRTR) | 250 | Real only |
| Train (TSTR) | 308 | 250 real + 58 synthetic at-risk |
| Validation | 54 | Real only |
| Test | 54 | Real only |

---

## Model: `C45DecisionTree`

A custom Python implementation of the **C4.5 decision tree** algorithm with error-based pruning and diagnostic output support.

### Files

- `src/C45DecisionTree.py` — Core implementation
- `src/Dataclasses.py` — `Node` and `DiagnosticOutput` dataclasses

### Constructor Parameters

```python
C45DecisionTree(
    max_depth           = None,   # Maximum tree depth (None = unlimited)
    min_samples_split   = 2,      # Minimum samples to attempt a split
    min_samples_leaf    = 1,      # Minimum samples required at a leaf
    conf_fact           = 0.25,   # Confidence factor for error-based pruning
    min_gain_ratio      = 1e-3,   # Minimum gain ratio to accept a split
    max_thresholds      = None,   # Cap on candidate thresholds per feature (quantile sampled)
    feature_domain_mapping = {}   # Maps feature names → domain labels for diagnostics
)
```

**Deployment hyperparameters (fixed from evaluation):**

```python
BEST_PARAMS = {
    "conf_fact":        0.25,
    "min_samples_leaf": 10,
    "max_depth":        5,
}
```

### How It Works

**Splitting criterion:** Gain Ratio (C4.5 standard), with a minimum threshold to prevent trivial splits.

```
GainRatio(X, F) = Gain(X, F) / SplitInfo(X, F)
```

**Error-based pruning:** After the tree is built, subtrees are pruned bottom-up using a pessimistic upper-confidence-bound error estimate (controlled by `conf_fact`). A subtree is collapsed to a leaf when the estimated leaf error ≤ estimated subtree error.

**Feature importance:** Global importance is computed as the gain-ratio-weighted fraction of training samples at each split node, normalized to sum to 1.

```
Importance(F_j) = Σ GainRatio(node_n, F_j) * (|D_n| / |D|)
```

### Key Methods

| Method | Description |
|---|---|
| `fit(X, y, raw_features)` | Train the tree. `X` contains all features; only `raw_features` are used for splits. Full `X` is used to compute feature statistics for diagnostics. |
| `predict(X)` | Return predicted class labels as a NumPy array. |
| `predict_with_diagnostics(X)` | Return a list of `DiagnosticOutput` objects — one per sample — with confidence, decision path, domain severity scores, and task importance scores. |
| `get_feature_importance()` | Return the global feature importance dictionary. |
| `get_depth()` | Return the depth of the fitted tree. |
| `get_leaves_num()` | Return the number of leaf nodes. |
| `save_model(filepath, optimal_threshold)` | Serialize the model package (tree + threshold) to a `.pkl` file via `pickle`. |
| `C45DecisionTree.load_model(filepath)` | Class method. Load and return `(tree, optimal_threshold)` from a `.pkl` file. |

### DiagnosticOutput

Each call to `predict_with_diagnostics()` returns a `DiagnosticOutput` per sample:

```python
@dataclass
class DiagnosticOutput:
    predicted_class:        str              # "0" or "1"
    confidence:             float            # P(predicted class | leaf), Laplace-smoothed
    decision_path:          List[Tuple]      # [(feature, threshold, direction), ...]
    decision_path_readable: str              # Human-readable path string
    domain_severity_scores: Dict[str, float] # Normalized per-domain severity
    task_importance_scores: Dict[str, float] # Normalized per-task importance
    leaf_distribution:      Dict[Any, int]   # Raw class counts at leaf
```

**Domain severity** is computed along the prediction path using information gain as the weight and the sample's z-score as the magnitude indicator. Derived features (NP, SN, AF, BC, AS, PF) contribute via z-score alone (no gain ratio available since they are never split nodes).

**Task importance** uses gain ratio as the weight along the decision path.

Both scores are normalized to sum to 1 per prediction.

---

## Training Pipeline

### Evaluation: `notebooks/tstr_vs_trtr.ipynb`

Runs the full 6-phase evaluation comparing TSTR and TRTR. This notebook is used for research and model selection — not deployment.

**Phases:**
1. **Threshold sweep** — selects classification threshold via F2-score maximization on the validation set using raw tree probabilities.
2. **Cross-validation** — 5-fold stratified CV on real data; synthetic rows pinned to every training fold (never validation folds).
3. **CV summary comparison** — side-by-side TRTR vs TSTR metric table.
4. **Fold variance inspection** — per-fold breakdown with correlation analysis.
5. **Test set evaluation** — final held-out test metrics with CV-to-test drift analysis.
6. **Global feature importance** — tree split importance for the TSTR model.

### Deployment: `train.py`

Trains the final model for deployment. Supports both TSTR (default) and TRTR (`--no-synth`) modes.

**Usage:**

```bash
# TSTR mode (real + synthetic training data)
python train.py --threshold 0.35

# TRTR mode (real data only)
python train.py --threshold 0.35 --no-synth

# Custom output path
python train.py --threshold 0.35 --out models/funa_c45_v1.pkl
```

**Training steps:**
1. Loads real training data and (optionally) synthetic data.
2. Trains a C4.5 tree with fixed deployment hyperparameters.
3. Evaluates on the validation set at the provided threshold (informational only).
4. Evaluates on the held-out test set at the provided threshold.
5. Saves the model package (tree + threshold) to disk.
6. Runs a demonstration on `test_deployment.csv`, printing metrics and 10 sample diagnostics.

**Output `.pkl` structure:**
```python
{
    'model':             C45DecisionTree,   # fitted tree
    'optimal_threshold': float,             # locked classification threshold
}
```

**Domain mapping used in deployment:**

| Feature | Domain Label |
|---|---|
| NC | Number Comparison |
| DM | Digit-Dot Matching |
| NP | Overall Processing Efficiency |
| SN | Symbolic vs. Non-Symbolic Processing Difference |
| NS | Number Series |
| ADD | Single-Digit Addition |
| SUB | Single-Digit Subtraction |
| CA | Multi-Digit Addition and Subtraction |
| AF | Overall Arithmetic Fluency |
| BC | Basic vs. Complex Arithmetic Contrast |
| AS | Addition vs. Subtraction Asymmetry |
| PF | Processing-Fluency Integration |

---

## Evaluation Results Summary

Full results and interpretation are in `documentation/TSTR_results.md`. Key findings:

### Cross-Validation (5-Fold)

| Metric | TRTR | TSTR | Δ |
|---|---|---|---|
| Recall | 0.5832 ± 0.0885 | 0.6458 ± 0.0841 | **+0.0626** |
| Precision | 0.6671 ± 0.0614 | 0.6061 ± 0.1188 | -0.0610 |
| F1 | 0.6194 ± 0.0677 | 0.6127 ± 0.0416 | -0.0068 |
| F2 | 0.5965 ± 0.0793 | 0.6295 ± 0.0574 | **+0.0330** |

### Test Set (n=54, threshold=0.35)

| Metric | TRTR | TSTR |
|---|---|---|
| Recall | 0.5714 | **0.6667** |
| Precision | **0.7059** | 0.6087 |
| F1 | 0.6316 | **0.6364** |
| Accuracy | **0.7407** | 0.7037 |

At n=54, one misclassified sample = 0.048 metric swing. Removing the flawed probability calibrator resolved the massive CV-to-test drift seen previously. Both models now show highly consistent performance between CV and Test (drift < 2.5%). **TSTR** is the clear winner for deployment, as the synthetic data successfully boosted the true Recall (+9.5%) and F2-Score (+6.0%) on the held-out test set compared to the Real-Only model.

### Feature Importance (TSTR Tree)

| Feature | Importance | Domain |
|---|---|---|
| NC | 0.4928 | Number Comparison |
| NS | 0.2584 | Number Series |
| ADD | 0.1876 | Single-Digit Addition |
| SUB | 0.0612 | Single-Digit Subtraction |
| DM | ~0.000 | Digit-Dot Matching |
| CA | ~0.000 | Multi-Digit Arithmetic |

Four features — **NC, NS, ADD, and SUB** — account for 100% of split importance. The tree ignored all incomplete/timeout flags as split nodes.

---

## Dependencies

You can install all required packages via requirements.txt:

```
pip install -r requirements.txt
```

---

## Quick Start

### 1. Label the raw dataset

```bash
python scripts/RMAT_Labeling.py
```

Reads `datasets/raw/FUNADB_rawdata_SUPPL.csv`, produces `datasets/processed/FUNADB_labled.csv`.

---

### 2. Clean, Augment, and Prototype (Notebooks)

Before running the ML scripts, you must generate the split datasets and synthetic samples using the Jupyter notebooks.

- `notebooks/dataset_analysis.ipynb`: Run this to perform Exploratory Data Analysis (EDA) and handle sentinel/missing values. Produces `dataset/processed/cleaned_dataset.csv`.

- `notebooks/synthetic_data_generation.ipynb`: Run this to train the GAN on the minority class and generate the stratified splits. Produces `train.csv`, `val.csv`, `test.csv`, `s_train.csv`, and their deployment counterparts.

- `notebooks/ML_development.ipynb`: (Optional) Use this sandbox to interactively prototype the C4.5 model, test hyperparameters, and visually inspect tree behavior before running the rigid evaluation scripts.

---

### 3. Run TSTR vs TRTR evaluation

Open and execute `notebooks/tstr_vs_trtr.ipynb` in your IDE or Jupyter environment. This runs the full 6-phase evaluation in parallel. Expects the split CSVs to be present in `datasets/processed/`.

### 4. Train the deployment model

```bash
python scripts/train.py --threshold 0.35
```

Trains the final C4.5 tree on the full dataset and saves the artifact to `models/funa_c45.pkl` by default.

### 5. Load and use the model

```python
from src.C45DecisionTree import C45DecisionTree
import pandas as pd

# Load the model package
tree, threshold = C45DecisionTree.load_model("models/v1.pkl")

# Load deployment data
df = pd.read_csv("datasets/processed/test_deployment.csv")
X = df[["NC", "DM", "NS", "ADD", "SUB", "CA", "NP", "SN", "AF", "BC", "AS", "PF"]]

# Run prediction with diagnostic breakdown
diagnostics = tree.predict_with_diagnostics(X)
# In production, use the tree's raw confidence for probabilities
probs = [d.confidence if int(d.predicted_class) == 1 else 1 - d.confidence for d in diagnostics]

predictions = [1 if p >= threshold else 0 for p in probs]

# Inspect a specific diagnostic output
d = diagnostics[0]
print("Path taken:", d.decision_path_readable)
print("Domain Severities:", d.domain_severity_scores)
print("Task Importances:", d.task_importance_scores)
```

---

