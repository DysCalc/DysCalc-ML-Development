# C4.5 Decision Tree — Algorithm Documentation

> **Module:** `src/C45DecisionTree.py`
> **Dataclasses:** `src/Dataclasses.py` (`Node`, `DiagnosticOutput`)

---

## Table of Contents

1. [Overview](#overview)
2. [Algorithm Theory](#algorithm-theory)
   - [Entropy](#entropy)
   - [Information Gain](#information-gain)
   - [Split Information & Gain Ratio](#split-information--gain-ratio)
   - [Error-Based Pruning](#error-based-pruning)
3. [Architecture](#architecture)
   - [Tree Structure](#tree-structure)
   - [Data Classes](#data-classes)
4. [Hyperparameters](#hyperparameters)
5. [Training Pipeline](#training-pipeline)
   - [Splitting Strategy](#splitting-strategy)
   - [Threshold Quantile Sampling](#threshold-quantile-sampling)
   - [Recursive Tree Building](#recursive-tree-building)
   - [Post-Training Pruning](#post-training-pruning)
   - [Feature Importance](#feature-importance)
6. [Inference Pipeline](#inference-pipeline)
   - [Class Prediction](#class-prediction)
   - [Probability Estimation](#probability-estimation)
   - [Diagnostic Prediction](#diagnostic-prediction)
7. [Diagnostic Scoring System](#diagnostic-scoring-system)
   - [Confidence Score](#confidence-score)
   - [Decision Path](#decision-path)
   - [Domain Severity Scores](#domain-severity-scores)
   - [Task Importance Scores](#task-importance-scores)
   - [Derived Feature Handling](#derived-feature-handling)
8. [Model Serialization](#model-serialization)
9. [API Reference](#api-reference)
10. [Usage Examples](#usage-examples)

---

## Overview

`C45DecisionTree` is a custom implementation of the **C4.5 decision tree** algorithm (Quinlan, 1993), extended with clinical diagnostic scoring for dyscalculia screening. Unlike scikit-learn's `DecisionTreeClassifier` (which implements CART), this implementation uses **gain ratio** as its splitting criterion and **error-based pruning** with a confidence factor — both hallmarks of the original C4.5 algorithm.

The classifier is designed to serve two roles:

1. **Binary classification** — predicting whether a student is `At-Risk` or `Typical` for dyscalculia.
2. **Diagnostic interpretability** — producing per-prediction explanations including domain-level severity scores, task-level importance scores, and human-readable decision paths, enabling clinicians to understand *why* a classification was made.

---

## Algorithm Theory

### Entropy

Entropy measures the impurity or uncertainty in a dataset. For a set $S$ with class proportions $p_i$:

$$
H(S) = -\sum_{i=1}^{c} p_i \cdot \log_2(p_i)
$$

- $H(S) = 0$ when the set is **pure** (all samples belong to one class).
- $H(S) = 1$ for a binary set with equal proportions.

**Implementation note:** A small epsilon ($\varepsilon = 10^{-9}$) is added inside the logarithm to prevent `log(0)`.

### Information Gain

Information gain quantifies the reduction in entropy from splitting a set $S$ on feature $F$:

$$
\text{Gain}(S, F) = H(S) - \sum_{v \in \text{values}(F)} \frac{|S_v|}{|S|} \cdot H(S_v)
$$

Where $S_v$ is the subset of $S$ where feature $F$ takes value $v$. Higher gain means the feature better separates the classes.

### Split Information & Gain Ratio

A weakness of information gain is its bias toward features with many distinct values. C4.5 addresses this with the **gain ratio**, which normalizes gain by the feature's **split information**:

$$
\text{SplitInfo}(S, F) = -\sum_{v \in \text{values}(F)} \frac{|S_v|}{|S|} \cdot \log_2\left(\frac{|S_v|}{|S|}\right)
$$

$$
\text{GainRatio}(S, F) = \frac{\text{Gain}(S, F)}{\text{SplitInfo}(S, F)}
$$

This penalizes features that create many small partitions while rewarding features that create a few large, pure partitions.

### Error-Based Pruning

After the full tree is grown, C4.5 applies **error-based pruning (EBP)** bottom-up to reduce overfitting. For each internal node, the algorithm compares:

1. **Subtree error** — the sum of pessimistic error estimates across all leaves of the subtree.
2. **Leaf error** — the pessimistic error estimate if the subtree were replaced by a single leaf.

If collapsing the subtree to a leaf does not increase the pessimistic error, the subtree is pruned.

The pessimistic error uses a **normal approximation to the binomial confidence interval**:

$$
e_{\text{upper}} = \frac{f + z \cdot \sqrt{\frac{f(1 - f)}{N}} + \frac{z^2}{2N}}{1 + \frac{z^2}{N}}
$$

Where:
- $f = \frac{\text{errors}}{N}$ is the observed error rate at the node.
- $z = \Phi^{-1}(1 - \text{cf})$ is the z-score from the confidence factor `cf`.
- A lower `cf` → higher $z$ → more aggressive pruning.

Changing `conf_fact` does not necessarily produce a different tree. The parameter only changes the pessimistic error estimates used in the subtree-versus-leaf pruning comparison. If every internal node reaches the same keep/prune outcome under two different confidence factors, the final SVG tree structure will be identical even though the evaluated configuration values differ.

Probability thresholding is separate from this pruning process. A threshold such as `P(At-Risk) >= 0.40` is applied after training to the leaf probabilities returned by `predict_proba()`. It affects the final class label used for evaluation, but it does not affect gain-ratio split selection, pruning, or the exported tree structure.

---

## Architecture

### Tree Structure

The tree is a binary tree of `Node` objects linked via `left` and `right` pointers:

```
             [Internal Node]
            feature: "task_3"
          threshold: 0.4521
          ┌──────┴──────┐
          │             │
     [≤ 0.4521]   [> 0.4521]
          │             │
       [Leaf]      [Internal]
     At-Risk      feature: "task_7"
                    ...
```

All splits are binary: `feature ≤ threshold` goes left, `feature > threshold` goes right.

### Data Classes

#### `Node`

| Field               | Type                     | Description                                      |
|---------------------|--------------------------|--------------------------------------------------|
| `type`              | `"leaf"` or `"internal"` | Node type                                        |
| `distribution`      | `Counter`                | Class distribution of samples at this node       |
| `label`             | `str` (optional)         | Predicted class (leaf nodes only)                |
| `samples`           | `int` (optional)         | Number of training samples at this node          |
| `feature`           | `str` (optional)         | Split feature name (internal nodes only)         |
| `gain_ratio`        | `float` (optional)       | Gain ratio of the split                          |
| `information_gain`  | `float` (optional)       | Information gain of the split                    |
| `threshold`         | `float` (optional)       | Split threshold value                            |
| `left`              | `Node` (optional)        | Left child (≤ threshold)                         |
| `right`             | `Node` (optional)        | Right child (> threshold)                        |

#### `DiagnosticOutput`

| Field                      | Type                            | Description                                   |
|----------------------------|---------------------------------|-----------------------------------------------|
| `predicted_class`          | `str`                           | Predicted label (e.g., `"1"` for At-Risk)     |
| `confidence`               | `float`                         | Laplace-smoothed leaf-purity probability      |
| `decision_path`            | `List[Tuple[str, float, str]]`  | List of `(feature, threshold, direction)`     |
| `decision_path_readable`   | `str`                           | Human-readable AND-joined path description    |
| `domain_severity_scores`   | `Dict[str, float]`              | Normalized severity per clinical domain       |
| `task_importance_scores`   | `Dict[str, float]`              | Normalized importance per feature/task        |
| `leaf_distribution`        | `Dict[Any, int]`                | Raw class counts at the reached leaf          |

---

## Hyperparameters

| Parameter                | Type             | Default  | Description                                                                                                                                  |
|--------------------------|------------------|----------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `max_depth`              | `int` or `None`  | `None`   | Maximum tree depth. `None` = unlimited growth (pruning controls complexity instead).                                                         |
| `min_samples_split`      | `int`            | `2`      | Minimum samples required at a node to attempt a split. Nodes below this threshold become leaves.                                             |
| `min_samples_leaf`       | `int`            | `1`      | Minimum samples required in each child after a split. Splits that violate this constraint are skipped.                                       |
| `conf_fact`              | `float`          | `0.25`   | Confidence factor for error-based pruning. Lower values = more aggressive pruning. Quinlan's default is `0.25`.                              |
| `min_gain_ratio`         | `float`          | `1e-3`   | Minimum gain ratio to accept a split. Splits below this threshold are rejected, preventing negligible information splits.                    |
| `max_thresholds`         | `int` or `None`  | `None`   | Maximum candidate thresholds evaluated per feature. When set, thresholds are sampled via quantiles for speed on large/augmented datasets.     |
| `feature_domain_mapping` | `Dict[str, str]` | `{}`     | Maps feature names → clinical domain labels (e.g., `"task_3" → "Number Processing"`). Required for domain-level severity scoring.           |

---

## Training Pipeline

### Splitting Strategy

For each feature at each node, the algorithm:

1. Extracts all unique values of the feature.
2. Computes **midpoints** between consecutive sorted unique values as candidate thresholds.
3. For each candidate threshold, splits the data into left ($\leq$) and right ($>$) subsets.
4. Calculates the **gain ratio** of the split.
5. Selects the threshold with the highest gain ratio.

The best feature across all features (highest gain ratio above `min_gain_ratio`) is chosen as the split.

### Threshold Quantile Sampling

When `max_thresholds` is set, the full set of midpoints is downsampled using **uniform quantile sampling**:

```python
quantiles = np.linspace(0, 100, max_thresholds)
thresholds = np.unique(np.percentile(all_midpoints, quantiles))
```

This ensures even coverage of the feature's range while bounding the computational cost — critical when training on augmented synthetic datasets that may introduce thousands of unique values per feature.

### Recursive Tree Building

The `_build_tree` method recursively partitions the data. A leaf node is created when any stopping condition is met:

| Condition                               | Outcome         |
|-----------------------------------------|-----------------|
| All samples have the same class         | Pure leaf        |
| Fewer than `min_samples_split` samples  | Majority leaf    |
| Depth ≥ `max_depth`                     | Majority leaf    |
| No valid split found (gain ratio ≤ threshold or no feature can split) | Majority leaf    |

### Post-Training Pruning

After the full tree is grown, `_prune_tree` traverses the tree **bottom-up**:

1. Recursively prune children first.
2. At each internal node, compare the pessimistic subtree error vs. pessimistic leaf error.
3. If replacing the subtree with a leaf doesn't increase error, **prune** (replace with majority-class leaf).

This produces a simpler tree that generalizes better, especially important given the small clinical dataset.

### Feature Importance

After pruning, **global feature importance** is calculated by traversing the tree:

$$
\text{Importance}(F_j) = \sum_{n \in \text{nodes splitting on } F_j} \text{GainRatio}(n) \cdot \frac{|D_n|}{|D|}
$$

Where $|D_n|$ is the number of training samples at node $n$ and $|D|$ is the total training set size. Scores are then **normalized** to sum to 1.0.

---

## Inference Pipeline

### Class Prediction

`predict(X)` traverses each sample through the tree from root to leaf:

- At each internal node, compare `x[feature]` against the node's `threshold`.
- Go **left** if `x[feature] ≤ threshold`, otherwise go **right**.
- Return the majority class at the reached leaf.

> **Note:** `predict()` uses only the `raw_features` (the features used during tree construction), not the full diagnostic feature set.

### Probability Estimation

`predict_proba(X, positive_class=1)` returns the probability of the positive class at each sample's leaf node using **Laplace smoothing**:

$$
P(y = c \mid \text{leaf}) = \frac{n_c + 1}{n_{\text{total}} + K}
$$

Where $K$ is the number of classes. This prevents probability estimates of exactly 0 or 1 from pure leaves and enables threshold-based inference.

### Diagnostic Prediction

`predict_with_diagnostics(X)` produces a full `DiagnosticOutput` per sample. It combines the tree traversal with additional scoring — see [Diagnostic Scoring System](#diagnostic-scoring-system) below.

---

## Diagnostic Scoring System

The diagnostic system extends the base C4.5 algorithm to provide clinically interpretable explanations. All equations reference the research proposal numbering.

### Confidence Score

**(Eq. 3.34)**

$$
P(y = c \mid X_i) = \frac{n_c + 1}{n_{\text{At-Risk}} + n_{\text{Typical}} + K}
$$

A Laplace-corrected probability based on the class distribution at the reached leaf node. The +1 correction avoids assigning P=1.0 on pure leaves that never observed the minority class during training.

### Decision Path

**(Eq. 3.38)**

$$
\text{Path}_i = \{(F_{n_j},\ \theta_{n_j},\ \text{dir}_{n_j}), \ldots\}
$$

A list of `(feature, threshold, direction)` tuples representing the sequence of decisions made to reach the leaf. Also stored as a human-readable string:

```
task_3 <= 0.4521 AND task_7 > 0.6103 AND task_12 <= 0.3280
```

### Domain Severity Scores

**(Eq. 3.35)**

$$
DS_{i,d} = \sum_{n \in \text{path}} w_n \cdot z_{i,n} \cdot \mathbb{1}[\text{feature}_n \in \text{domain } d]
$$

Where:
- $w_n$ = **information gain** at node $n$ (the raw reduction in entropy).
- $z_{i,n}$ = the absolute z-score of the student's feature value at node $n$, computed against training population statistics: $z = \frac{|x - \mu|}{\sigma + \varepsilon}$.
- $\mathbb{1}[\cdot]$ = indicator function, activated when the split feature belongs to domain $d$.

Scores are **normalized** so that all domains sum to 1.0, producing a severity profile across clinical domains (e.g., Number Processing, Spatial Numerosity, etc.).

### Task Importance Scores

**(Eq. 3.36)**

$$
\text{TaskImp}_{i,t} = \sum_{n \in \text{path}} \text{GainRatio}(n) \cdot z_{i,n} \cdot \mathbb{1}[\text{split feature}_n = t]
$$

Similar to domain severity, but:
- Uses **gain ratio** (not information gain) as the weight.
- Operates at the **individual feature/task** level rather than the domain level.

Scores are normalized to sum to 1.0.

### Derived Feature Handling

Features that are **diagnostic-only** (derived features like composite scores: NP, SN, AF, BC, AS, PF) never appear as tree split nodes because the tree is trained only on `raw_features`. These derived features receive scores via a **post-path extension**:

This raw-only split policy is intentional. The derived features are deterministic functions of the same raw task scores used for model fitting, so including both raw and derived versions in split selection would give the tree multiple redundant ways to partition on the same underlying information. For the FUNA-DB sample size, that redundancy can encourage overly specific branches and weaken generalization. The implementation therefore uses raw task features for supervised fitting and reserves derived features for post-hoc diagnostic interpretation.

$$
\text{TaskImp}_{i,\text{derived}} = z_{i,\text{derived}}
$$

$$
DS_{i,d(\text{derived})} \mathrel{+}= z_{i,\text{derived}}
$$

This ensures derived features contribute to the diagnostic profile proportionally to how anomalous the student's value is relative to the training population, consistent with the proposal's intent that these features "enhance interpretability and capture domain-specific deficits."

---

## Model Serialization

### `save_model(filepath, optimal_threshold=0.50)`

Serializes the entire model object along with its calibrated decision threshold into a pickle file:

```python
model_package = {
    'model':             self,                # Full C45DecisionTree instance
    'optimal_threshold': optimal_threshold,   # Calibrated probability cutoff
    'conf_fact':         self.conf_fact,
    'min_samples_leaf':  self.min_samples_leaf,
    'max_depth':         self.max_depth,
    'epsilon':           self.epsilon,
}
```

The `optimal_threshold` is locked at save time — typically calibrated during evaluation (e.g., via Youden's J or F2-optimized search) to maximize clinical recall.

### `load_model(filepath)` (classmethod)

Returns a tuple of `(model, optimal_threshold, conf_fact, min_samples_leaf, max_depth, epsilon)`.

---

## API Reference

### Constructor

```python
C45DecisionTree(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    conf_fact=0.25,
    min_gain_ratio=1e-3,
    max_thresholds=None,
    feature_domain_mapping=None
)
```

### Methods

| Method                          | Returns                    | Description                                                  |
|---------------------------------|----------------------------|--------------------------------------------------------------|
| `fit(X, y, raw_features)`      | `self`                     | Train the tree. `X` is the full feature DataFrame, `raw_features` is the subset used for splits. |
| `predict(X)`                    | `np.ndarray`               | Predict class labels.                                        |
| `predict_proba(X, positive_class=1)` | `np.ndarray`          | Predict positive-class probabilities (Laplace-smoothed).     |
| `predict_with_diagnostics(X)`   | `List[DiagnosticOutput]`   | Predict with full diagnostic outputs.                        |
| `get_feature_importance()`      | `Dict[str, float]`         | Normalized global feature importance scores.                 |
| `print_tree(node, depth, prefix)` | `None`                  | Print tree structure to stdout.                              |
| `get_depth(node)`               | `int`                      | Get the maximum depth of the tree.                           |
| `get_leaves_num(node)`          | `int`                      | Count the number of leaf nodes.                              |
| `save_model(filepath, optimal_threshold)` | `None`           | Serialize model + threshold to pickle file.                  |
| `load_model(filepath)`          | `tuple`                    | *Classmethod.* Load model from pickle file.                  |

---

## Usage Examples

### Training

```python
from src.C45DecisionTree import C45DecisionTree

DOMAIN_MAP = {
    "task_1": "Number Processing",
    "task_2": "Number Processing",
    "task_3": "Spatial Numerosity",
    # ... full mapping
}

raw_features = ["task_1", "task_2", "task_3", ...]  # raw features used for tree splits

tree = C45DecisionTree(
    max_depth=6,
    min_samples_leaf=3,
    conf_fact=0.25,
    max_thresholds=50,
    feature_domain_mapping=DOMAIN_MAP
)

tree.fit(X_train, y_train, raw_features=raw_features)
```

### Prediction

```python
# Class labels
predictions = tree.predict(X_test)

# Probabilities (for threshold-based inference)
probabilities = tree.predict_proba(X_test, positive_class=1)
at_risk = (probabilities >= optimal_threshold).astype(int)
```

### Diagnostic Output

```python
diagnostics = tree.predict_with_diagnostics(X_test)

for diag in diagnostics:
    print(f"Class: {diag.predicted_class}")
    print(f"Confidence: {diag.confidence:.3f}")
    print(f"Path: {diag.decision_path_readable}")
    print(f"Domain Severity: {diag.domain_severity_scores}")
    print(f"Task Importance: {diag.task_importance_scores}")
```

### Serialization

```python
# Save with calibrated threshold
tree.save_model("models/<model_name/version>.pkl", optimal_threshold=0.42)

# Load
model, threshold, *_ = C45DecisionTree.load_model("models/<model_name/version>.pkl")
```

### Inspection

```python
print(f"Tree depth: {tree.get_depth()}")
print(f"Leaf count: {tree.get_leaves_num()}")
print(f"Feature importance: {tree.get_feature_importance()}")
tree.print_tree()
```
