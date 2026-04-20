import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score, f1_score, recall_score, precision_score, accuracy_score
from C45DecisionTree import C45DecisionTree
from sklearn.isotonic import IsotonicRegression
from pathlib import Path

def fit_calibrator(probs: List[float], y_true: pd.Series):
    """Fit isotonic regression calibrator."""
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(probs, y_true)
    return calibrator

def apply_calibrator(calibrator, probs: List[float]) -> List[float]:
    """Apply fitted calibrator to probabilities."""
    return calibrator.transform(probs)

# -----------------------------
# Feature + Domain Definitions
# -----------------------------
FUNA_DB_DOMAIN_MAPPING: Dict[str, str] = {
    "NC":  "Number Comparison",
    "DM":  "Digit-Dot Matching",
    "NP":  "Overall Processing Efficiency",
    "SN":  "Symbolic vs. Non-Symbolic Processing Difference",
    "NS":  "Number Series",
    "ADD": "Single-Digit Addition",
    "SUB": "Single-Digit Subtraction",
    "CA":  "Multi-Digit Addition and Subtraction",
    "AF":  "Overall Arithmetic Fluency",
    "BC":  "Basic vs. Complex Arithmetic Contrast",
    "AS":  "Addition vs. Subtraction Asymmetry",
    "PF":  "Processing-Fluency Integration",
}
FUNA_DB_RAW_FEATURES: List[str] = [
    "NC", "DM", "NS", "ADD", "SUB", "CA",
    "NC_incomplete", "DM_incomplete", "NS_incomplete",
    "ADD_incomplete", "SUB_incomplete", "CA_incomplete",
]
FUNA_DB_DIAGNOSTIC_FEATURES: List[str] = [
    "NC", "DM", "NS", "ADD", "SUB", "CA",
    "NP", "SN", "AF", "BC", "AS", "PF",
    "NC_incomplete", "DM_incomplete", "NS_incomplete",
    "ADD_incomplete", "SUB_incomplete", "CA_incomplete",
]

# Features that signal missing/incomplete data.
# The tree may still split on these (they carry real signal),
# but we exclude them from diagnostic z-score severity so that
# "timed out" is never conflated with "performed poorly".
INCOMPLETE_FLAGS = {
    "NC_incomplete", "DM_incomplete", "NS_incomplete",
    "ADD_incomplete", "SUB_incomplete", "CA_incomplete"
}

# -----------------------------
# Helpers
# -----------------------------
def split_xy(df: pd.DataFrame, label_col: str = 'Label'):
    X = df[FUNA_DB_DIAGNOSTIC_FEATURES]
    y = df[label_col]
    return X, y

def get_probs(tree, X: pd.DataFrame) -> List[float]:
    """
    Extract classification probabilities from diagnostic outputs.
    Incomplete flags are zeroed out from task_importance AFTER prediction —
    the tree's split decisions are unaffected.
    """
    diagnostics = tree.predict_with_diagnostics(X)

    probs = []
    for d in diagnostics:
        # Zero out incomplete flags from task importance
        for flag in INCOMPLETE_FLAGS:
            if flag in d.task_importance_scores:
                d.task_importance_scores[flag] = 0.0

        # Re-normalize task importance after zeroing
        total_task = sum(d.task_importance_scores.values())
        if total_task > 0:
            for f in d.task_importance_scores:
                d.task_importance_scores[f] /= total_task

        # Note: domain_severity is not re-zeroed here because incomplete flags
        # don't map to any domain in FUNA_DB_DOMAIN_MAPPING — they only appear
        # in task_importance. If you add them to the mapping later, zero them
        # out from domain_severity here too.

        p = d.confidence if int(d.predicted_class) == 1 else 1 - d.confidence
        probs.append(p)

    return probs

def probs_to_preds(probs: List[float], threshold: float) -> List[int]:
    return [1 if p >= threshold else 0 for p in probs]

def compute_metrics(y_true, preds) -> dict:
    return {
        'f1':        f1_score(y_true, preds, zero_division=0),
        'fbeta':     fbeta_score(y_true, preds, beta=2, zero_division=0),
        'recall':    recall_score(y_true, preds, zero_division=0),
        'precision': precision_score(y_true, preds, zero_division=0),
        'accuracy':  accuracy_score(y_true, preds),
    }

def run_cv(X_train: pd.DataFrame, y_train: pd.Series, label: str,
           best_params: dict, threshold: float,
           n_real: Optional[int] = None) -> tuple[dict, list[dict]]:

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_f1, cv_fbeta, cv_recall, cv_precision, cv_accuracy = [], [], [], [], []
    fold_records = []

    if n_real is not None:
        X_real  = X_train.iloc[:n_real].reset_index(drop=True)
        y_real  = y_train.iloc[:n_real].reset_index(drop=True)
        X_synth = X_train.iloc[n_real:].reset_index(drop=True)
        y_synth = y_train.iloc[n_real:].reset_index(drop=True)
    else:
        X_real, y_real = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
        X_synth, y_synth = X_train.iloc[:0], y_train.iloc[:0]

    print(f'\nStarting 5-Fold Cross-Validation [{label}]...')
    if n_real is not None:
        print(f'  (folding over {len(X_real)} real samples; '
              f'{len(X_synth)} synthetic rows pinned to every train fold)')

    for fold, (train_index, val_index) in enumerate(skf.split(X_real, y_real), 1):

        X_val_fold = X_real.iloc[val_index]
        y_val_fold = y_real.iloc[val_index]

        X_trn_fold = pd.concat([X_real.iloc[train_index], X_synth], ignore_index=True)
        y_trn_fold = pd.concat([y_real.iloc[train_index], y_synth], ignore_index=True)

        tree = C45DecisionTree(**best_params, feature_domain_mapping=FUNA_DB_DOMAIN_MAPPING)
        tree.fit(X_trn_fold, y_trn_fold,
                 raw_features=FUNA_DB_RAW_FEATURES)

        # 🔥 NEW: calibration per fold
        probs_train = get_probs(tree, X_trn_fold)
        calibrator  = fit_calibrator(probs_train, y_trn_fold)

        probs_val = apply_calibrator(calibrator, get_probs(tree, X_val_fold))
        preds     = probs_to_preds(probs_val, threshold)
        m         = compute_metrics(y_val_fold, preds)

        cv_f1.append(m['f1'])
        cv_fbeta.append(m['fbeta'])
        cv_recall.append(m['recall'])
        cv_precision.append(m['precision'])
        cv_accuracy.append(m['accuracy'])

        fold_records.append({
            'fold': fold,
            'recall': m['recall'],
            'fbeta': m['fbeta'],
            'tree_depth': tree.get_depth(),
            'tree_leaves': tree.get_leaves_num(),
            'n_train':       len(X_trn_fold),
            'class_0_train': int((y_trn_fold == 0).sum()),
            'class_1_train': int((y_trn_fold == 1).sum()),
            'n_val':         len(X_val_fold),
            'class_0_val':   int((y_val_fold == 0).sum()),
            'class_1_val':   int((y_val_fold == 1).sum()),
        })

        print(f'  Fold {fold}: F1={m["f1"]:.4f}, F2={m["fbeta"]:.4f}, Recall={m["recall"]:.4f}')

    aggregated = {
        'mean_f1': np.mean(cv_f1), 'std_f1': np.std(cv_f1),
        'mean_fbeta': np.mean(cv_fbeta), 'std_fbeta': np.std(cv_fbeta),
        'mean_recall': np.mean(cv_recall), 'std_recall': np.std(cv_recall),
        'mean_precision': np.mean(cv_precision), 'std_precision': np.std(cv_precision),
        'mean_accuracy': np.mean(cv_accuracy), 'std_accuracy': np.std(cv_accuracy),
    }

    return aggregated, fold_records

def print_cv_results(label: str, metrics: dict, threshold: float):
    print(f'\n=== Cross-Validation Performance [{label}] (5-Fold) ===')
    print(f'  Recall:    {metrics["mean_recall"]:.4f} +/- {metrics["std_recall"]:.4f}')
    print(f'  Precision: {metrics["mean_precision"]:.4f} +/- {metrics["std_precision"]:.4f}')
    print(f'  F1-Score:  {metrics["mean_f1"]:.4f} +/- {metrics["std_f1"]:.4f}')
    print(f'  F2-Score:  {metrics["mean_fbeta"]:.4f} +/- {metrics["std_fbeta"]:.4f}')
    print(f'  Accuracy:  {metrics["mean_accuracy"]:.4f} +/- {metrics["std_accuracy"]:.4f}')
    print(f'\n  Note: Metrics evaluated at threshold = {threshold}')

def print_variance_inspection(label: str, fold_records: list[dict]):
    """
    Print per-fold breakdown to diagnose what drives recall variance.
    Flags folds whose recall deviates more than 1 std from the mean.
    """
    df = pd.DataFrame(fold_records)
    mean_recall = df['recall'].mean()
    std_recall  = df['recall'].std()

    print(f'\n--- Fold Variance Inspection [{label}] ---')
    print(f'{"Fold":>5} {"N_train":>8} {"C0_tr":>6} {"C1_tr":>6} '
          f'{"N_val":>6} {"C0_val":>7} {"C1_val":>7} '
          f'{"Depth":>6} {"Leaves":>7} {"Recall":>8} {"F2":>8} {"Flag"}')
    print('-' * 95)

    for _, row in df.iterrows():
        flag = ' ⚠ HIGH' if row['recall'] > mean_recall + std_recall else \
               ' ⚠ LOW'  if row['recall'] < mean_recall - std_recall else ''
        print(f'{int(row["fold"]):>5} {int(row["n_train"]):>8} {int(row["class_0_train"]):>6} '
              f'{int(row["class_1_train"]):>6} {int(row["n_val"]):>6} '
              f'{int(row["class_0_val"]):>7} {int(row["class_1_val"]):>7} '
              f'{int(row["tree_depth"]):>6} {int(row["tree_leaves"]):>7} '
              f'{row["recall"]:>8.4f} {row["fbeta"]:>8.4f}{flag}')

    print(f'\n  Recall range : {df["recall"].min():.4f} - {df["recall"].max():.4f}')
    print(f'  Recall mean  : {mean_recall:.4f} +/- {std_recall:.4f}')

    corr_c1    = df['class_1_train'].corr(df['recall'])
    corr_depth = df['tree_depth'].corr(df['recall'])
    print(f'\n  Correlation: class_1_train vs recall = {corr_c1:+.3f}')
    print(f'  Correlation: tree_depth     vs recall = {corr_depth:+.3f}')
    if abs(corr_c1) > 0.6:
        print('  ⚠ Strong correlation with positive-class count in training fold.')
        print('    Synthetic data may not be uniformly covering all subgroups.')

# -----------------------------
# Load Data
# -----------------------------
print('Loading 70/15/15 Datasets...')
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "datasets"

r_train = pd.read_csv(DATASET_DIR / "train.csv")
s_train = pd.read_csv(DATASET_DIR / "s_train.csv")
val_df  = pd.read_csv(DATASET_DIR / "val.csv")
test_df = pd.read_csv(DATASET_DIR / "test.csv")

# -----------------------------
# Prepare Train Sets
# -----------------------------
train_trtr = r_train.copy()                                     # TRTR: real only
train_tstr = pd.concat([r_train, s_train], ignore_index=True)  # TSTR: real + synthetic

print(f'TRTR Train shape (Real 70%):         {train_trtr.shape}')
print(f'TSTR Train shape (Real + Synth):     {train_tstr.shape}')
print(f'Validation shape (Real 15%):         {val_df.shape}')
print(f'Test shape (Real 15%):               {test_df.shape}')

print(f'\nSynthetic data class distribution:')
print(s_train['Label'].value_counts().rename({0: 'Typical (0)', 1: 'At-Risk (1)'}).to_string())

# -----------------------------
# Split X / y
# -----------------------------
X_train_trtr, y_train_trtr = split_xy(train_trtr)
X_train_tstr, y_train_tstr = split_xy(train_tstr)
X_val,  y_val  = split_xy(val_df)
X_test, y_test = split_xy(test_df)

# -----------------------------
# Hyperparameters
# -----------------------------
best_params = {
    'conf_fact':        0.25,
    'min_samples_leaf': 3,
    'max_depth':        10,
}
print('\nUsing fixed hyperparameters:')
print(best_params)

# ═══════════════════════════════════════════════════════════
# PHASE 1 — Threshold Sweep on Validation Set
# Train on full TRTR, sweep thresholds, pick best by F2.
# Calibrator is fitted on TRTR train probabilities (in-sample)
# so that val is used purely for threshold selection.
# This locked threshold is reused for both TRTR and TSTR.
# ═══════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('PHASE 1: Threshold Sweep (trained on TRTR, evaluated on val)')
print('=' * 60)

THRESHOLDS = np.arange(0.35, 0.76, 0.05).round(2)

def tune_threshold(X_train, y_train, X_val, y_val, label):

    tree = C45DecisionTree(**best_params, feature_domain_mapping=FUNA_DB_DOMAIN_MAPPING)
    tree.fit(X_train, y_train, raw_features=FUNA_DB_RAW_FEATURES)

    # calibrate using train (consistent with your design)
    train_probs = get_probs(tree, X_train)
    calibrator  = fit_calibrator(train_probs, y_train)

    val_probs = apply_calibrator(calibrator, get_probs(tree, X_val))

    best_thresh = None
    best_f2 = -1

    for t in THRESHOLDS:
        preds = probs_to_preds(val_probs, t)
        f2 = fbeta_score(y_val, preds, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thresh = t

    print(f'✔ [{label}] Best threshold = {best_thresh:.2f} (F2={best_f2:.4f})')
    return best_thresh

print('\nTuning thresholds separately...')

BEST_THRESHOLD_TRTR = tune_threshold(
    X_train_trtr, y_train_trtr, X_val, y_val, 'TRTR'
)

BEST_THRESHOLD_TSTR = tune_threshold(
    X_train_tstr, y_train_tstr, X_val, y_val, 'TSTR'
)

# ═══════════════════════════════════════════════════════════
# PHASE 2 — TRTR vs TSTR Cross-Validation
# Both conditions evaluated at BEST_THRESHOLD.
# TSTR folds only over real samples; synthetic rows are pinned
# to every training fold (Option A — no val leakage).
# No internal calib split — threshold was locked in Phase 1.
# ═══════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print(f'PHASE 2: TRTR vs TSTR CV  (trtr_threshold = {BEST_THRESHOLD_TRTR:.2f} | tstr = {BEST_THRESHOLD_TSTR})')
print('=' * 60)

cv_metrics_trtr, fold_records_trtr = run_cv(
    X_train_trtr, y_train_trtr, 'TRTR', best_params, BEST_THRESHOLD_TRTR
)
cv_metrics_tstr, fold_records_tstr = run_cv(
    X_train_tstr, y_train_tstr, 'TSTR', best_params, BEST_THRESHOLD_TSTR,
    n_real=len(r_train)
)

print_cv_results('TRTR', cv_metrics_trtr, BEST_THRESHOLD_TRTR)
print_cv_results('TSTR', cv_metrics_tstr, BEST_THRESHOLD_TSTR)

# ═══════════════════════════════════════════════════════════
# PHASE 3 — Side-by-side CV Summary
# ═══════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('PHASE 3: CV Summary Comparison')
print('=' * 60)

metrics_labels = [
    ('Recall',    'mean_recall',    'std_recall'),
    ('Precision', 'mean_precision', 'std_precision'),
    ('F1-Score',  'mean_f1',        'std_f1'),
    ('F2-Score',  'mean_fbeta',     'std_fbeta'),
    ('Accuracy',  'mean_accuracy',  'std_accuracy'),
]

print(f'\n{"Metric":>12}  {"TRTR (mean+/-std)":>26}  {"TSTR (mean+/-std)":>26}  {"D (TSTR-TRTR)":>15}')
print('-' * 85)
for name, mean_key, std_key in metrics_labels:
    trtr_val = cv_metrics_trtr[mean_key]
    tstr_val = cv_metrics_tstr[mean_key]
    delta    = tstr_val - trtr_val
    sign     = '+' if delta >= 0 else ''
    print(f'{name:>12}  '
          f'{trtr_val:.4f} +/- {cv_metrics_trtr[std_key]:.4f}         '
          f'{tstr_val:.4f} +/- {cv_metrics_tstr[std_key]:.4f}         '
          f'{sign}{delta:.4f}')

# ═══════════════════════════════════════════════════════════
# PHASE 4 — Fold Variance Inspection
# ═══════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('PHASE 4: Fold Variance Inspection')
print('=' * 60)

print_variance_inspection('TRTR', fold_records_trtr)
print_variance_inspection('TSTR', fold_records_tstr)

# ═══════════════════════════════════════════════════════════
# PHASE 4b — TSTR Outlier Fold Investigation
# Dynamically identifies the lowest-recall TSTR fold and
# inspects the real at-risk cases in its val split to surface
# subgroups the synthetic data may not be covering well.
# ═══════════════════════════════════════════════════════════

# Detect outlier fold before printing header
tstr_fold_df     = pd.DataFrame(fold_records_tstr)
outlier_fold_no  = int(tstr_fold_df.loc[tstr_fold_df['recall'].idxmin(), 'fold'])
outlier_recall   = tstr_fold_df.loc[tstr_fold_df['recall'].idxmin(), 'recall']
mean_tstr_recall = tstr_fold_df['recall'].mean()

print('\n' + '=' * 60)
print(f'PHASE 4b: TSTR Fold {outlier_fold_no} Outlier Investigation')
print('=' * 60)

print(f'\n  Outlier fold: Fold {outlier_fold_no} '
      f'(recall={outlier_recall:.4f} vs mean={mean_tstr_recall:.4f})')

# Re-run the same split used in Phase 2 to recover val indices.
# CV in Phase 2 folded over real samples only — replicate that here.
X_real_inv = X_train_tstr.iloc[:len(r_train)].reset_index(drop=True)
y_real_inv = y_train_tstr.iloc[:len(r_train)].reset_index(drop=True)

skf_inv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold_i, (_, val_idx) in enumerate(skf_inv.split(X_real_inv, y_real_inv), 1):
    if fold_i == outlier_fold_no:
        outlier_val_idx = val_idx
        break

# Recover real val rows for the outlier fold
real_val_fold_df = r_train.iloc[outlier_val_idx].copy()
score_cols       = ["NC", "DM", "NS", "ADD", "SUB", "CA"]

print(f'\n  Val split composition for Fold {outlier_fold_no}:')
print(f'    Real samples: {len(real_val_fold_df)}  '
      f'(class 0: {int((real_val_fold_df["Label"]==0).sum())}  '
      f'class 1: {int((real_val_fold_df["Label"]==1).sum())})')
print(f'    Synthetic samples: 0  (all synthetic pinned to train — Option A)')

real_pos_val = real_val_fold_df[real_val_fold_df['Label'] == 1]

if len(real_pos_val) > 0:
    print(f'\n  Real at-risk cases in Fold {outlier_fold_no} val (n={len(real_pos_val)}):')
    print(f'  {"Feature":>6} {"Mean":>8} {"Std":>8} {"Min":>8} {"Max":>8}')
    print('  ' + '-' * 44)
    for col in score_cols:
        if col in real_pos_val.columns:
            vals = real_pos_val[col]
            print(f'  {col:>6} {vals.mean():>8.3f} {vals.std():>8.3f} '
                  f'{vals.min():>8.3f} {vals.max():>8.3f}')

    real_pos_train = r_train.iloc[
        [i for i in range(len(r_train)) if i not in outlier_val_idx]
    ]
    real_pos_train = real_pos_train[real_pos_train['Label'] == 1]

    if len(real_pos_train) > 0:
        print(f'\n  Real at-risk cases in Fold {outlier_fold_no} training (n={len(real_pos_train)}):')
        print(f'  {"Feature":>6} {"Mean":>8} {"Std":>8} {"Min":>8} {"Max":>8}')
        print('  ' + '-' * 44)
        for col in score_cols:
            if col in real_pos_train.columns:
                vals = real_pos_train[col]
                print(f'  {col:>6} {vals.mean():>8.3f} {vals.std():>8.3f} '
                      f'{vals.min():>8.3f} {vals.max():>8.3f}')

    print(f'\n  If the val group has systematically different feature ranges,')
    print(f'  those subgroups are underrepresented in your synthetic data.')
else:
    print(f'\n  No real at-risk cases in Fold {outlier_fold_no} val split.')
    print(f'  Recall = 0 is expected — no positive examples to predict.')

# ═══════════════════════════════════════════════════════════
# PHASE 5 — Final Test Set Evaluation
# Train on full TRTR / TSTR.
# Fit calibrator on the full val set (val was used only for
# threshold selection in Phase 1 — not for calibration —
# so reusing it here for calibration is clean).
# Evaluate on held-out test set at BEST_THRESHOLD.
# ═══════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print(f'PHASE 5: Test Set Evaluation  (TRTR threshold = {BEST_THRESHOLD_TRTR:.2f} | TSTR threshold = {BEST_THRESHOLD_TSTR:.2f})')
print('=' * 60)

results = {}
threshold_map = {
    'TRTR': BEST_THRESHOLD_TRTR,
    'TSTR': BEST_THRESHOLD_TSTR
}
for label, X_tr, y_tr in [('TRTR', X_train_trtr, y_train_trtr),
                            ('TSTR', X_train_tstr, y_train_tstr)]:
    final_tree = C45DecisionTree(**best_params, feature_domain_mapping=FUNA_DB_DOMAIN_MAPPING)
    final_tree.fit(X_tr, y_tr, raw_features=FUNA_DB_RAW_FEATURES)

    # Fit calibrator on full val set (clean — threshold already locked)
    val_probs_raw = get_probs(final_tree, X_val)
    calibrator    = fit_calibrator(val_probs_raw, y_val)

    # Evaluate on test
    test_probs_raw  = get_probs(final_tree, X_test)
    test_probs      = apply_calibrator(calibrator, test_probs_raw)
    test_preds      = probs_to_preds(test_probs, threshold_map[label])
    m               = compute_metrics(y_test, test_preds)
    results[label]  = m

    # Keep TSTR final tree for Phase 6
    if label == 'TSTR':
        tstr_final_tree = final_tree

    print(f'\n  [{label}] Test Set Results:')
    print(f'    Recall:    {m["recall"]:.4f}')
    print(f'    Precision: {m["precision"]:.4f}')
    print(f'    F1-Score:  {m["f1"]:.4f}')
    print(f'    F2-Score:  {m["fbeta"]:.4f}')
    print(f'    Accuracy:  {m["accuracy"]:.4f}')

# Side-by-side test comparison
test_metrics_labels = [
    ('Recall',    'recall'),
    ('Precision', 'precision'),
    ('F1-Score',  'f1'),
    ('F2-Score',  'fbeta'),
    ('Accuracy',  'accuracy'),
]
print(f'\n{"Metric":>12} {"TRTR (test)":>14} {"TSTR (test)":>14} {"D (TSTR-TRTR)":>15}')
print('-' * 58)
for name, key in test_metrics_labels:
    trtr_val = results['TRTR'][key]
    tstr_val = results['TSTR'][key]
    delta    = tstr_val - trtr_val
    sign     = '+' if delta >= 0 else ''
    print(f'{name:>12}  {trtr_val:>12.4f}  {tstr_val:>12.4f}  {sign}{delta:.4f}')

# CV vs Test consistency check
print(f'\n--- CV vs Test Consistency Check ---')
for label, cv_m, test_m in [('TRTR', cv_metrics_trtr, results['TRTR']),
                              ('TSTR', cv_metrics_tstr, results['TSTR'])]:
    recall_drift = test_m['recall'] - cv_m['mean_recall']
    f2_drift     = test_m['fbeta']  - cv_m['mean_fbeta']
    sign_r = '+' if recall_drift >= 0 else ''
    sign_f = '+' if f2_drift     >= 0 else ''
    print(f'  [{label}]  Recall drift (test - CV mean): {sign_r}{recall_drift:.4f}  |  '
          f'F2 drift: {sign_f}{f2_drift:.4f}')
    if abs(recall_drift) > 0.10:
        print(f'    ⚠ Recall drift > 0.10 — possible overfitting or distribution mismatch.')

# ═══════════════════════════════════════════════════════════
# PHASE 6 — Global Feature Importance Analysis
# Uses the TSTR final tree from Phase 5.
# Split into (a) incomplete flags and (b) task features to
# answer two questions:
#   1. Are incomplete flags genuinely used as split criteria?
#   2. Which task features drive the tree's decisions?
# ═══════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('PHASE 6: Global Feature Importance (TSTR tree)')
print('=' * 60)

importance = tstr_final_tree.get_feature_importance()

flag_importance = {k: v for k, v in importance.items() if k in INCOMPLETE_FLAGS}
task_importance = {k: v for k, v in importance.items() if k not in INCOMPLETE_FLAGS}

flag_importance = dict(sorted(flag_importance.items(), key=lambda x: x[1], reverse=True))
task_importance = dict(sorted(task_importance.items(), key=lambda x: x[1], reverse=True))

print('\n  Incomplete flag importance (are they used as split nodes?):')
if flag_importance:
    for feat, score in flag_importance.items():
        bar = '#' * int(score * 40)
        print(f'    {feat:<20} {score:.4f}  {bar}')
    total_flag_share = sum(flag_importance.values())
    print(f'\n  -> Flags account for {total_flag_share:.2%} of total tree importance.')
    if total_flag_share > 0.05:
        print('  [OK] Incomplete flags carry meaningful split weight -- retain them as features.')
    else:
        print('  [i]  Flags have negligible split weight -- the tree largely ignores them.')
else:
    print('    (none -- the tree did not split on any incomplete flag)')
    print('  [i]  Incomplete flags were never chosen as split nodes.')

print('\n  Task feature importance (excluding incomplete flags):')
for feat, score in task_importance.items():
    bar = '#' * int(score * 40)
    print(f'    {feat:<20} {score:.4f}  {bar}')


FLAG_IMPORTANCE_THRESHOLD = 0.01

used_flags   = [f for f, s in flag_importance.items() if s >= FLAG_IMPORTANCE_THRESHOLD]
unused_flags = [f for f, s in flag_importance.items() if s <  FLAG_IMPORTANCE_THRESHOLD]

print('\n' + '=' * 60)
print('PHASE 6b: Deployment CSV Export')
print('=' * 60)

if not flag_importance:
    # Tree never split on any flag — drop all of them
    unused_flags = list(INCOMPLETE_FLAGS)
    used_flags   = []
elif not unused_flags:
    print('\n  All incomplete flags carry meaningful split weight.')
    print('  No columns dropped — deployment CSVs not generated.')
else:
    print(f'\n  Unused incomplete flags (importance < {FLAG_IMPORTANCE_THRESHOLD:.0%}):')
    for f in unused_flags:
        print(f'    - {f}  (importance={flag_importance.get(f, 0.0):.4f})')
    if used_flags:
        print(f'\n  Retained incomplete flags (importance >= {FLAG_IMPORTANCE_THRESHOLD:.0%}):')
        for f in used_flags:
            print(f'    + {f}  (importance={flag_importance.get(f, 0.0):.4f})')

# ── Save deployment CSVs whenever there are flags to drop ──
if unused_flags:
    cols_to_drop = [f for f in unused_flags if f in r_train.columns]
    dataset_map  = {
        "train":   r_train,
        "s_train": s_train,
        "val":     val_df,
        "test":    test_df,
    }
    print(f'\n  Dropping {len(cols_to_drop)} unused flag column(s): {cols_to_drop}')
    print(f'  Saving deployment CSVs...')
    for name, df in dataset_map.items():
        drop_present = [c for c in cols_to_drop if c in df.columns]
        if not drop_present:
            print(f'    [{name}] No matching columns to drop — skipped.')
            continue
        deployment_df   = df.drop(columns=drop_present)
        deployment_path = DATASET_DIR / f"{name}_deployment.csv"
        deployment_df.to_csv(deployment_path, index=False)
        print(f'    [{name}] Saved → {deployment_path.name}  '
              f'(shape: {df.shape} → {deployment_df.shape})')
    print('\n  ✔ Deployment CSVs written to:', DATASET_DIR)