import itertools
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score, f1_score, recall_score, precision_score, accuracy_score
from C45DecisionTree import C45DecisionTree
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

def fit_calibrator(probs: List[float], y_true: pd.Series):
    """
    Fit isotonic regression calibrator.
    """
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(probs, y_true)
    return calibrator


def apply_calibrator(calibrator, probs: List[float]) -> List[float]:
    """
    Apply fitted calibrator to probabilities.
    """
    return calibrator.transform(probs)

# -----------------------------
# Feature + Domain Definitions
# -----------------------------
FUNA_DB_DOMAIN_MAPPING: Dict[str, str] = {
    "NC": "Number Comparison",
    "DM": "Digit-Dot Matching",
    "NP": "Overall Processing Efficiency",
    "SN": "Symbolic vs. Non-Symbolic Processing Difference",
    "NS": "Number Series",
    "ADD": "Single-Digit Addition",
    "SUB": "Single-Digit Subtraction",
    "CA":  "Multi-Digit Addition and Subtraction",
    "AF": "Overall Arithmetic Fluency",
    "BC": "Basic vs. Complex Arithmetic Contrast",
    "AS": "Addition vs. Subtraction Asymmetry",
    "PF": "Processing-Fluency Integration",
}
FUNA_DB_RAW_FEATURES: List[str] = [
    "NC", "DM", "NS", "ADD", "SUB", "CA",
    # "NC_incomplete", "DM_incomplete", "NS_incomplete",
    # "ADD_incomplete", "SUB_incomplete", "CA_incomplete",
    # "any_incomplete",
]
FUNA_DB_DIAGNOSTIC_FEATURES: List[str] = [
    "NC", "DM", "NS", "ADD", "SUB", "CA",
    "NP", "SN", "AF", "BC", "AS", "PF",
    # "NC_incomplete", "DM_incomplete", "NS_incomplete",
    # "ADD_incomplete", "SUB_incomplete", "CA_incomplete",
    # "any_incomplete",
]

# Features that signal missing/incomplete data.
# The tree may still split on these (they carry real signal),
# but we exclude them from diagnostic z-score severity so that
# "timed out" is never conflated with "performed poorly".
INCOMPLETE_FLAGS = {
    "NC_incomplete", "DM_incomplete", "NS_incomplete",
    "ADD_incomplete", "SUB_incomplete", "CA_incomplete",
    "any_incomplete",
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
           best_params: dict, threshold: float) -> tuple[dict, list[dict]]:

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_f1, cv_fbeta, cv_recall, cv_precision, cv_accuracy = [], [], [], [], []
    fold_records = []

    print(f'\nStarting 5-Fold Cross-Validation [{label}]...')

    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):

        X_trn_fold = X_train.iloc[train_index]
        X_val_fold = X_train.iloc[val_index]
        y_trn_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]

        # -----------------------------
        # 🔥 NEW: split train → (train_sub, calib)
        # -----------------------------
        X_sub, X_calib, y_sub, y_calib = train_test_split(
            X_trn_fold, y_trn_fold,
            test_size=0.2,
            stratify=y_trn_fold,
            random_state=42
        )

        # Train model on sub-train
        tree = C45DecisionTree(**best_params, feature_domain_mapping=FUNA_DB_DOMAIN_MAPPING)
        tree.fit(X_sub, y_sub, raw_features=FUNA_DB_RAW_FEATURES)

        # -----------------------------
        # 🔥 Fit calibrator on CALIB set
        # -----------------------------
        probs_calib_raw = get_probs(tree, X_calib)
        calibrator = fit_calibrator(probs_calib_raw, y_calib)

        # -----------------------------
        # Evaluate on validation fold
        # -----------------------------
        probs_val_raw = get_probs(tree, X_val_fold)
        probs_val = apply_calibrator(calibrator, probs_val_raw)

        preds = probs_to_preds(probs_val, threshold)
        m = compute_metrics(y_val_fold, preds)

        cv_f1.append(m['f1'])
        cv_fbeta.append(m['fbeta'])
        cv_recall.append(m['recall'])
        cv_precision.append(m['precision'])
        cv_accuracy.append(m['accuracy'])

        fold_class_counts = y_trn_fold.value_counts().to_dict()

        fold_records.append({
            'fold': fold,
            'n_train': len(y_trn_fold),
            'n_val': len(y_val_fold),
            'class_0_train': fold_class_counts.get(0, 0),
            'class_1_train': fold_class_counts.get(1, 0),
            'class_0_val': int(y_val_fold.value_counts().get(0, 0)),
            'class_1_val': int(y_val_fold.value_counts().get(1, 0)),
            'tree_depth': tree.get_depth(),
            'tree_leaves': tree.get_leaves_num(),
            **m,
        })

        print(f'  Fold {fold}: F1={m["f1"]:.4f}, F2={m["fbeta"]:.4f}, '
              f'Recall={m["recall"]:.4f}')

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
    print(f'  Recall:    {metrics["mean_recall"]:.4f} ± {metrics["std_recall"]:.4f}')
    print(f'  Precision: {metrics["mean_precision"]:.4f} ± {metrics["std_precision"]:.4f}')
    print(f'  F1-Score:  {metrics["mean_f1"]:.4f} ± {metrics["std_f1"]:.4f}')
    print(f'  F2-Score:  {metrics["mean_fbeta"]:.4f} ± {metrics["std_fbeta"]:.4f}')
    print(f'  Accuracy:  {metrics["mean_accuracy"]:.4f} ± {metrics["std_accuracy"]:.4f}')
    print(f'\n⚠  Note: Metrics evaluated at threshold = {threshold}')

def print_variance_inspection(label: str, fold_records: list[dict]):
    """
    Print per-fold breakdown to diagnose what drives recall variance in TSTR.
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

    print(f'\n  Recall range : {df["recall"].min():.4f} – {df["recall"].max():.4f}')
    print(f'  Recall mean  : {mean_recall:.4f} ± {std_recall:.4f}')

    # Check if class imbalance per fold correlates with recall swings
    corr_c1 = df['class_1_train'].corr(df['recall'])
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
r_train = pd.read_csv('dataset/train.csv')
s_train = pd.read_csv('dataset/s_train.csv')
val_df  = pd.read_csv('dataset/val.csv')
test_df = pd.read_csv('dataset/test.csv')

# -----------------------------
# Prepare Train Sets
# -----------------------------
train_trtr = r_train.copy()                                      # TRTR: real only
train_tstr = pd.concat([r_train, s_train], ignore_index=True)   # TSTR: real + synthetic

print(f'TRTR Train shape (Real 70%):         {train_trtr.shape}')
print(f'TSTR Train shape (Real + Synth):     {train_tstr.shape}')
print(f'Validation shape (Real 15%):         {val_df.shape}')
print(f'Test shape (Real 15%):               {test_df.shape}')

# Synthetic data summary
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
    'conf_fact':          0.25,
    'min_samples_leaf':   3,
    'max_depth':          10,
}
print('\nUsing fixed hyperparameters:')
print(best_params)

# ═══════════════════════════════════════════════════════════
# PHASE 1 — Threshold Sweep on Validation Set
# Train on full TRTR, sweep thresholds, pick best by F2.
# Val set is real-only and held out from all CV folds.
# ═══════════════════════════════════════════════════════════
print('\n' + '═' * 60)
print('PHASE 1: Threshold Sweep (trained on TRTR, evaluated on val)')
print('═' * 60)

THRESHOLDS = np.arange(0.35, 0.76, 0.05).round(2)

sweep_tree = C45DecisionTree(**best_params, feature_domain_mapping=FUNA_DB_DOMAIN_MAPPING)
sweep_tree.fit(X_train_trtr, y_train_trtr, raw_features=FUNA_DB_RAW_FEATURES)
val_probs_raw = get_probs(sweep_tree, X_val)

# Fit calibrator on validation set
train_probs_raw = get_probs(sweep_tree, X_train_trtr)
trtr_calibrator = fit_calibrator(train_probs_raw, y_train_trtr)

# Apply calibration
val_probs = apply_calibrator(trtr_calibrator, val_probs_raw)

sweep_results = []
print(f'\n{"Threshold":>10} {"Recall":>8} {"Precision":>10} {"F1":>8} {"F2":>8} {"Accuracy":>10}')
print('-' * 60)
for thresh in THRESHOLDS:
    preds = probs_to_preds(val_probs, thresh)
    m = compute_metrics(y_val, preds)
    sweep_results.append({'threshold': thresh, **m})
    print(f'{thresh:>10.2f} {m["recall"]:>8.4f} {m["precision"]:>10.4f} '
          f'{m["f1"]:>8.4f} {m["fbeta"]:>8.4f} {m["accuracy"]:>10.4f}')

# Pick best threshold by F2 (recall-weighted)
sweep_df       = pd.DataFrame(sweep_results)
best_row       = sweep_df.loc[sweep_df['fbeta'].idxmax()]
BEST_THRESHOLD = float(best_row['threshold'])

print(f'\n✔ Best threshold by F2: {BEST_THRESHOLD:.2f}')
print(f'  Recall={best_row["recall"]:.4f}, Precision={best_row["precision"]:.4f}, '
      f'F2={best_row["fbeta"]:.4f}, F1={best_row["f1"]:.4f}')

# ═══════════════════════════════════════════════════════════
# PHASE 1b — TSTR-Specific Threshold Sweep on Validation Set
# The threshold from Phase 1 was selected on a TRTR-trained
# model. A TSTR-trained model has a different probability
# distribution (balanced training → lower optimal threshold
# may not apply). This sweep finds the TSTR-optimal threshold
# so we can compare test performance fairly in Phase 6.
# ═══════════════════════════════════════════════════════════
print('\n' + '═' * 60)
print('PHASE 1b: TSTR Threshold Sweep (trained on TSTR, evaluated on val)')
print('═' * 60)

tstr_sweep_tree = C45DecisionTree(**best_params, feature_domain_mapping=FUNA_DB_DOMAIN_MAPPING)
tstr_sweep_tree.fit(X_train_tstr, y_train_tstr, raw_features=FUNA_DB_RAW_FEATURES)
tstr_val_probs_raw = get_probs(tstr_sweep_tree, X_val)

# Fit calibrator for TSTR
tstr_calibrator = fit_calibrator(tstr_val_probs_raw, y_val)

# Apply calibration
tstr_val_probs = apply_calibrator(tstr_calibrator, tstr_val_probs_raw)

tstr_sweep_results = []
print(f'\n{"Threshold":>10} {"Recall":>8} {"Precision":>10} {"F1":>8} {"F2":>8} {"Accuracy":>10}')
print('-' * 60)
for thresh in THRESHOLDS:
    preds = probs_to_preds(tstr_val_probs, thresh)
    m = compute_metrics(y_val, preds)
    tstr_sweep_results.append({'threshold': thresh, **m})
    print(f'{thresh:>10.2f} {m["recall"]:>8.4f} {m["precision"]:>10.4f} '
          f'{m["f1"]:>8.4f} {m["fbeta"]:>8.4f} {m["accuracy"]:>10.4f}')

tstr_sweep_df        = pd.DataFrame(tstr_sweep_results)
tstr_best_row        = tstr_sweep_df.loc[tstr_sweep_df['fbeta'].idxmax()]
TSTR_BEST_THRESHOLD  = float(tstr_best_row['threshold'])

print(f'\n✔ TSTR best threshold by F2: {TSTR_BEST_THRESHOLD:.2f}')
print(f'  Recall={tstr_best_row["recall"]:.4f}, Precision={tstr_best_row["precision"]:.4f}, '
      f'F2={tstr_best_row["fbeta"]:.4f}, F1={tstr_best_row["f1"]:.4f}')

if TSTR_BEST_THRESHOLD != BEST_THRESHOLD:
    print(f'\n  ℹ TSTR threshold ({TSTR_BEST_THRESHOLD:.2f}) differs from TRTR threshold '
          f'({BEST_THRESHOLD:.2f}).')
    print(f'    Phase 6 will re-evaluate test set using the TSTR-optimal threshold.')
else:
    print(f'\n  ✔ TSTR threshold matches TRTR threshold ({BEST_THRESHOLD:.2f}). '
          f'Phase 6 will still report for completeness.')

# ═══════════════════════════════════════════════════════════
# PHASE 2 — TRTR vs TSTR Cross-Validation
# Both conditions use the threshold selected in Phase 1.
# ═══════════════════════════════════════════════════════════
print('\n' + '═' * 60)
print(f'PHASE 2: TRTR vs TSTR CV  (threshold = {BEST_THRESHOLD:.2f})')
print('═' * 60)

cv_metrics_trtr, fold_records_trtr = run_cv(X_train_trtr, y_train_trtr, 'TRTR', best_params, BEST_THRESHOLD)
cv_metrics_tstr, fold_records_tstr = run_cv(X_train_tstr, y_train_tstr, 'TSTR', best_params, TSTR_BEST_THRESHOLD)

print_cv_results('TRTR', cv_metrics_trtr, BEST_THRESHOLD)
print_cv_results('TSTR', cv_metrics_tstr, TSTR_BEST_THRESHOLD)

# ═══════════════════════════════════════════════════════════
# PHASE 3 — Side-by-side Summary
# ═══════════════════════════════════════════════════════════
print('\n' + '═' * 60)
print('PHASE 3: Summary Comparison')
print('═' * 60)

metrics_labels = [
    ('Recall',    'mean_recall',    'std_recall'),
    ('Precision', 'mean_precision', 'std_precision'),
    ('F1-Score',  'mean_f1',        'std_f1'),
    ('F2-Score',  'mean_fbeta',     'std_fbeta'),
    ('Accuracy',  'mean_accuracy',  'std_accuracy'),
]

print(f'\n{"Metric":>12} {"TRTR (mean±std)":>22} {"TSTR (mean±std)":>22} {"Δ (TSTR−TRTR)":>15}')
print('-' * 75)
for name, mean_key, std_key in metrics_labels:
    trtr_val = cv_metrics_trtr[mean_key]
    tstr_val = cv_metrics_tstr[mean_key]
    delta    = tstr_val - trtr_val
    sign     = '+' if delta >= 0 else ''
    print(f'{name:>12}  '
          f'{trtr_val:.4f} ± {cv_metrics_trtr[std_key]:.4f}      '
          f'{tstr_val:.4f} ± {cv_metrics_tstr[std_key]:.4f}      '
          f'{sign}{delta:.4f}')

print(f'\n⚠  Threshold locked at {BEST_THRESHOLD:.2f} (selected by F2 on val set, TRTR model)')

# ═══════════════════════════════════════════════════════════
# PHASE 4 — Fold Variance Inspection
# Diagnose what drives recall swings across folds, especially
# for TSTR where synthetic data distribution may be uneven.
# ═══════════════════════════════════════════════════════════
print('\n' + '═' * 60)
print('PHASE 4: Fold Variance Inspection')
print('═' * 60)

print_variance_inspection('TRTR', fold_records_trtr)
print_variance_inspection('TSTR', fold_records_tstr)

# ═══════════════════════════════════════════════════════════
# PHASE 4b — TSTR Fold 2 Outlier Investigation
# Fold 2 had noticeably lower recall than the other TSTR folds.
# This phase identifies which synthetic samples were excluded
# from Fold 2's training set (i.e., fell into its val split)
# to find gaps in synthetic data coverage.
# ═══════════════════════════════════════════════════════════
print('\n' + '═' * 60)
print('PHASE 4b: TSTR Fold 2 Outlier Investigation')
print('═' * 60)

# Identify the outlier fold: lowest recall among TSTR folds
tstr_fold_df    = pd.DataFrame(fold_records_tstr)
outlier_fold_no = int(tstr_fold_df.loc[tstr_fold_df['recall'].idxmin(), 'fold'])
outlier_recall  = tstr_fold_df.loc[tstr_fold_df['recall'].idxmin(), 'recall']
mean_tstr_recall = tstr_fold_df['recall'].mean()

print(f'\n  Outlier fold: Fold {outlier_fold_no} '
      f'(recall={outlier_recall:.4f} vs mean={mean_tstr_recall:.4f})')

# Re-split TSTR to isolate the outlier fold's val indices
n_splits_inv = 5
skf_inv = StratifiedKFold(n_splits=n_splits_inv, shuffle=True, random_state=42)
for fold_i, (train_idx, val_idx) in enumerate(skf_inv.split(X_train_tstr, y_train_tstr), 1):
    if fold_i == outlier_fold_no:
        outlier_val_idx = val_idx
        break

# Recover full TSTR rows for val split of outlier fold
tstr_val_fold_df = train_tstr.iloc[outlier_val_idx].copy()
tstr_val_fold_df['_source'] = np.where(
    tstr_val_fold_df.index < len(r_train), 'real', 'synthetic'
)

synth_in_val = tstr_val_fold_df[tstr_val_fold_df['_source'] == 'synthetic']
real_in_val  = tstr_val_fold_df[tstr_val_fold_df['_source'] == 'real']

print(f'\n  Val split composition for Fold {outlier_fold_no}:')
print(f'    Real samples:      {len(real_in_val)}  '
      f'(class 0: {int((real_in_val["Label"]==0).sum())}  '
      f'class 1: {int((real_in_val["Label"]==1).sum())})')
print(f'    Synthetic samples: {len(synth_in_val)}  '
      f'(class 0: {int((synth_in_val["Label"]==0).sum())}  '
      f'class 1: {int((synth_in_val["Label"]==1).sum())})')

# Show feature stats of synthetic positives excluded from training in this fold
synth_pos_excluded = synth_in_val[synth_in_val['Label'] == 1]
if len(synth_pos_excluded) > 0:
    score_cols = ["NC", "DM", "NS", "ADD", "SUB", "CA"]
    print(f'\n  Synthetic at-risk samples excluded from Fold {outlier_fold_no} training '
          f'(n={len(synth_pos_excluded)}):')
    print(f'  {"Feature":>6} {"Mean":>8} {"Std":>8} {"Min":>8} {"Max":>8}')
    print('  ' + '-' * 44)
    for col in score_cols:
        if col in synth_pos_excluded.columns:
            print(f'  {col:>6} {synth_pos_excluded[col].mean():>8.3f} '
                  f'{synth_pos_excluded[col].std():>8.3f} '
                  f'{synth_pos_excluded[col].min():>8.3f} '
                  f'{synth_pos_excluded[col].max():>8.3f}')

    # Compare to synthetic positives that stayed IN training for this fold
    synth_in_train = train_tstr.iloc[
        [i for i in range(len(train_tstr)) if i not in outlier_val_idx]
    ]
    synth_pos_trained = synth_in_train[
        (synth_in_train.index >= len(r_train)) & (synth_in_train['Label'] == 1)
    ]
    if len(synth_pos_trained) > 0:
        print(f'\n  Synthetic at-risk samples kept in Fold {outlier_fold_no} training '
              f'(n={len(synth_pos_trained)}):')
        print(f'  {"Feature":>6} {"Mean":>8} {"Std":>8} {"Min":>8} {"Max":>8}')
        print('  ' + '-' * 44)
        for col in score_cols:
            if col in synth_pos_trained.columns:
                print(f'  {col:>6} {synth_pos_trained[col].mean():>8.3f} '
                      f'{synth_pos_trained[col].std():>8.3f} '
                      f'{synth_pos_trained[col].min():>8.3f} '
                      f'{synth_pos_trained[col].max():>8.3f}')
        print(f'\n  ℹ Compare the two tables above: if the excluded group has '
              f'systematically\n    different feature ranges, those subgroups are '
              f'underrepresented in your\n    synthetic data.')
else:
    print(f'\n  No synthetic at-risk samples in Fold {outlier_fold_no} val split — '
          f'the recall drop is driven purely\n  by real sample difficulty, '
          f'not synthetic coverage gaps.')

# ═══════════════════════════════════════════════════════════
# PHASE 5 — Final Test Set Evaluation
# Train on full TRTR / TSTR, evaluate on held-out test set.
# Uses the threshold locked in Phase 1.
# ═══════════════════════════════════════════════════════════
print('\n' + '═' * 60)
print(f'PHASE 5: Test Set Evaluation  (threshold = {BEST_THRESHOLD:.2f})')
print('═' * 60)

results = {}
for label, X_tr, y_tr in [('TRTR', X_train_trtr, y_train_trtr),
                            ('TSTR', X_train_tstr, y_train_tstr)]:
    final_tree = C45DecisionTree(**best_params, feature_domain_mapping=FUNA_DB_DOMAIN_MAPPING)
    final_tree.fit(X_tr, y_tr, raw_features=FUNA_DB_RAW_FEATURES)

    test_probs_raw = get_probs(final_tree, X_test)

    # Use the correct calibrator
    if label == 'TRTR':
        test_probs = apply_calibrator(trtr_calibrator, test_probs_raw)
    else:
        test_probs = apply_calibrator(tstr_calibrator, test_probs_raw)
    test_preds = probs_to_preds(test_probs, BEST_THRESHOLD)
    m = compute_metrics(y_test, test_preds)
    results[label] = m

    print(f'\n  [{label}] Test Set Results:')
    print(f'    Recall:    {m["recall"]:.4f}')
    print(f'    Precision: {m["precision"]:.4f}')
    print(f'    F1-Score:  {m["f1"]:.4f}')
    print(f'    F2-Score:  {m["fbeta"]:.4f}')
    print(f'    Accuracy:  {m["accuracy"]:.4f}')

# Side-by-side test comparison
# Note: test results use plain keys (recall, f1, ...) not mean_ prefixed ones
test_metrics_labels = [
    ('Recall',    'recall'),
    ('Precision', 'precision'),
    ('F1-Score',  'f1'),
    ('F2-Score',  'fbeta'),
    ('Accuracy',  'accuracy'),
]
print(f'\n{"Metric":>12} {"TRTR (test)":>14} {"TSTR (test)":>14} {"Δ (TSTR−TRTR)":>15}')
print('-' * 58)
for name, key in test_metrics_labels:
    trtr_val = results['TRTR'][key]
    tstr_val = results['TSTR'][key]
    delta    = tstr_val - trtr_val
    sign     = '+' if delta >= 0 else ''
    print(f'{name:>12}  {trtr_val:>12.4f}  {tstr_val:>12.4f}  {sign}{delta:.4f}')

# CV vs Test consistency check
print(f'\n--- CV vs Test Consistency Check ---')
# FIX 3: test_m uses plain keys ('recall', 'fbeta'), not 'mean_recall'/'mean_fbeta'
# — use the same plain keys that compute_metrics() returns
for label, cv_m, test_m in [('TRTR', cv_metrics_trtr, results['TRTR']),
                              ('TSTR', cv_metrics_tstr, results['TSTR'])]:
    recall_drift = test_m['recall'] - cv_m['mean_recall']
    f2_drift     = test_m['fbeta']  - cv_m['mean_fbeta']
    sign_r = '+' if recall_drift >= 0 else ''
    sign_f = '+' if f2_drift     >= 0 else ''
    print(f'  [{label}]  Recall drift (test − CV mean): {sign_r}{recall_drift:.4f}  |  '
          f'F2 drift: {sign_f}{f2_drift:.4f}')
    if abs(recall_drift) > 0.10:
        print(f'    ⚠ Recall drift > 0.10 — possible overfitting or distribution mismatch.')

# ═══════════════════════════════════════════════════════════
# PHASE 6 — TSTR Re-evaluation with TSTR-Optimal Threshold
# Phase 5 evaluated TSTR at the TRTR threshold (BEST_THRESHOLD).
# A balanced-trained model may need a higher threshold to avoid
# over-flagging on the real-world (imbalanced) test distribution.
# This phase re-evaluates the TSTR model at TSTR_BEST_THRESHOLD
# and shows the precision/recall trade-off across all thresholds.
# ═══════════════════════════════════════════════════════════
print('\n' + '═' * 60)
print('PHASE 6: TSTR Re-evaluation at TSTR-Optimal Threshold')
print('═' * 60)

# Re-use the TSTR final tree trained in Phase 5
# (already trained on full TSTR; we just need test probs again)
tstr_final_tree = C45DecisionTree(**best_params, feature_domain_mapping=FUNA_DB_DOMAIN_MAPPING)
tstr_final_tree.fit(X_train_tstr, y_train_tstr, raw_features=FUNA_DB_RAW_FEATURES)
tstr_test_probs_raw = get_probs(tstr_final_tree, X_test)
tstr_test_probs = apply_calibrator(tstr_calibrator, tstr_test_probs_raw)

# Full threshold sweep on test set for TSTR (diagnostic only —
# threshold was selected on val, not test)
print(f'\n  TSTR test-set performance across thresholds:')
print(f'  {"Threshold":>10} {"Recall":>8} {"Precision":>10} {"F1":>8} {"F2":>8} {"Accuracy":>10}')
print('  ' + '-' * 60)
tstr_test_sweep = []
for thresh in THRESHOLDS:
    preds = probs_to_preds(tstr_test_probs, thresh)
    m = compute_metrics(y_test, preds)
    tstr_test_sweep.append({'threshold': thresh, **m})
    marker = ' ◄ TRTR thresh' if thresh == BEST_THRESHOLD else \
             ' ◄ TSTR thresh' if thresh == TSTR_BEST_THRESHOLD else ''
    # Avoid double-marking if thresholds are equal
    if BEST_THRESHOLD == TSTR_BEST_THRESHOLD and thresh == BEST_THRESHOLD:
        marker = ' ◄ both thresholds'
    print(f'  {thresh:>10.2f} {m["recall"]:>8.4f} {m["precision"]:>10.4f} '
          f'{m["f1"]:>8.4f} {m["fbeta"]:>8.4f} {m["accuracy"]:>10.4f}{marker}')

# Side-by-side: TRTR @ TRTR-thresh  vs  TSTR @ TRTR-thresh  vs  TSTR @ TSTR-thresh
tstr_at_trtr_thresh = results['TSTR']   # already computed in Phase 5
tstr_at_tstr_thresh = compute_metrics(
    y_test, probs_to_preds(tstr_test_probs, TSTR_BEST_THRESHOLD)
)
trtr_at_trtr_thresh = results['TRTR']

print(f'\n  Summary — test set, three configurations:')
print(f'  {"Metric":>12} {"TRTR@{:.2f}".format(BEST_THRESHOLD):>16} '
      f'{"TSTR@{:.2f}".format(BEST_THRESHOLD):>16} '
      f'{"TSTR@{:.2f}".format(TSTR_BEST_THRESHOLD):>16}')
print('  ' + '-' * 64)
for name, key in [('Recall', 'recall'), ('Precision', 'precision'),
                  ('F1-Score', 'f1'), ('F2-Score', 'fbeta'), ('Accuracy', 'accuracy')]:
    v_trtr = trtr_at_trtr_thresh[key]
    v_tstr_trtr = tstr_at_trtr_thresh[key]
    v_tstr_tstr = tstr_at_tstr_thresh[key]
    print(f'  {name:>12}  {v_trtr:>14.4f}  {v_tstr_trtr:>14.4f}  {v_tstr_tstr:>14.4f}')

# Interpretation hint
prec_gain = tstr_at_tstr_thresh['precision'] - tstr_at_trtr_thresh['precision']
rec_loss  = tstr_at_trtr_thresh['recall']    - tstr_at_tstr_thresh['recall']
print(f'\n  Raising TSTR threshold from {BEST_THRESHOLD:.2f} → {TSTR_BEST_THRESHOLD:.2f}:')
print(f'    Precision change : {prec_gain:+.4f}')
print(f'    Recall change    : {-rec_loss:+.4f}')
if prec_gain > 0 and rec_loss <= 0.05:
    print(f'    ✔ Precision improves with negligible recall cost — '
          f'TSTR_BEST_THRESHOLD is the better operating point.')
elif prec_gain > 0 and rec_loss > 0.05:
    print(f'    ⚠ Precision improves but recall drops by >{rec_loss:.2f}. '
          f'Weigh the clinical trade-off before adopting the higher threshold.')
else:
    print(f'    ℹ No precision gain from raising threshold — '
          f'keep TRTR threshold ({BEST_THRESHOLD:.2f}) for TSTR model too.')
    
print(f"Global Importance: {tstr_final_tree.get_feature_importance()}")