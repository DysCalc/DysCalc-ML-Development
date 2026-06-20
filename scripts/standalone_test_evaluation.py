import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import fbeta_score, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, mean_squared_error
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.C45DecisionTree import C45DecisionTree

# --- Constants & Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATASET_DIR = ROOT_DIR / "datasets" / "processed"

DATASETS_FILENAMES = {
    'train': PROCESSED_DATASET_DIR / 'train.csv',
    's_train': PROCESSED_DATASET_DIR / 's_train.csv',
    'val': PROCESSED_DATASET_DIR / 'val.csv',
    'test': PROCESSED_DATASET_DIR / 'test.csv'
}

NUMBER_PROCESSING = ['NC', 'DM']
ARITHMETIC_FLUENCY = ['NS', 'ADD', 'SUB', 'CA']
FEATURE_COLUMNS = NUMBER_PROCESSING + ARITHMETIC_FLUENCY
INCOMPLETE_FLAGS = [f"{col}_incomplete" for col in FEATURE_COLUMNS]
DERIVED_FEATURES = ['NP', 'SN', 'AF', 'BC', 'AS', 'PF']
FUNA_DB_RAW_FEATURES = FEATURE_COLUMNS + list(INCOMPLETE_FLAGS)
FUNA_DB_DIAGNOSTIC_FEATURES = FUNA_DB_RAW_FEATURES + DERIVED_FEATURES

FUNA_DB_DOMAIN_MAPPING = {
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

# --- Params from user ---
TRTR_THRESHOLDED_PARAMS = {
    "conf_fact": 0.45,
    "min_samples_leaf": 10,
    "max_depth": 15,
}
BEST_THRESHOLD_TRTR = 0.35

TRSTR_THRESHOLDED_PARAMS = {
    "conf_fact": 0.1,
    "min_samples_leaf": 15,
    "max_depth": 15,
}
BEST_THRESHOLD_TRSTR = 0.40

TRTR_NO_THRESHOLD_PARAMS = {
    "conf_fact": 0.50,
    "min_samples_leaf": 10,
    "max_depth": 15,
}

TRSTR_NO_THRESHOLD_PARAMS = {
    "conf_fact": 0.1,
    "min_samples_leaf": 15,
    "max_depth": 15,
}

# --- Helpers ---
def split_xy(df: pd.DataFrame, label_col: str = 'Label'):
    X = df[FUNA_DB_DIAGNOSTIC_FEATURES]
    y = df[label_col]
    return X, y

def get_preds(tree, X: pd.DataFrame):
    return [int(pred) for pred in tree.predict(X)]

def get_probs(tree, X: pd.DataFrame):
    return tree.predict_proba(X, positive_class=1)

def probs_to_preds(probs, threshold):
    return (np.array(probs) >= threshold).astype(int)

def wilson_ci(y_true, y_pred, metric='recall', alpha=0.05):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if metric == 'recall':
        n = np.sum(y_true == 1)
        count = np.sum((y_true == 1) & (y_pred == 1))
    elif metric == 'precision':
        n = np.sum(y_pred == 1)
        count = np.sum((y_true == 1) & (y_pred == 1))
    else:
        return (0.0, 0.0)

    if n == 0:
        return (0.0, 0.0)

    p = count / n
    z = norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    
    return center - spread, center + spread

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred),
        "fbeta": fbeta_score(y_true, y_pred, beta=2),
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "rmse": np.sqrt(mean_squared_error(y_true, y_prob))
    }

def get_bootstrap_ci(y_true, y_pred, y_prob, metric_name, n_bootstraps=1000, alpha=0.05):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    n = len(y_true)
    stats = []

    for _ in range(n_bootstraps):
        idx = resample(np.arange(n))
        metrics = compute_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        stats.append(metrics[metric_name])
        
    lower = np.percentile(stats, 100 * (alpha / 2))
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return lower, upper

# --- Load Data ---
df_train = pd.read_csv(DATASETS_FILENAMES['train'])
df_s_train = pd.read_csv(DATASETS_FILENAMES['s_train'])
df_test = pd.read_csv(DATASETS_FILENAMES['test'])

# Prepare splits
X_train_trtr, y_train_trtr = split_xy(df_train)

df_trstr_train = pd.concat([df_train, df_s_train], ignore_index=True)
X_train_trstr, y_train_trstr = split_xy(df_trstr_train)

X_test, y_test = split_xy(df_test)

# --- Define Tests ---
TEST_RUNS = {
    'no_threshold': {
        'TRTR': {'params': TRTR_NO_THRESHOLD_PARAMS, 'threshold': None, 'X_train': X_train_trtr, 'y_train': y_train_trtr},
        'TRSTR': {'params': TRSTR_NO_THRESHOLD_PARAMS, 'threshold': None, 'X_train': X_train_trstr, 'y_train': y_train_trstr},
    },
    'thresholded': {
        'TRTR': {'params': TRTR_THRESHOLDED_PARAMS, 'threshold': BEST_THRESHOLD_TRTR, 'X_train': X_train_trtr, 'y_train': y_train_trtr},
        'TRSTR': {'params': TRSTR_THRESHOLDED_PARAMS, 'threshold': BEST_THRESHOLD_TRSTR, 'X_train': X_train_trstr, 'y_train': y_train_trstr},
    }
}

# --- Evaluate ---
for mode, run_map in TEST_RUNS.items():
    print(f'\n--- {mode.upper()} TEST EVALUATION WITH CONFIDENCE INTERVALS ---')
    
    for label in ['TRTR', 'TRSTR']:
        cfg = run_map[label]
        print(f'\n  [{label} | {mode}] Locked params: {cfg["params"]}')
        if cfg['threshold'] is not None:
            print(f'  [{label} | {mode}] Locked threshold: {cfg["threshold"]:.2f}')

        final_tree = C45DecisionTree(**cfg['params'], feature_domain_mapping=FUNA_DB_DOMAIN_MAPPING)
        final_tree.fit(cfg['X_train'], cfg['y_train'], raw_features=FUNA_DB_RAW_FEATURES)

        test_probs = get_probs(final_tree, X_test)
        if cfg['threshold'] is None:
            test_preds = get_preds(final_tree, X_test)
        else:
            test_preds = probs_to_preds(test_probs, cfg['threshold'])

        m = compute_metrics(y_test, test_preds, test_probs)

        # Confidence intervals
        ci_recall = wilson_ci(y_test, test_preds, 'recall')
        ci_precision = wilson_ci(y_test, test_preds, 'precision')
        print("    Calculating Bootstrap CI for F1...", flush=True)
        ci_f1 = get_bootstrap_ci(y_test, test_preds, test_probs, 'f1')
        print("    Calculating Bootstrap CI for F2...", flush=True)
        ci_f2 = get_bootstrap_ci(y_test, test_preds, test_probs, 'fbeta')
        print("    Calculating Bootstrap CI for Accuracy...", flush=True)
        ci_acc = get_bootstrap_ci(y_test, test_preds, test_probs, 'accuracy')
        print("    Calculating Bootstrap CI for AUC...", flush=True)
        ci_auc = get_bootstrap_ci(y_test, test_preds, test_probs, 'auc')
        print("    Calculating Bootstrap CI for RMSE...", flush=True)
        ci_rmse = get_bootstrap_ci(y_test, test_preds, test_probs, 'rmse')

        print(f'\n  [{label} | {mode}] Test Set Results:', flush=True)
        print(f'    Recall:    {m["recall"]:.4f}  [95% CI: {ci_recall[0]:.4f}, {ci_recall[1]:.4f}] (Wilson)', flush=True)
        print(f'    Precision: {m["precision"]:.4f}  [95% CI: {ci_precision[0]:.4f}, {ci_precision[1]:.4f}] (Wilson)', flush=True)
        print(f'    F1-Score:  {m["f1"]:.4f}  [95% CI: {ci_f1[0]:.4f}, {ci_f1[1]:.4f}] (Bootstrap)', flush=True)
        print(f'    F2-Score:  {m["fbeta"]:.4f}  [95% CI: {ci_f2[0]:.4f}, {ci_f2[1]:.4f}] (Bootstrap)', flush=True)
        print(f'    Accuracy:  {m["accuracy"]:.4f}  [95% CI: {ci_acc[0]:.4f}, {ci_acc[1]:.4f}] (Bootstrap)', flush=True)
        print(f'    AUC-ROC:   {m["auc"]:.4f}  [95% CI: {ci_auc[0]:.4f}, {ci_auc[1]:.4f}] (Bootstrap)', flush=True)
        print(f'    RMSE:      {m["rmse"]:.4f}  [95% CI: {ci_rmse[0]:.4f}, {ci_rmse[1]:.4f}] (Bootstrap)', flush=True)
