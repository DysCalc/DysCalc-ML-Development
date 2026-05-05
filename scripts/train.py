"""
train.py — Deployment Training Script
======================================
Trains the final C4.5 TSTR model on the full real + synthetic training set,
fits an isotonic calibrator, and saves the complete model package to disk.

Usage
-----
    python train.py --threshold 0.35                   # required
    python train.py --threshold 0.35 --out models/v1.pkl
    python train.py --threshold 0.35 --no-synth        # TRTR mode (real data only)

Output
------
    <out>.pkl     — pickled model package via tree.save_model()
                    contains: model, optimal_threshold, calibrator

Dependencies
------------
    pandas, numpy, scikit-learn, scipy
    C45DecisionTree.py, Dataclasses.py  (must be on PYTHONPATH or same directory)
"""

import argparse
import logging
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score, accuracy_score

from src.C45DecisionTree import C45DecisionTree

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Feature definitions  (mirrors tstr_vs_trtr.py)
# ─────────────────────────────────────────────
DOMAIN_MAPPING = {
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

RAW_FEATURES = [
    "NC", "DM", "NS", "ADD", "SUB", "CA",
]

DIAGNOSTIC_FEATURES = [
    "NC", "DM", "NS", "ADD", "SUB", "CA",
    "NP", "SN", "AF", "BC", "AS", "PF",
]

LABEL_COL = "Label"

# ─────────────────────────────────────────────
# Hyperparameters  (fixed from evaluation)
# ─────────────────────────────────────────────
BEST_PARAMS = {
    "conf_fact":        0.25,
    "min_samples_leaf": 3,
    "max_depth":        10,
}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_csv(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info(f"Loaded {label}: {df.shape}  |  class dist: {df[LABEL_COL].value_counts().to_dict()}")
    return df


def split_xy(df: pd.DataFrame):
    return df[DIAGNOSTIC_FEATURES], df[LABEL_COL]


def get_probs(tree: C45DecisionTree, X: pd.DataFrame) -> list[float]:
    """
    Extract P(at-risk) from diagnostic outputs.
    Zeroes out incomplete flags from task_importance post-prediction
    (does not affect tree split decisions).
    """
    diagnostics = tree.predict_with_diagnostics(X)
    probs = []
    for d in diagnostics:
        total = sum(d.task_importance_scores.values())
        if total > 0:
            for f in d.task_importance_scores:
                d.task_importance_scores[f] /= total

        p = d.confidence if int(d.predicted_class) == 1 else 1 - d.confidence
        probs.append(p)
    return probs


def fit_calibrator(probs: list[float], y: pd.Series) -> IsotonicRegression:
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(probs, y)
    return cal


def calibrate(calibrator: IsotonicRegression, probs: list[float]) -> list[float]:
    return calibrator.transform(probs)


def compute_metrics(y_true, probs: list[float], threshold: float) -> dict:
    preds = [1 if p >= threshold else 0 for p in probs]
    return {
        "recall":    recall_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "f1":        f1_score(y_true, preds, zero_division=0),
        "f2":        fbeta_score(y_true, preds, beta=2, zero_division=0),
        "accuracy":  accuracy_score(y_true, preds),
    }


def print_metrics(label: str, m: dict, threshold: float):
    log.info(f"── {label} ──────────────────────────────")
    log.info(f"  Threshold : {threshold:.2f}")
    log.info(f"  Recall    : {m['recall']:.4f}")
    log.info(f"  Precision : {m['precision']:.4f}")
    log.info(f"  F1        : {m['f1']:.4f}")
    log.info(f"  F2        : {m['f2']:.4f}")
    log.info(f"  Accuracy  : {m['accuracy']:.4f}")


def demonstrate_model(model_path: str) -> None:
    """Loads the saved model structure from disk, evaluates on test data, and outputs 10 diagnostics."""
    log.info("\n" + "=" * 60)
    log.info("Demonstrating Loaded Model on Unseen Test Set")
    log.info("=" * 60)

    # 1. Load the model from disk
    loaded_tree, optimal_threshold, calibrator = C45DecisionTree.load_model(model_path)

    # 2. Load unseen test data
    test_df = load_csv("datasets/processed/test_deployment.csv", "test (unseen)")
    X_test, y_test = split_xy(test_df)

    # 3. Generate raw predictions & compute probabilities
    diagnostics = loaded_tree.predict_with_diagnostics(X_test)
    raw_probs = get_probs(loaded_tree, X_test)

    # 4. Calibrate probabilities 
    if calibrator is not None:
        final_probs = calibrate(calibrator, raw_probs)
    else:
        final_probs = raw_probs

    test_metrics = compute_metrics(y_test, final_probs, optimal_threshold)
    print_metrics(f"Test metrics at locked threshold={optimal_threshold:.2f}", test_metrics, optimal_threshold)

    # 5. Output Sample Diagnostics
    log.info(f"\n[Sample Diagnostics for First 10 Tests]")
    for i, diag in enumerate(diagnostics[:10]):
        prob = final_probs[i]
        pred_class = 1 if prob >= optimal_threshold else 0
        pred_label = "At-Risk (1)" if pred_class == 1 else "Typical (0)"

        # String formatting for the task importance dictionary
        task_imp = ", ".join([f"{k}: {v:.2f}" for k, v in diag.task_importance_scores.items() if v > 0])

        log.info(f"\n  Test Case #{i+1}:")
        log.info(f"    Raw Confidence   : {diag.confidence:.4f} (Class {diag.predicted_class})")
        log.info(f"    Final Cal. Prob. : {prob:.4f}")
        log.info(f"    Final Prediction : {pred_label}")
        log.info(f"    Decision Path    : {diag.decision_path_readable}")
        log.info(f"    Domain Severity  : {diag.domain_severity_scores}")
        log.info(f"    Task Importance  : {task_imp if task_imp else 'N/A'}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def train(out_path: str, use_synth: bool, threshold: float) -> None:
    log.info("=" * 60)
    log.info("FunaDB C4.5 Deployment Training")
    log.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────
    log.info("\n[1/4] Loading datasets...")
    r_train = load_csv("datasets/processed/train_deployment.csv", "real train")
    val_df  = load_csv("datasets/processed/val_deployment.csv",   "validation")

    if use_synth:
        s_train  = load_csv("datasets/processed/s_train_deployment.csv", "synthetic train")
        train_df = pd.concat([r_train, s_train], ignore_index=True)
        mode     = "TSTR"
    else:
        train_df = r_train.copy()
        mode     = "TRTR"

    log.info(f"Mode         : {mode}")
    log.info(f"Train shape  : {train_df.shape}")
    log.info(f"Val shape    : {val_df.shape}")

    X_train, y_train = split_xy(train_df)
    X_val,   y_val   = split_xy(val_df)

    # ── 2. Train tree ─────────────────────────────────────────
    log.info("\n[2/4] Training C4.5 decision tree...")
    tree = C45DecisionTree(**BEST_PARAMS, feature_domain_mapping=DOMAIN_MAPPING)
    tree.fit(X_train, y_train, raw_features=RAW_FEATURES)
    log.info(f"Tree depth  : {tree.get_depth()}")
    log.info(f"Tree leaves : {tree.get_leaves_num()}")

    # ── 3. Fit calibrator on training set ─────────────────────
    # Calibrator is fitted on training probabilities so the validation
    # set is kept clean for evaluation only.
    log.info("\n[3/4] Fitting isotonic calibrator on training set...")
    train_probs_raw = get_probs(tree, X_train)
    calibrator      = fit_calibrator(train_probs_raw, y_train)

    # Evaluate on val at the provided threshold (informational only)
    val_probs_cal = calibrate(calibrator, get_probs(tree, X_val))
    val_metrics   = compute_metrics(y_val, val_probs_cal, threshold)
    print_metrics(f"Validation metrics at threshold={threshold:.2f}", val_metrics, threshold)

    # ── 4. Save model ─────────────────────────────────────────
    log.info("\n[4/4] Saving model...")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tree.save_model(str(out), optimal_threshold=threshold, calibrator=calibrator)
    log.info(f"Model saved → {out.resolve()}")
    log.info("\nDone.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save the FunaDB C4.5 deployment model.")
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Classification threshold for P(at-risk) (e.g. 0.35).",
    )
    parser.add_argument(
        "--out",
        default="models/funa_c45.pkl",
        help="Output path for the saved model package (default: models/funa_c45.pkl)",
    )
    parser.add_argument(
        "--no-synth",
        action="store_true",
        help="Train on real data only (TRTR mode). Default is TSTR (real + synthetic).",
    )
    args = parser.parse_args()

    if not (0.0 < args.threshold < 1.0):
        log.error("--threshold must be between 0 and 1 (exclusive).")
        sys.exit(1)

    try:
        train(out_path=args.out, use_synth=not args.no_synth, threshold=args.threshold)
        demonstrate_model(model_path=args.out)
    except FileNotFoundError as e:
        log.error(f"Dataset not found: {e}")
        sys.exit(1)
    except Exception as e:
        log.exception(f"Training failed: {e}")
        sys.exit(1)