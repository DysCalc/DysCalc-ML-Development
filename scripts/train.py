"""
train.py — Deployment Training Script
======================================
Trains the final C4.5 TSTR model on the full real + synthetic training set,
and saves the complete model package to disk.

Usage
-----
    python train.py                                    # TSTR mode with thresholding
    python train.py --out models/v1.pkl
    python train.py --no-synth                         # TRTR mode (real data only)
    python train.py --no-threshold                     # Native C4.5 prediction mode

Output
------
    <out>.pkl     — pickled model package via tree.save_model()
                    contains: model, optimal_threshold

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
from graphviz import Digraph
from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score, accuracy_score

from src.C45DecisionTree import C45DecisionTree

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
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

# Derived features are retained in X for diagnostics, but are intentionally
# excluded from tree splits because they are deterministic transformations of
# raw task scores and would create redundant split candidates.
DIAGNOSTIC_FEATURES = [
    "NC", "DM", "NS", "ADD", "SUB", "CA",
    "NP", "SN", "AF", "BC", "AS", "PF",
]

LABEL_COL = "Label"

# ─────────────────────────────────────────────
# Hyperparameters  (fixed from evaluation)
# ─────────────────────────────────────────────
TRTR_THRESHOLDED_PARAMS = {
    "conf_fact":        0.50,
    "min_samples_leaf": 10,
    "max_depth":        6,
    "threshold":        0.35,
}

TSTR_THRESHOLDED_PARAMS = {
    "conf_fact":        0.4,
    "min_samples_leaf": 11,
    "max_depth":        7,
    "threshold":        0.40,
}

TRTR_NO_THRESHOLD_PARAMS = {
    "conf_fact":        0.50,
    "min_samples_leaf": 10,
    "max_depth":        6,
}

TSTR_NO_THRESHOLD_PARAMS = {
    "conf_fact":        0.25,
    "min_samples_leaf": 11,
    "max_depth":        7,
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


def compute_metrics(y_true, preds) -> dict:
    return {
        "recall":    recall_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "f1":        f1_score(y_true, preds, zero_division=0),
        "f2":        fbeta_score(y_true, preds, beta=2, zero_division=0),
        "accuracy":  accuracy_score(y_true, preds),
    }


def predict_for_evaluation(tree: C45DecisionTree, X: pd.DataFrame, threshold: float | None):
    if threshold is None:
        preds = tree.predict(X).astype(int)
        probs = get_probs(tree, X)
        return preds, probs

    probs = get_probs(tree, X)
    preds = [1 if p >= threshold else 0 for p in probs]
    return preds, probs


def print_metrics(label: str, m: dict, threshold: float | None):
    log.info(f"── {label} ──────────────────────────────")
    if threshold is None:
        log.info("  Prediction: native C4.5 class output")
    else:
        log.info(f"  Threshold : {threshold:.2f}")
    log.info(f"  Recall    : {m['recall']:.4f}")
    log.info(f"  Precision : {m['precision']:.4f}")
    log.info(f"  F1        : {m['f1']:.4f}")
    log.info(f"  F2        : {m['f2']:.4f}")
    log.info(f"  Accuracy  : {m['accuracy']:.4f}")

def export_tree_svg(tree_model: C45DecisionTree, base_filename: str) -> None:
    """Exports the trained tree as an SVG visualization using Graphviz."""
    dot = Digraph(comment='C4.5 Decision Tree')
    dot.attr(dpi='300', nodesep='0.8', ranksep='1.2')
    dot.attr('node', shape='box', style='rounded,filled', fontname='helvetica', fontsize='14', fillcolor='#f8f9fa', margin='0.3,0.15')
    dot.attr('edge', fontname='helvetica', fontsize='12', penwidth='1.2')

    def add_nodes_edges(node, dot, parent_id=None, edge_label=""):
        if node is None:
            return
            
        node_id = str(id(node))
        
        if node.type == "leaf":
            class_label = "At-Risk" if str(node.label) == "1" else "Typical"
            fillcolor = '#ffcccb' if class_label == 'At-Risk' else '#d4edda'
            
            count_0 = node.distribution.get(0, node.distribution.get('0', 0))
            count_1 = node.distribution.get(1, node.distribution.get('1', 0))
            
            label = f"Class: {class_label}\\n"
            label += f"Samples: {node.samples}\\n"
            label += f"Dist: [0: {count_0}, 1: {count_1}]"
            dot.node(node_id, label, fillcolor=fillcolor, shape='ellipse')
        else:
            label = f"{node.feature} <= {node.threshold:.2f}\\n"
            label += f"Samples: {node.samples}\\n"
            label += f"Gain Ratio: {node.gain_ratio:.4f}"
            dot.node(node_id, label)

        if parent_id is not None:
            dot.edge(parent_id, node_id, label=edge_label)

        if node.type == "internal":
            add_nodes_edges(node.left, dot, node_id, "True")
            add_nodes_edges(node.right, dot, node_id, "False")

    if tree_model.tree is not None:
        add_nodes_edges(tree_model.tree, dot)
        dot.render(base_filename, format='svg', cleanup=True)


def demonstrate_model(model_path: str, use_thresholding: bool) -> None:
    """Loads the saved model structure from disk, evaluates on test data, and outputs 10 diagnostics."""
    log.info("\n" + "=" * 60)
    log.info("Demonstrating Loaded Model on Unseen Test Set")
    log.info("=" * 60)

    # 1. Load the model from disk
    loaded_tree, optimal_threshold, conf_fact, min_samples_leaf, max_depth, epsilon = C45DecisionTree.load_model(model_path)

    # 2. Load unseen test data
    test_df = load_csv("datasets/processed/test_deployment.csv", "test (unseen)")
    X_test, y_test = split_xy(test_df)

    # 3. Generate predictions and raw tree probabilities
    diagnostics = loaded_tree.predict_with_diagnostics(X_test)
    final_probs = get_probs(loaded_tree, X_test)

    threshold = optimal_threshold if use_thresholding else None
    test_preds, final_probs = predict_for_evaluation(loaded_tree, X_test, threshold)
    test_metrics = compute_metrics(y_test, test_preds)

    if use_thresholding:
        print_metrics(f"Test metrics at locked threshold={optimal_threshold:.2f}", test_metrics, threshold)
    else:
        print_metrics("Test metrics with native C4.5 predictions", test_metrics, threshold)

    # 5. Output Sample Diagnostics
    log.info(f"\n[Sample Diagnostics for First 10 Tests]")
    for i, diag in enumerate(diagnostics[:10]):
        prob = final_probs[i]
        pred_class = test_preds[i]
        pred_label = "At-Risk (1)" if pred_class == 1 else "Typical (0)"

        # String formatting for the task importance dictionary
        task_imp = ", ".join([f"{k}: {v:.2f}" for k, v in diag.task_importance_scores.items() if v > 0])

        log.info(f"\n  Test Case #{i+1}:")
        log.info(f"    Raw Confidence   : {diag.confidence:.4f} (Class {diag.predicted_class})")
        log.info(f"    P(at-risk)       : {prob:.4f}")
        log.info(f"    Final Prediction : {pred_label}")
        log.info(f"    Decision Path    : {diag.decision_path_readable}")
        log.info(f"    Domain Severity  : {diag.domain_severity_scores}")
        log.info(f"    Task Importance  : {task_imp if task_imp else 'N/A'}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def train(out_path: str, use_synth: bool, use_thresholding: bool) -> None:
    log.info("=" * 60)
    log.info("FunaDB C4.5 Deployment Training")
    log.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────
    log.info("\n[1/6] Loading datasets...")
    r_train = load_csv("datasets/processed/train_deployment.csv", "real train")
    val_df  = load_csv("datasets/processed/val_deployment.csv",   "validation")
    test_df = load_csv("datasets/processed/test_deployment.csv",  "test")

    if use_synth:
        s_train  = load_csv("datasets/processed/s_train_deployment.csv", "synthetic train")
        train_df = pd.concat([r_train, s_train], ignore_index=True)
        mode     = "Synthetic-Augmented (TSTR)"
        params   = (
            TSTR_THRESHOLDED_PARAMS.copy()
            if use_thresholding
            else TSTR_NO_THRESHOLD_PARAMS.copy()
        )
    else:
        train_df = r_train.copy()
        mode     = "Real-Only (TRTR)"
        params   = (
            TRTR_THRESHOLDED_PARAMS.copy()
            if use_thresholding
            else TRTR_NO_THRESHOLD_PARAMS.copy()
        )

    threshold = params.pop("threshold", None)
    prediction_rule = (
        f"Probability threshold at {threshold:.2f}"
        if use_thresholding
        else "Native C4.5 class prediction"
    )

    log.info(f"Mode         : {mode}")
    log.info(f"Prediction   : {prediction_rule}")
    log.info(f"Train shape  : {train_df.shape}")
    log.info(f"Val shape    : {val_df.shape}")
    log.info(f"Test shape   : {test_df.shape}")

    X_train, y_train = split_xy(train_df)
    X_val,   y_val   = split_xy(val_df)
    X_test,  y_test  = split_xy(test_df)

    # ── 2. Train tree ─────────────────────────────────────────
    log.info("\n[2/6] Training C4.5 decision tree...")
    tree = C45DecisionTree(**params, feature_domain_mapping=DOMAIN_MAPPING)
    tree.fit(X_train, y_train, raw_features=RAW_FEATURES)

    log.info(f"Tree depth  : {tree.get_depth()}")
    log.info(f"Tree leaves : {tree.get_leaves_num()}")

    # ── 3. Evaluate validation set ────────────────────────────
    log.info("\n[3/6] Evaluating validation set...")
    val_preds, _ = predict_for_evaluation(tree, X_val, threshold)
    val_metrics = compute_metrics(y_val, val_preds)
    print_metrics("Validation metrics", val_metrics, threshold)

    # ── 4. Evaluate test set ──────────────────────────────────
    log.info("\n[4/6] Evaluating held-out test set...")
    test_preds, _ = predict_for_evaluation(tree, X_test, threshold)
    test_metrics = compute_metrics(y_test, test_preds)
    print_metrics("Test metrics", test_metrics, threshold)

    # ── 5. Save model ─────────────────────────────────────────
    log.info("\n[5/6] Saving model...")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    saved_threshold = threshold if threshold is not None else 0.50
    tree.save_model(str(out), optimal_threshold=saved_threshold)
    if threshold is None:
        log.info("Saved threshold 0.50 for package compatibility; this run used native C4.5 predictions.")
    log.info(f"Model saved → {out.resolve()}")
    
    # ── 6. Save Tree Visualization ────────────────────────────
    log.info("\n[6/6] Saving tree visualization...")
    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    svg_base = fig_dir / f"{out.stem}_full_tree"
    
    export_tree_svg(tree, str(svg_base))
    log.info(f"Tree visualization saved → {svg_base}.svg")
    
    log.info("\nDone.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save the FunaDB C4.5 deployment model.")
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
    parser.add_argument(
        "--no-threshold",
        action="store_true",
        help="Use native C4.5 class predictions instead of the validation-selected probability threshold.",
    )
    args = parser.parse_args()

    try:
        use_thresholding = not args.no_threshold
        train(out_path=args.out, use_synth=not args.no_synth, use_thresholding=use_thresholding)
        demonstrate_model(model_path=args.out, use_thresholding=use_thresholding)
    except FileNotFoundError as e:
        log.error(f"Dataset not found: {e}")
        sys.exit(1)
    except Exception as e:
        log.exception(f"Training failed: {e}")
        sys.exit(1)
