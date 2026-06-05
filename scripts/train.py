"""
train.py — Deployment Training Script
======================================
Trains the final C4.5 TRSTR model on the full real + synthetic training set,
and saves the complete model package to disk.

Usage
-----
    python train.py                                    # TRSTR mode with thresholding
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
    "min_samples_leaf": 22,
    "max_depth":        15,
    "threshold":        0.40,
}

TRSTR_THRESHOLDED_PARAMS = {
    "conf_fact":        0.50,
    "min_samples_leaf": 39,
    "max_depth":        15,
    "threshold":        0.45,
}

TRTR_NO_THRESHOLD_PARAMS = {
    "conf_fact":        0.50,
    "min_samples_leaf": 12,
    "max_depth":        15,
}

TRSTR_NO_THRESHOLD_PARAMS = {
    "conf_fact":        0.50,
    "min_samples_leaf": 40,
    "max_depth":        15,
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



def export_tree_image(tree_model: C45DecisionTree, base_filename: str) -> None:
    """Exports the trained tree as SVG and PNG visualizations using Graphviz."""
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
        dot.render(base_filename, format='svg', cleanup=False)
        dot.render(base_filename, format='png', cleanup=True)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def train(out_path: str, use_synth: bool, use_thresholding: bool) -> None:
    log.info("=" * 60)
    log.info("FunaDB C4.5 Deployment Training")
    log.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────
    log.info("\n[1/3] Loading datasets...")
    r_train = load_csv("../datasets/processed/deployment/train_deployment.csv", "real train")
    val_df  = load_csv("../datasets/processed/deployment/val_deployment.csv",   "validation")
    test_df = load_csv("../datasets/processed/deployment/test_deployment.csv",  "test")

    # Combine all real data into a single training set
    r_full = pd.concat([r_train, val_df, test_df], ignore_index=True)

    if use_synth:
        s_train  = load_csv("../datasets/processed/deployment/s_full_deployment.csv", "synthetic train")
        train_df = pd.concat([r_full, s_train], ignore_index=True)
        mode     = "Synthetic-Augmented (TRSTR) Full"
        params   = (
            TRSTR_THRESHOLDED_PARAMS.copy()
            if use_thresholding
            else TRSTR_NO_THRESHOLD_PARAMS.copy()
        )
    else:
        train_df = r_full.copy()
        mode     = "Real-Only (TRTR) Full"
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
    log.info(f"Combined Train shape: {train_df.shape}")

    X_train, y_train = split_xy(train_df)

    # ── 2. Train tree ─────────────────────────────────────────
    log.info("\n[2/3] Training C4.5 decision tree...")
    tree = C45DecisionTree(**params, feature_domain_mapping=DOMAIN_MAPPING)
    tree.fit(X_train, y_train, raw_features=RAW_FEATURES)

    log.info(f"Tree depth  : {tree.get_depth()}")
    log.info(f"Tree leaves : {tree.get_leaves_num()}")

    # ── 2.5 Feature Importance ────────────────────────────────
    importances = tree.get_feature_importance()
    log.info("\nGlobal Feature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        log.info(f"  {feat:<5}: {imp:.4f}")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 5))
        sorted_idx = pd.Series(importances).sort_values(ascending=False)
        sns.barplot(x=sorted_idx.values, y=list(sorted_idx.index), ax=ax, palette="viridis")
        ax.set_title("Global Feature Importance (C4.5 Gain Ratio)")
        ax.set_xlabel("Normalized Importance")
        plt.tight_layout()
        fi_out = Path("outputs/figures/deployment") / f"{Path(out_path).stem}_feature_importance.png"
        fi_out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fi_out, dpi=300)
        plt.close()
        log.info(f"Feature importance plot saved → {fi_out}")
    except ImportError:
        log.warning("matplotlib or seaborn not installed; skipping feature importance plot.")

    # ── 3. Save model & Visualization ─────────────────────────
    log.info("\n[3/3] Saving model and visualization...")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    saved_threshold = threshold if threshold is not None else 0.50
    tree.save_model(str(out), optimal_threshold=saved_threshold)
    if threshold is None:
        log.info("Saved threshold 0.50 for package compatibility; this run used native C4.5 predictions.")
    log.info(f"Model saved → {out.resolve()}")
    
    fig_dir = Path("outputs/figures/deployment")
    fig_dir.mkdir(parents=True, exist_ok=True)
    svg_base = fig_dir / f"{out.stem}_full_tree"
    
    export_tree_image(tree, str(svg_base))
    log.info(f"Tree visualizations saved → {svg_base}.svg and {svg_base}.png")
    
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
        help="Train on real data only (TRTR mode). Default is TRSTR (real + synthetic).",
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
    except FileNotFoundError as e:
        log.error(f"Dataset not found: {e}")
        sys.exit(1)
    except Exception as e:
        log.exception(f"Training failed: {e}")
        sys.exit(1)
