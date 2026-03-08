from collections import Counter
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from Dataclasses import Node, DiagnosticOutput

class C45DecisionTree:
    def __init__(self, 
                 max_depth: Optional[int] = None, 
                 min_samples_split: int = 2, 
                 min_samples_leaf: int = 1, 
                 conf_fact: float = 0.25,
                 feature_domain_mapping: Optional[Dict[str, str]] = None
        ) -> None:
        """
        Object template for C4.5 Decision Tree Algorithm for ML.
        :param max_depth: Max
        :type max_depth: int | None
        :param min_samples_split: Description
        :type min_samples_split: int
        :param min_samples_leaf: Description
        :type min_samples_leaf: int
        :param conf_fact: Description
        :type conf_fact: float
        """
        self.max_depth: Optional[int] = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.conf_fact: float = conf_fact
        self.tree: Optional[Node] = None
        self.feature_names: List[str] = []
        self.feature_domain_mapping: dict[str, str] = feature_domain_mapping or {}

        # Global feature importance calculation
        self._feature_importance: Dict[str, float] = {}
        self._total_samples: int = 0

        self.epsilon = 1e-9  # to prevent invalid operations by 0

    def _entropy(self, y) -> float:
        """Calculate Entropy(X) = -∑ p_i * log2(p_i)"""
        if len(y) == 0:
            return 0

        counts = Counter(y)
        props = np.array([count / len(y) for count in counts.values()])
        return -np.sum(props * np.log2(props + self.epsilon))

    def _information_gain(self, y, subsets) -> float:
        """Calculate Gain(X, F) = H(X) - ∑ (|X_v| / |X|) * H(X_v)"""
        dataset_entropy = self._entropy(y)
        subsets_entropy = 0.0

        for subset in subsets:
            if len(subset) > 0:
                prop = len(subset) / len(y)
                subsets_entropy += prop * self._entropy(subset)

        return dataset_entropy - subsets_entropy

    def _split_information(self, y, subsets) -> float:
        """Calculate SplitInfo(X, F) = -∑ (|X_v| / |X|) * log2(|X_v| / |X|)"""
        split_info = 0
        total_samples = len(y)
        for subset in subsets:
            if len(subset) > 0:
                prop = len(subset) / total_samples
                split_info += prop * np.log2(prop + self.epsilon)

        return -split_info

    def _gain_ratio(self, y, subsets) -> float:
        """Calculate GainRatio(X, F) = Gain(X, F) / SplitInfo(X,F)"""
        information_gain = self._information_gain(y, subsets)
        split_info = self._split_information(y, subsets)

        return information_gain / (split_info + self.epsilon)

    def _node_error_rate(self, y) -> float:
        """Calculate the error rate for the Node"""
        if len(y) == 0:
            return 0

        most_common = Counter(y).most_common(1)[0][1]

        return (len(y) - most_common) / len(y)

    def _calc_pruning_error(self, n_samples: int, n_errors: int) -> float:
        if n_samples == 0:
            return 0

        # Use normal approximation for binomial confidence interval
        normalize_z = stats.norm.ppf(1 - self.conf_fact)

        # Error Rate
        error_rate = n_errors / n_samples

        # Upper confidence limit using pessimistic estimate
        upper_limit = (
            error_rate
            + normalize_z * np.sqrt(error_rate * (1 - error_rate) / n_samples)
            + normalize_z * normalize_z / (2 * n_samples)
        ) / (1 + normalize_z * normalize_z / n_samples)

        return upper_limit * n_samples

    def _split(self, X, y, feature: str):
        """Split data on a continuous feature by finding best threshold"""
        unique_values = np.unique(X[feature])

        if len(unique_values) <= 1:
            return None, None, None

        # Try thresholds between consecutive values
        thresholds = [
            (unique_values[i] + unique_values[i + 1]) / 2
            for i in range(len(unique_values) - 1)
        ]

        best_gr = -float("inf")
        best_threshold = None
        best_subset_X = None
        best_subset_y = None

        for threshold in thresholds:
            mask_left = X[feature] <= threshold
            mask_right = X[feature] > threshold

            subsets_y = [y[mask_left], y[mask_right]]

            # Skip if split creates subset < self.min_samples_leaf
            if (
                len(subsets_y[0]) < self.min_samples_leaf
                or len(subsets_y[1]) < self.min_samples_leaf
            ):
                continue

            gr = self._gain_ratio(y, subsets_y)

            if gr > best_gr:
                best_gr = gr
                best_threshold = threshold
                best_subset_X = [X[mask_left], X[mask_right]]
                best_subset_y = subsets_y

        return best_subset_X, best_subset_y, best_threshold

    def _best_split(self, X: pd.DataFrame, y):
        """Find the best feature to split on using gain ratio"""

        best_gr = -float("inf")
        best_feature = None
        best_subset_X = None
        best_subset_y = None
        best_threshold = None

        for feature in X.columns:
            subset_X, subset_y, threshold = self._split(X, y, feature)

            if subset_X is None:
                continue

            gr = self._gain_ratio(y, subset_y)

            if gr > best_gr:
                best_gr = gr
                best_feature = feature
                best_subset_X = subset_X
                best_subset_y = subset_y
                best_threshold = threshold
        return best_feature, best_subset_X, best_subset_y, best_threshold, best_gr

    def _build_tree(self, X, y, depth=0):
        """Recursive tree builder."""
        if len(np.unique(y)) == 1:  # Pure Node
            # Use .iloc[0] because y is a pandas Series that maintains its original index
            return Node(
                type="leaf", label=y.iloc[0], samples=len(y), distribution=Counter(y)
            )

        default_leaf_node = Node(
            type="leaf",
            label=Counter(y).most_common(1)[0][0],
            samples=len(y),
            distribution=Counter(y),
        )
        if len(y) < self.min_samples_split:  # Not enough samples
            return default_leaf_node

        if self.max_depth is not None and depth >= self.max_depth:
            return default_leaf_node

        best_feature, subsets_X, subsets_y, threshold, gain_ratio = self._best_split(X, y)

        if ((best_feature is None)
            or (subsets_X is None)
            or (subsets_y is None)
            or gain_ratio <= 0
        ):  # No valid split found
            return default_leaf_node

        # Recursively build children
        left = self._build_tree(subsets_X[0], subsets_y[0], depth=depth + 1)
        right = self._build_tree(subsets_X[1], subsets_y[1], depth=depth + 1)

        return Node(
            type="internal",
            distribution=Counter(y),
            samples=len(y),
            feature=best_feature,
            gain_ratio=gain_ratio,
            threshold=threshold,
            left=left,
            right=right,
        )

    def _prune_tree(self, node: Node) -> Node:
        """Prone tree using error-based pruning with cf."""
        if node.type == "leaf":
            return node
        
        if node.left:
            node.left = self._prune_tree(node.left)
        
        if node.right:
            node.right = self._prune_tree(node.right)

        if node.samples and node.samples > 0:
            subtree_error = self._calc_subtree_err(node)  # Calc error if retain subtree

            # Calc error for leaf substitution
            most_common_class = node.distribution.most_common(1)[0][0]
            error_as_leaf = node.samples - node.distribution[most_common_class]
            leaf_error = self._calc_pruning_error(node.samples, error_as_leaf)

            if (leaf_error is not None) and subtree_error and leaf_error <= subtree_error:
                return Node(
                    type="leaf",
                    label=most_common_class,
                    samples=node.samples,
                    distribution=node.distribution,
                )
            
        return node

    def _calc_subtree_err(self, node: Node):
        """Calculate the total error of the subtree."""
        if node.type == "leaf":
            if node.samples and node.samples > 0:
                most_common_class = node.distribution.most_common(1)[0][0]
                errors = node.samples - node.distribution[most_common_class]
                return self._calc_pruning_error(node.samples, errors)
            return 0

        left_error = 0 if node.left is None else self._calc_subtree_err(node.left)
        right_error = 0 if node.right is None else self._calc_subtree_err(node.right)

        return left_error + right_error
    
    def _calculate_global_feature_importance(self, node: Node) -> None:
        """
        Calculate global feature importance across the entire tree.
        Importance(F_j) = Σ GainRatio(node_n, F_j) * (|D_n| / |D|)
        """
        if node.type == "leaf" or node.feature is None:
            return
        
        # Weight by proportion of samples at this node
        if node.samples and node.gain_ratio:
            weight = node.samples / self._total_samples if self._total_samples > 0 else 0
            weighted_gain = node.gain_ratio * weight

            if node.feature not in self._feature_importance:
                self._feature_importance[node.feature] = 0
            
            self._feature_importance[node.feature] += weighted_gain

            if node.left:
                self._calculate_global_feature_importance(node.left)
            
            if node.right:
                self._calculate_global_feature_importance(node.right)

    def fit(self, X: pd.DataFrame, y) -> 'C45DecisionTree':
        self.feature_names = X.columns.tolist()
        
        # Ensure y is a pd.Series and indices align for safe boolean masking
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        self._total_samples = len(y)

        # Build tree keeping it as a pandas object
        self.tree = self._build_tree(X, y)

        # Apply pruning to tree
        if self.tree is not None:
            self.tree = self._prune_tree(self.tree)

            # Calculate global feature importance
            self._calculate_global_feature_importance(self.tree)

        return self

    def _predict_single(self, x: pd.Series, node: Node) -> Tuple[Optional[str], List[Node]]:
        """
        Predict a single instance and return the path taken.
        
        Returns:
            (predicted_label, path_nodes)
        """
        path = [node]

        while node.type != "leaf":
            if node.feature is None or node.threshold is None:
                break

            feature_value = x[node.feature]

            if feature_value <= node.threshold:
                if node.left is None:
                    break
                node = node.left
            else:
                if node.right is None:
                    break
                node = node.right
            
            path.append(node)

        return node.label if node.type == "leaf" else None, path

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X: DataFrame with same features as training data
            
        Returns:
            Array of predicted class labels
        """
        if self.tree is None:
            raise ValueError("Tree not fitted. Call fit() first.")

        predictions = []

        for _, row in X.iterrows():
            pred, _ = self._predict_single(row, self.tree)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_with_diagnostics(self, X: pd.DataFrame) -> List[DiagnosticOutput]:
        """
        Predict with comprehensive diagnostic outputs.
        
        Args:
            X: DataFrame with same features as training data
            
        Returns:
            List of DiagnosticOutput objects, one per sample
        """

        if self.tree is None:
            raise ValueError("Tree not fitted. Call fit() first.")
        
        diagnostics = []
        for _, row in X.iterrows():
            pred, path = self._predict_single(row, self.tree)
        
            if pred is None:
                pred = self.tree.label if (self.tree.type == "leaf" and self.tree.label is not None) else "Unknown"
                path = [self.tree]
            
            leaf_node = path[-1]
            leaf_dist = dict(leaf_node.distribution)
            total = sum(leaf_dist.values())
            confidence = leaf_dist.get(pred, 0) / (total + self.epsilon)

            decision_path = []
            decision_path_readable_parts = []

            for i in range(len(path) - 1):
                node, next_node = path[i:i+2]

                if node.feature and node.threshold is not None:
                    direction = "<=" if next_node == node.left else ">"
                    decision_path.append((node.feature, node.threshold, direction))
                    decision_path_readable_parts.append(f"{node.feature} {direction} {node.threshold:.4f}")
                
            decision_path_readable = " AND ".join(decision_path_readable_parts)
            if not decision_path_readable:
                decision_path_readable = f"Direct classification as {pred}"

            domain_severity = {}
            for domain in set(self.feature_domain_mapping.values()):
                domain_severity[domain] = 0.0

            for i in range(len(path) - 1):
                node = path[i]
                if node.feature and node.gain_ratio:
                    domain = self.feature_domain_mapping.get(node.feature)
                    if domain and domain in domain_severity:
                        domain_severity[domain] += node.gain_ratio
            
            task_importance = {}
            for feature in self.feature_names:
                task_importance[feature] = 0.0
            
            for i in range(len(path) - 1):
                node = path[i]
                if node.feature and node.gain_ratio:
                    task_importance[node.feature] += node.gain_ratio
            
            # Create diagnostic output
            diagnostic = DiagnosticOutput(
                predicted_class=pred,
                confidence=confidence,
                decision_path=decision_path,
                decision_path_readable=decision_path_readable,
                domain_severity_scores=domain_severity,
                task_importance_scores=task_importance,
                leaf_distribution=leaf_dist,
            )
            
            diagnostics.append(diagnostic)

        return diagnostics

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get global feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        return self._feature_importance.copy()
    
    def print_tree(self, node: Optional[Node] = None, depth: int = 0, prefix: str = "") -> None:
        node = self.tree if node is None else node

        if node:
            if node.type == "leaf":
                dist = dict(node.distribution)
                print(f"{prefix}Leaf: class = {node.label}, samples = {node.samples}, distribution = {dist}")
            else:
                print(f"{prefix}{node.feature} <= {node.threshold:.4f} (GR: {node.gain_ratio:.4f}, samples: {node.samples})")
                self.print_tree(node.left, depth + 1, f"{prefix} L: ")
                self.print_tree(node.right, depth + 1, f"{prefix} R: ")

    def get_depth(self, node: Optional[Node] = None) -> int:
        """Get the depth of the tree."""
        n = self.tree if node is None else node

        if n is None or n.type == "leaf":
            return 0

        left_depth = self.get_depth(n.left)
        right_depth = self.get_depth(n.right)

        return 1 + max(left_depth, right_depth)

    def get_leaves_num(self, node: Optional[Node] = None) -> int:
        """Get the number of leaves in the tree."""

        n = self.tree if node is None else node

        if n is None:
            return 0

        if n.type == "leaf":
            return 1

        return self.get_leaves_num(n.left) + self.get_leaves_num(n.right)


if __name__ == "__main__":
    decisionTree = C45DecisionTree()

    decisionTree.print_tree()