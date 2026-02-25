from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from node import Node

class C45DecisionTree:

    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2, min_samples_leaf: int = 1, conf_fact: float = 0.25) -> None:
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
        self.feature_names: list[str] = []

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
            return Node(
                type="leaf", label=y[0], samples=len(y), distribution=Counter(y)
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

        if (
            (best_feature is None)
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

        if node.left and node.right:
            node.left = self._prune_tree(node.left)
            node.right = self._prune_tree(node.right)

            if node.samples:
                subtree_error = self._calc_subtree_err(
                    node
                )  # Calc error if retain subtree

                # Calc error for leaf substitution
                most_common_class = node.distribution.most_common(1)[0][0]
                error_as_leaf = node.samples - node.distribution[most_common_class]

                leaf_error = self._calc_pruning_error(node.samples, error_as_leaf)
                if leaf_error and subtree_error and leaf_error <= subtree_error:
                    return Node(
                        type="leaf",
                        label=most_common_class,
                        samples=node.samples,
                        distribution=node.distribution,
                    )
        return node

    def _calc_subtree_err(self, node: Node):
        """Calculate the total error of the subtree."""
        if node.type == "leaf" and node.samples:
            most_common_class = node.distribution.most_common(1)[0][0]
            errors = node.samples - node.distribution[most_common_class]
            return self._calc_pruning_error(node.samples, errors)

        if node.left and node.right:
            left_error = self._calc_subtree_err(node.left)
            right_error = self._calc_subtree_err(node.right)

            return left_error + right_error

    def fit(self, X: pd.DataFrame, y) -> C45DecisionTree:
        self.feature_names = X.columns.tolist()
        X_vals = X.values

        y_vals = np.array(y)

        # Build tree
        self.tree = self._build_tree(X_vals, y_vals)

        # Apply pruning to tree
        if self.tree is not None:
            self.tree = self._prune_tree(self.tree)

        return self

    def predict(self, x, node: Node) -> Optional[str]:
        if node.type == "leaf":
            return node.label

        feature_value = x[node.feature]

        if node.left and node.right:
            return (
                self.predict(x, node.left)
                if feature_value <= node.threshold
                else self.predict(x, node.right)
            )

        return None

    def print_tree(
        self, node: Optional[Node] = None, depth: int = 0, prefix: str = ""
    ) -> None:
        node = self.tree if node is None else node

        if node:
            if node.type == "leaf":
                dist = dict(node.distribution)
                print(
                    f"{prefix}Leaf: class = {node.label}, samples = {node.samples}, distribution = {dist}"
                )
            else:
                print(
                    f"{prefix}{node.feature} <= {node.threshold:.4f} (GR: {node.gain_ratio:.4f}, samples: {node.samples})"
                )
                self.print_tree(node.left, depth + 1, f"{prefix} L: ")
                self.print_tree(node.right, depth + 1, f"{prefix} R: ")

    def get_depth(self, node: Node | None = None) -> int:
        """Get the depth of the tree."""
        n = self.tree if node is None else node

        if n is None or n.type == "leaf":
            return 0

        left_depth = self.get_depth(n.left)
        right_depth = self.get_depth(n.right)

        return 1 + max(left_depth, right_depth)

    def get_leaves_num(self, node: Node | None = None) -> int:
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