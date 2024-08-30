from collections.abc import Iterable, Iterator

import numpy as np

from ..feature.container import FeatureContainer
from ..feature.encoder import FeatureEncoder
from ..typing import BaseTree, Node, numeric


class Tree(FeatureContainer, Iterable[Node]):
    """
    Class to represent a tree.

    This class is a wrapper around the sklearn.tree.BaseTree

    Parameters
    ----------
    tree: _Tree
        The tree to be represented.
    encoder: FeatureEncoder
        The encoder of the dataset.

    Attributes
    ----------
    root: Node
        The root node of the tree.
    n_nodes: int
        The number of nodes in the tree.
    max_depth: int
        The maximum depth of the tree.
    internal_nodes: set[Node]
        The set of internal nodes in the tree.
    leaves: set[Node]
        The set of leaf nodes in the tree.
    node_depth: dict[Node, int]
        The depth of each node in the tree.
    left: dict[Node, Node]
        The left child of each internal node.
    right: dict[Node, Node]
        The right child of each internal node.
    feature: dict[Node, str]
        The feature split at each internal node.
    threshold: dict[Node, float]
        The threshold split at each internal node.
        This is only present for numerical encoder.
    category: dict[Node, str]
        The category split at each internal node.
        This is only present for categorical encoder.
    prob: defaultdict[int, dict[Node, float]]
        The probability of each class at each leaf node.

    """

    n_nodes: int
    root: Node
    internal_nodes: set[Node]
    leaves: set[Node]

    max_depth: int
    node_depth: dict[Node, int]

    left: dict[Node, Node]
    right: dict[Node, Node]

    feature: dict[Node, str]
    threshold: dict[Node, numeric]
    category: dict[Node, str]
    value: dict[Node, np.ndarray]

    n_classes: int

    def __init__(self, tree: BaseTree, encoder: FeatureEncoder) -> None:
        FeatureContainer.__init__(self, encoder)
        self.internal_nodes = set()
        self.leaves = set()

        self.node_depth = {}

        self.left = {}
        self.right = {}

        self.feature = {}
        self.threshold = {}
        self.category = {}
        self.value = {}

        self._parse_tree(tree)

    def nodes_at_depth(self, depth: int) -> set[Node]:
        return {n for n in self.internal_nodes if self.node_depth[n] == depth}

    def nodes_split_on(self, feature: str) -> set[Node]:
        return {n for n in self.internal_nodes if self.feature[n] == feature}

    def read_internal(self, tree: BaseTree, node: Node) -> None:
        i = tree.feature[node]
        f = self.columns[i]
        if f in self.inverse_categories:
            self.category[node] = f
            f = self.inverse_categories[f]
        self.feature[node] = f
        if f in self.continuous:
            self.threshold[node] = tree.threshold[node]

    def read_leaf(self, tree: BaseTree, node: Node) -> None:
        self.value[node] = tree.value[node].flatten()
        if self.n_classes == -1:
            self.n_classes = len(self.value[node])

    def __iter__(self) -> Iterator[Node]:
        return iter(range(self.n_nodes))

    def __len__(self) -> int:
        return self.n_nodes

    def _parse_tree(self, tree: BaseTree) -> None:
        self.root = 0
        self.n_nodes = tree.node_count
        self.n_classes = -1
        self._dfs(tree, self.root, 0)
        self.max_depth = max(self.node_depth.values())

    def _dfs(self, tree: BaseTree, node: Node, depth: int) -> None:
        self.node_depth[node] = depth
        left = tree.children_left[node]
        right = tree.children_right[node]
        if left == right:
            self.leaves.add(node)
            self.read_leaf(tree, node)
        else:
            self.internal_nodes.add(node)
            self.read_internal(tree, node)
            self.left[node] = left
            self.right[node] = right
            self._dfs(tree, left, depth + 1)
            self._dfs(tree, right, depth + 1)
