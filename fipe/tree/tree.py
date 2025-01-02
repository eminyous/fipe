from collections.abc import Iterable, Iterator
from typing import Literal

from ..feature import FeatureContainer, FeatureEncoder
from ..typing import LeafValue, Number


class Tree(FeatureContainer, Iterable[int]):
    """
    Abstract class for tree models.

    Each base tree has:
    - n_leaves: number of leaves
    - root_id: root node id
    - internal_nodes: set of internal nodes
    - leaves: set of leaves

    - max_depth: maximum depth of the tree
    - node_depth: dictionary with node depths
    - node_feature: dictionary with split features

    - left: dictionary with left child nodes
    - right: dictionary with right child nodes

    - threshold: dictionary with split thresholds
    - category: dictionary with split categories
    - value: dictionary with leaf values (LT)

    Each tree can be iterated over its nodes.

    An implementation of this class should implement the
    _parse_tree method to parse a tree.
    """

    LEFT_NAME = "left"
    RIGHT_NAME = "right"

    n_leaves: int
    root_id: int
    internal_nodes: set[int]
    leaves: set[int]

    node_depth: dict[int, int]
    node_feature: dict[int, str]

    left: dict[int, int]
    right: dict[int, int]

    threshold: dict[int, Number]
    category: dict[int, str]

    leaf_value: dict[int, LeafValue]

    __leaf_offset: int

    __node_at_depth: dict[int, set[int]]
    __node_split_on: dict[str, set[int]]
    __max_depth: int | None

    def __init__(self, encoder: FeatureEncoder) -> None:
        FeatureContainer.__init__(self, encoder=encoder)
        self.internal_nodes = set()
        self.leaves = set()

        self.node_depth = {}
        self.node_feature = {}

        self.left = {}
        self.right = {}

        self.threshold = {}
        self.category = {}
        self.leaf_value = {}

        self.__node_at_depth = {}
        self.__node_split_on = {}
        self.__max_depth = None
        self.__leaf_offset = 0

    @property
    def max_depth(self) -> int:
        if self.__max_depth is None:
            self.__max_depth = max(self.node_depth.values())
        return self.__max_depth

    def nodes_at_depth(self, depth: int) -> set[int]:
        if depth not in self.__node_at_depth:
            nodes = self._nodes_at_depth(depth=depth)
            self.__node_at_depth[depth] = nodes
        return self.__node_at_depth[depth]

    def nodes_split_on(self, feature: str) -> set[int]:
        if feature not in self.__node_split_on:
            nodes = self._nodes_split_on(feature=feature)
            self.__node_split_on[feature] = nodes
        return self.__node_split_on[feature]

    def add_internal_node(
        self,
        node: int,
        column_index: int,
        threshold: Number | None,
    ) -> None:
        self.internal_nodes.add(node)
        column = self.columns[column_index]
        if column in self.inverse_categories:
            self.category[node] = column
        feature = self.inverse_categories.get(column, column)
        self.node_feature[node] = feature
        if feature in self.continuous:
            self.threshold[node] = threshold

    def add_child(
        self,
        node: int,
        child: int,
        which: Literal["left", "right"],
    ) -> None:
        if which == self.LEFT_NAME:
            self.left[node] = child
        elif which == self.RIGHT_NAME:
            self.right[node] = child
        else:
            msg = "Invalid child name."
            raise ValueError(msg)

    def add_leaf(self, node: int, value: LeafValue) -> None:
        self.leaves.add(node)
        self.leaf_value[node] = value

    @property
    def leaf_offset(self) -> int:
        return self.__leaf_offset

    @leaf_offset.setter
    def leaf_offset(self, offset: int) -> None:
        self.__leaf_offset = offset

    @property
    def n_nodes(self) -> int:
        return 2 * self.n_leaves - 1

    def predict(self, leaf_index: int) -> LeafValue:
        leaf = self.leaf_offset + leaf_index
        return self.leaf_value[leaf]

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.n_nodes))

    def __len__(self) -> int:
        return self.n_nodes

    def _nodes_at_depth(self, depth: int) -> set[int]:
        return {
            node
            for node in self.internal_nodes
            if self.node_depth[node] == depth
        }

    def _nodes_split_on(self, feature: str) -> set[int]:
        return {
            node
            for node in self.internal_nodes
            if self.node_feature[node] == feature
        }
