from collections.abc import Iterable, Iterator
from typing import Literal

from ..feature import FeatureContainer, FeatureEncoder
from ..typing import MNumber, Number


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
    nodes: set[int]
    leaves: set[int]

    depth: dict[int, int]
    feature: dict[int, str]

    left: dict[int, int]
    right: dict[int, int]

    threshold: dict[int, Number]
    category: dict[int, str]

    leaf_value: dict[int, MNumber]

    __leaf_offset: int
    __node_at_depth: dict[int, set[int]]
    __node_split_on: dict[str, set[int]]
    __max_depth: int | None
    __leaf_value_shape: tuple[int, ...]

    def __init__(self, encoder: FeatureEncoder) -> None:
        FeatureContainer.__init__(self, encoder=encoder)
        self.nodes = set()
        self.leaves = set()

        self.depth = {}
        self.feature = {}

        self.left = {}
        self.right = {}

        self.threshold = {}
        self.category = {}
        self.leaf_value = {}

        self.__leaf_offset = 0
        self.__node_at_depth = {}
        self.__node_split_on = {}
        self.__max_depth = None
        self.__leaf_value_shape = ()

    @property
    def max_depth(self) -> int:
        if self.__max_depth is None:
            self.__max_depth = max(self.depth.values())
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

    def add_node(
        self,
        node: int,
        index: int,
        threshold: Number | None,
    ) -> None:
        self.nodes.add(node)
        column = self.columns[index]
        if column in self.inverse_categories:
            self.category[node] = column
        feature = self.inverse_categories.get(column, column)
        self.feature[node] = feature
        if feature in self.continuous and threshold is not None:
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

    def add_leaf(self, node: int, value: MNumber) -> None:
        self.leaves.add(node)
        self.leaf_value[node] = value
        self.__leaf_value_shape = value.shape

    @property
    def leaf_value_shape(self) -> tuple[int, ...]:
        return self.__leaf_value_shape

    @property
    def leaf_offset(self) -> int:
        return self.__leaf_offset

    @leaf_offset.setter
    def leaf_offset(self, offset: int) -> None:
        self.__leaf_offset = offset

    @property
    def n_nodes(self) -> int:
        return 2 * self.n_leaves - 1

    def predict(self, index: int) -> MNumber:
        leaf = self.leaf_offset + index
        return self.leaf_value[leaf]

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.n_nodes))

    def __len__(self) -> int:
        return self.n_nodes

    def _nodes_at_depth(self, depth: int) -> set[int]:
        return {node for node in self.nodes if self.depth[node] == depth}

    def _nodes_split_on(self, feature: str) -> set[int]:
        return {node for node in self.nodes if self.feature[node] == feature}
