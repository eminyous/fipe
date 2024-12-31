from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Generic, Literal

from ..feature import FeatureContainer, FeatureEncoder
from ..typing import LV, PT, Number


class BaseTree(FeatureContainer, Iterable[int], Generic[LV, PT]):
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

    Each tree can be iterated over its nodes and has a
    predict method that returns the value of a leaf node.

    An implementation of this class should implement the
    _parse_tree method to parse a tree.
    """

    __metaclass__ = ABCMeta

    LEFT_NAME = "left"
    RIGHT_NAME = "right"

    n_leaves: int
    root_id: int
    internal_nodes: set[int]
    leaves: set[int]

    max_depth: int
    node_depth: dict[int, int]
    node_feature: dict[int, str]

    left: dict[int, int]
    right: dict[int, int]

    threshold: dict[int, Number]
    category: dict[int, str]

    leaf_value: dict[int, LV]

    def __init__(
        self,
        tree: PT,
        encoder: FeatureEncoder,
    ) -> None:
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
        self._parse_tree(tree)
        self.max_depth = max(self.node_depth.values())

    def nodes_at_depth(self, depth: int) -> set[int]:
        return {
            node
            for node in self.internal_nodes
            if self.node_depth[node] == depth
        }

    def nodes_split_on(self, feature: str) -> set[int]:
        return {
            node
            for node in self.internal_nodes
            if self.node_feature[node] == feature
        }

    @property
    def leaf_offset(self) -> int:
        return self.n_leaves - 1

    @property
    def n_nodes(self) -> int:
        return 2 * self.n_leaves - 1

    def predict(self, leaf_index: int) -> LV:
        leaf_id = leaf_index + self.leaf_offset
        return self.leaf_value[leaf_id]

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.n_nodes))

    def __len__(self) -> int:
        return self.n_nodes

    @abstractmethod
    def _parse_tree(self, tree: PT) -> None:
        raise NotImplementedError

    def _set_child(
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

    def _set_internal(
        self,
        node: int,
        index: int,
        threshold: Number | None,
    ) -> None:
        column = self.columns[index]
        if column in self.inverse_categories:
            self.category[node] = column
        feature = self.inverse_categories.get(column, column)
        self.node_feature[node] = feature
        if feature in self.continuous:
            self.threshold[node] = threshold
