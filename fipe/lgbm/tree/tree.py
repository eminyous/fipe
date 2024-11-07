from collections.abc import Iterable, Iterator
from typing import Any, Literal

from ...feature import FeatureContainer, FeatureEncoder
from ...typing import Node
from ..typing import BaseTree

NUM_LEAVES_KEY = "num_leaves"
NUM_CAT_KEY = "num_cat"
TREE_STRUCTURE_KEY = "tree_structure"

LEAF_INDEX_KEY = "leaf_index"
LEAF_VALUE_KEY = "leaf_value"
SPLIT_INDEX_KEY = "split_index"
CHILD_KEY = "child"
LEFT_NAME = "left"
RIGHT_NAME = "right"
LEFT_CHILD_KEY = "left_child"
RIGHT_CHILD_KEY = "right_child"
SPLIT_FEATURE_KEY = "split_feature"
THRESHOLD_KEY = "threshold"


class Tree(FeatureContainer, Iterable[Node]):
    n_leaves: int
    root_id: Node
    internal_nodes: set[Node]
    leaves: set[Node]

    max_depth: int
    node_depth: dict[Node, int]

    left: dict[Node, Node]
    right: dict[Node, Node]

    feature: dict[Node, str]
    threshold: dict[Node, float]
    category: dict[Node, str]
    value: dict[Node, float]

    def __init__(
        self,
        tree: BaseTree,
        encoder: FeatureEncoder,
    ) -> None:
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
        return {
            node
            for node in self.internal_nodes
            if self.node_depth[node] == depth
        }

    def nodes_split_on(self, feature: str) -> set[Node]:
        return {
            node
            for node in self.internal_nodes
            if self.feature[node] == feature
        }

    def predict(self, leaf_index: int) -> float:
        leaf_id = leaf_index + self.leaf_offset
        return self.value[leaf_id]

    def __iter__(self) -> Iterator[Node]:
        return iter(range(self.n_nodes))

    def __len__(self) -> int:
        return self.n_nodes

    def _get_node_id(
        self,
        node: dict[str, Any],
    ) -> Node:
        if LEAF_INDEX_KEY in node:
            return int(node[LEAF_INDEX_KEY]) + self.leaf_offset
        if SPLIT_INDEX_KEY in node:
            return int(node[SPLIT_INDEX_KEY])
        msg = "Invalid node structure."
        raise ValueError(msg)

    def _read_leaf(self, node: dict[str, Any], node_id: Node) -> None:
        self.leaves.add(node_id)
        self.value[node_id] = node[LEAF_VALUE_KEY]

    def _read_internal(self, node: dict[str, Any], node_id: Node) -> None:
        self.internal_nodes.add(node_id)
        feature_index = node[SPLIT_FEATURE_KEY]
        feature = self.columns[feature_index]
        if feature in self.inverse_categories:
            self.category[node_id] = feature
            feature = self.inverse_categories[feature]
        self.feature[node_id] = feature
        if feature in self.continuous:
            self.threshold[node_id] = node[THRESHOLD_KEY]

    def _set_child(
        self,
        node_id: Node,
        child_id: Node,
        which: Literal["left", "right"],
    ) -> None:
        if which == LEFT_NAME:
            self.left[node_id] = child_id
        elif which == RIGHT_NAME:
            self.right[node_id] = child_id
        else:
            msg = "Invalid child name."
            raise ValueError(msg)

    def _read_child(
        self,
        node: dict[str, Any],
        node_id: Node,
        depth: int,
        which: Literal["left", "right"],
    ) -> None:
        child_key = f"{which}_{CHILD_KEY}"
        child = node[child_key]
        child_id = self._get_node_id(node=child)
        child_depth = depth + 1
        self._set_child(
            node_id=node_id,
            child_id=child_id,
            which=which,
        )
        self._parse_node(
            node=child,
            depth=child_depth,
        )

    def _read_children(
        self,
        node: dict[str, Any],
        node_id: Node,
        depth: int,
    ) -> None:
        for which in [LEFT_NAME, RIGHT_NAME]:
            self._read_child(
                node=node,
                node_id=node_id,
                depth=depth,
                which=which,
            )

    def _parse_node(
        self,
        node: dict[str, Any],
        depth: int,
    ) -> None:
        node_id = self._get_node_id(node)
        self.node_depth[node_id] = depth
        if LEAF_INDEX_KEY in node:
            self._read_leaf(node=node, node_id=node_id)
        elif SPLIT_INDEX_KEY in node:
            self._read_internal(node=node, node_id=node_id)
            self._read_children(
                node=node,
                node_id=node_id,
                depth=depth,
            )
        else:
            msg = "Invalid node structure."
            raise ValueError(msg)

    @property
    def leaf_offset(self) -> int:
        return self.n_leaves - 1

    @property
    def n_nodes(self) -> int:
        return 2 * self.n_leaves - 1

    def _parse_tree(self, tree: BaseTree) -> None:
        self.root_id = 0
        self.n_leaves = int(tree[NUM_LEAVES_KEY])

        root = tree[TREE_STRUCTURE_KEY]
        self._parse_node(node=root, depth=0)
        self.max_depth = max(self.node_depth.values())
