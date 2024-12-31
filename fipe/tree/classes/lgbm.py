from typing import Literal

from ...typing import Number, ParsableTreeLGBM
from ..base import BaseTree


class TreeLGBM(BaseTree[Number, ParsableTreeLGBM]):
    NUM_LEAVES_KEY = "num_leaves"
    NUM_CAT_KEY = "num_cat"
    TREE_STRUCTURE_KEY = "tree_structure"

    LEAF_INDEX_KEY = "leaf_index"
    LEAF_VALUE_KEY = "leaf_value"
    SPLIT_INDEX_KEY = "split_index"
    SPLIT_FEATURE_KEY = "split_feature"
    THRESHOLD_KEY = "threshold"

    CHILD_KEY = "child"
    CHILD_KEY_FMT = "{which}_{key}"

    def predict(self, leaf_index: int) -> Number:
        leaf = self.leaf_offset + leaf_index
        return self.leaf_value[leaf]

    def _parse_tree(self, tree: ParsableTreeLGBM) -> None:
        self.root_id = 0
        self.n_leaves = int(tree[self.NUM_LEAVES_KEY])

        root = tree[self.TREE_STRUCTURE_KEY]
        self._parse_node(node=root, depth=0)

    def _parse_node(self, node: ParsableTreeLGBM, depth: int) -> None:
        node_id = self._get_node_id(node)
        self.node_depth[node_id] = depth
        if self.LEAF_INDEX_KEY in node:
            self._read_leaf(node=node, node_id=node_id)
        elif self.SPLIT_INDEX_KEY in node:
            self._read_internal(node=node, node_id=node_id)
            self._read_children(node=node, node_id=node_id, depth=depth)
        else:
            msg = "Invalid node structure."
            raise ValueError(msg)

    def _get_node_id(self, node: ParsableTreeLGBM) -> int:
        if self.LEAF_INDEX_KEY in node:
            return int(node[self.LEAF_INDEX_KEY]) + self.leaf_offset
        if self.SPLIT_INDEX_KEY in node:
            return int(node[self.SPLIT_INDEX_KEY])
        msg = "Invalid node structure."
        raise ValueError(msg)

    def _read_leaf(self, node: ParsableTreeLGBM, node_id: int) -> None:
        value = float(node[self.LEAF_VALUE_KEY])
        self._set_leaf(node=node_id, value=value)

    def _read_internal(self, node: ParsableTreeLGBM, node_id: int) -> None:
        index = int(node[self.SPLIT_FEATURE_KEY])
        threshold = float(node[self.THRESHOLD_KEY])
        self._set_internal_node(node=node_id, index=index, threshold=threshold)

    def _read_child(
        self,
        node: ParsableTreeLGBM,
        node_id: int,
        depth: int,
        which: Literal["left", "right"],
    ) -> None:
        child_key = self.CHILD_KEY_FMT.format(key=self.CHILD_KEY, which=which)
        child = dict(node[child_key])
        child_id = self._get_node_id(node=child)
        child_depth = depth + 1
        self._set_child(node=node_id, child=child_id, which=which)
        self._parse_node(node=child, depth=child_depth)

    def _read_children(
        self,
        node: ParsableTreeLGBM,
        node_id: int,
        depth: int,
    ) -> None:
        whichs = (self.LEFT_NAME, self.RIGHT_NAME)
        for which in whichs:
            self._read_child(
                node=node,
                node_id=node_id,
                depth=depth,
                which=which,
            )
