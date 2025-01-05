from typing import override

import numpy as np

from ...typing import (
    LightGBMNode,
    LightGBMTree,
    MNumber,
    Number,
)
from .generic import GenericParser


class LightGBMParser(GenericParser[LightGBMTree, LightGBMNode]):
    NUM_LEAVES_KEY = "num_leaves"
    NUM_CAT_KEY = "num_cat"
    TREE_STRUCTURE_KEY = "tree_structure"

    LEAF_INDEX_KEY = "leaf_index"
    LEAF_VALUE_KEY = "leaf_value"
    SPLIT_INDEX_KEY = "split_index"
    SPLIT_FEATURE_KEY = "split_feature"
    THRESHOLD_KEY = "threshold"

    CHILD_KEY_FMT = "{which}_child"

    @override
    def parse_n_nodes(self) -> int:
        n_leaves = int(self.base[self.NUM_LEAVES_KEY])
        self.leaf_offset = n_leaves - 1
        return 2 * n_leaves - 1

    @override
    def parse_root(self) -> LightGBMNode:
        return self.base[self.TREE_STRUCTURE_KEY]

    @override
    def read_node(self, node: LightGBMNode) -> tuple[int, Number]:
        index = int(node[self.SPLIT_FEATURE_KEY])
        threshold = Number(node[self.THRESHOLD_KEY])
        return index, threshold

    @override
    def read_children(
        self,
        node: LightGBMNode,
    ) -> tuple[LightGBMNode, LightGBMNode]:
        whichs = ("left", "right")
        keys = (self.CHILD_KEY_FMT.format(which=which) for which in whichs)
        children = (node[key] for key in keys)
        return tuple(children)

    @override
    def read_leaf(self, node: LightGBMNode) -> MNumber:
        return np.array(node[self.LEAF_VALUE_KEY])

    @override
    def is_leaf(self, node: LightGBMNode) -> bool:
        return self.SPLIT_INDEX_KEY not in node

    @override
    def read_node_id(self, node: LightGBMNode) -> int:
        if self.SPLIT_INDEX_KEY in node:
            return int(node[self.SPLIT_INDEX_KEY])
        if self.LEAF_INDEX_KEY in node:
            return int(node[self.LEAF_INDEX_KEY]) + self.leaf_offset
        raise ValueError
