from ...typing import LightGBMParsableNode, LightGBMParsableTree, Number
from ..parser import GenericTreeParser


class LightGBMTreeParser(
    GenericTreeParser[LightGBMParsableTree, LightGBMParsableNode]
):
    NUM_LEAVES_KEY = "num_leaves"
    NUM_CAT_KEY = "num_cat"
    TREE_STRUCTURE_KEY = "tree_structure"

    LEAF_INDEX_KEY = "leaf_index"
    LEAF_VALUE_KEY = "leaf_value"
    SPLIT_INDEX_KEY = "split_index"
    SPLIT_FEATURE_KEY = "split_feature"
    THRESHOLD_KEY = "threshold"

    CHILD_KEY_FMT = "{which}_child"

    def parse_n_nodes(self) -> int:
        n_leaves = int(self.base[self.NUM_LEAVES_KEY])
        self.leaf_offset = n_leaves - 1
        return 2 * n_leaves - 1

    def parse_root(self) -> LightGBMParsableNode:
        return self.base[self.TREE_STRUCTURE_KEY]

    def get_internal_node(
        self,
        node: LightGBMParsableNode,
    ) -> tuple[int, float]:
        column_index = int(node[self.SPLIT_FEATURE_KEY])
        threshold = float(node[self.THRESHOLD_KEY])
        return column_index, threshold

    def get_children(
        self,
        node: LightGBMParsableNode,
    ) -> tuple[LightGBMParsableNode, LightGBMParsableNode]:
        whichs = ("left", "right")
        keys = (self.CHILD_KEY_FMT.format(which=which) for which in whichs)
        children = map(dict, map(node.get, keys))
        return tuple(children)

    def get_leaf_value(self, node: LightGBMParsableNode) -> Number:
        return Number(node[self.LEAF_VALUE_KEY])

    def is_leaf(self, node: LightGBMParsableNode) -> bool:
        return self.LEAF_INDEX_KEY in node

    def read_node_id(self, node: LightGBMParsableNode) -> int:
        if self.LEAF_INDEX_KEY in node:
            return int(node[self.LEAF_INDEX_KEY]) + self.leaf_offset
        if self.SPLIT_INDEX_KEY in node:
            return int(node[self.SPLIT_INDEX_KEY])
        raise ValueError
