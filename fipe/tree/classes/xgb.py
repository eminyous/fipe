import re

import pandas as pd

from ...typing import Number, ParsableTreeXGB
from ..base import BaseTree


class TreeXGB(BaseTree[Number, ParsableTreeXGB]):
    FEATURE_KEY = "Feature"
    THRESHOLD_KEY = "Split"
    LEFT_CHILD_KEY = "Yes"
    RIGHT_CHILD_KEY = "No"
    VALUE_KEY = "Gain"
    ID_KEY = "ID"
    NODE_KEY = "Node"

    IS_LEAF = "Leaf"

    def _parse_tree(self, tree: ParsableTreeXGB) -> None:
        self.root_id = 0
        n_nodes = len(tree)
        self.n_leaves = (n_nodes + 1) // 2
        self._parse_node(tree, self.root_id, 0)

    def _parse_node(
        self,
        tree: ParsableTreeXGB,
        node_id: int,
        depth: int,
    ) -> None:
        node = tree.xs(node_id, level=self.NODE_KEY).iloc[0]
        self.node_depth[node_id] = depth
        feature = str(node[self.FEATURE_KEY])
        if feature == self.IS_LEAF:
            self._read_leaf(node=node, node_id=node_id)
        else:
            self._read_internal(node=node, node_id=node_id, feature=feature)
            self._read_children(node, node_id, depth)

    def _read_leaf(self, node: pd.Series, node_id: int) -> None:
        self.leaves.add(node_id)
        self.leaf_value[node_id] = float(node[self.VALUE_KEY])

    def _read_internal(
        self,
        node: pd.Series,
        node_id: int,
        feature: str,
    ) -> None:
        self.internal_nodes.add(node_id)
        index = self._get_feature_index(feature)
        threshold = float(node[self.THRESHOLD_KEY])
        self._set_internal(node=node_id, index=index, threshold=threshold)

    def _read_children(
        self,
        tree: ParsableTreeXGB,
        node: pd.Series,
        node_id: int,
        depth: int,
    ) -> None:
        left_id = node[self.LEFT_CHILD_KEY]
        right_id = node[self.RIGHT_CHILD_KEY]
        left = tree.xs(left_id, level=self.ID_KEY).iloc[0]
        right = tree.xs(right_id, level=self.ID_KEY).iloc[0]
        left_id = int(left.name)
        right_id = int(right.name)

        children = (left_id, right_id)
        whichs = (self.LEFT_NAME, self.RIGHT_NAME)
        for which, child in zip(whichs, children, strict=True):
            self._set_child(node_id, child, which)

        for child in children:
            self._parse_node(tree, child, depth + 1)

    @staticmethod
    def _get_feature_index(feature: str) -> int:
        matcher = re.match(r"f(\d+)", feature)
        if matcher is None:
            msg = f"Invalid feature name: {feature}"
            raise ValueError(msg)
        return int(matcher.group(1))
