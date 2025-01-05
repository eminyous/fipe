import re
from typing import override

import numpy as np

from ...typing import MNumber, Number, XGBoostNode, XGBoostTree
from .parser import Parser


class XGBoostParser(Parser[XGBoostTree, XGBoostNode]):
    ID_KEY = "ID"
    NODE_KEY = "Node"
    FEATURE_KEY = "Feature"
    THRESHOLD_KEY = "Split"
    LEFT_CHILD_KEY = "Yes"
    RIGHT_CHILD_KEY = "No"
    VALUE_KEY = "Gain"

    IS_LEAF = "Leaf"
    FEATURE_PATTERN = r"f(\d+)"

    @override
    def parse_n_nodes(self) -> int:
        return len(self.base)

    @override
    def parse_root(self) -> XGBoostNode:
        return self.base.xs(self.DEFAULT_ROOT_ID, level=self.NODE_KEY).iloc[0]

    @override
    def read_node(self, node: XGBoostNode) -> tuple[int, Number]:
        matcher = re.match(self.FEATURE_PATTERN, node[self.FEATURE_KEY])
        if matcher is None:
            raise ValueError

        index = int(matcher.group(1))
        threshold = Number(node[self.THRESHOLD_KEY])
        return index, threshold

    @override
    def read_children(
        self,
        node: XGBoostNode,
    ) -> tuple[XGBoostNode, XGBoostNode]:
        left_id = str(node[self.LEFT_CHILD_KEY])
        right_id = str(node[self.RIGHT_CHILD_KEY])
        left = self.base.xs(left_id, level=self.ID_KEY).iloc[0]
        right = self.base.xs(right_id, level=self.ID_KEY).iloc[0]
        return left, right

    @override
    def read_leaf(self, node: XGBoostNode) -> MNumber:
        return np.array(node[self.VALUE_KEY])

    @override
    def is_leaf(self, node: XGBoostNode) -> bool:
        return str(node[self.FEATURE_KEY]) == self.IS_LEAF

    @override
    def read_node_id(self, node: XGBoostNode) -> int:
        return self._read_node_id_static(node=node)

    @staticmethod
    def _read_node_id_static(node: XGBoostNode) -> int:
        return int(str(node.name))
