import re
from typing import override

import numpy as np

from ...feature import FeatureEncoder
from ...typing import MNumber, Number, XGBoostParsableNode, XGBoostParsableTree
from .generic import GenericTreeParser


class XGBoostTreeParser(
    GenericTreeParser[XGBoostParsableTree, XGBoostParsableNode],
):
    ID_KEY = "ID"
    NODE_KEY = "Node"
    FEATURE_KEY = "Feature"
    THRESHOLD_KEY = "Split"
    LEFT_CHILD_KEY = "Yes"
    RIGHT_CHILD_KEY = "No"
    VALUE_KEY = "Gain"

    IS_LEAF = "Leaf"
    FEATURE_PATTERN = r"f(\d+)"

    def __init__(self, encoder: FeatureEncoder) -> None:
        GenericTreeParser.__init__(self, encoder=encoder)

    @override
    def parse_n_nodes(self) -> int:
        return len(self.base)

    @override
    def parse_root(self) -> XGBoostParsableNode:
        return self.base.xs(self.DEFAULT_ROOT_ID, level=self.NODE_KEY).iloc[0]

    @override
    def read_node(self, node: XGBoostParsableNode) -> tuple[int, Number]:
        matcher = re.match(self.FEATURE_PATTERN, node[self.FEATURE_KEY])
        if matcher is None:
            raise ValueError

        index = int(matcher.group(1))
        threshold = Number(node[self.THRESHOLD_KEY])
        return index, threshold

    @override
    def read_children(
        self,
        node: XGBoostParsableNode,
    ) -> tuple[XGBoostParsableNode, XGBoostParsableNode]:
        left_id = str(node[self.LEFT_CHILD_KEY])
        right_id = str(node[self.RIGHT_CHILD_KEY])
        left = self.base.xs(left_id, level=self.ID_KEY).iloc[0]
        right = self.base.xs(right_id, level=self.ID_KEY).iloc[0]
        return left, right

    @override
    def read_leaf(self, node: XGBoostParsableNode) -> MNumber:
        return np.array(node[self.VALUE_KEY])

    @override
    def is_leaf(self, node: XGBoostParsableNode) -> bool:
        return str(node[self.FEATURE_KEY]) == self.IS_LEAF

    @override
    def read_node_id(self, node: XGBoostParsableNode) -> int:
        return self._read_node_id_static(node=node)

    @staticmethod
    def _read_node_id_static(node: XGBoostParsableNode) -> int:
        return int(str(node.name))
