import re

from ...feature import FeatureEncoder
from ...typing import Number, XGBoostParsableNode, XGBoostParsableTree
from ..parser import GenericTreeParser


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

    def parse_n_nodes(self) -> int:
        return len(self.base)

    def parse_root(self) -> XGBoostParsableNode:
        return self.base.xs(self.DEFAULT_ROOT_ID, level=self.NODE_KEY).iloc[0]

    def get_internal_node(self, node: XGBoostParsableNode) -> tuple[int, float]:
        matcher = re.match(self.FEATURE_PATTERN, node[self.FEATURE_KEY])
        if matcher is None:
            raise ValueError

        column_index = int(matcher.group(1))
        threshold = float(node[self.THRESHOLD_KEY])
        return column_index, threshold

    def get_children(
        self,
        node: XGBoostParsableNode,
    ) -> tuple[XGBoostParsableNode, XGBoostParsableNode]:
        left_id = str(node[self.LEFT_CHILD_KEY])
        right_id = str(node[self.RIGHT_CHILD_KEY])
        left = self.base.xs(left_id, level=self.ID_KEY).iloc[0]
        right = self.base.xs(right_id, level=self.ID_KEY).iloc[0]
        return left, right

    def get_leaf_value(self, node: XGBoostParsableNode) -> Number:
        return Number(node[self.VALUE_KEY])

    def is_leaf(self, node: XGBoostParsableNode) -> bool:
        return str(node[self.FEATURE_KEY]) == self.IS_LEAF

    def read_node_id(self, node: XGBoostParsableNode) -> int:
        return self._read_node_id_static(node=node)

    @staticmethod
    def _read_node_id_static(node: XGBoostParsableNode) -> int:
        return int(node.name)
