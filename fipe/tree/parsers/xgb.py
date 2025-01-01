import pandas as pd

from ...feature import FeatureEncoder
from ...typing import Number, XGBoostParsableTree
from ..parser import GenericTreeParser


class XGBoostTreeParser(GenericTreeParser[XGBoostParsableTree, pd.Series]):
    FEATURE_KEY = "Feature"
    THRESHOLD_KEY = "Split"
    LEFT_CHILD_KEY = "Yes"
    RIGHT_CHILD_KEY = "No"
    VALUE_KEY = "Gain"
    ID_KEY = "ID"
    NODE_KEY = "Node"

    IS_LEAF = "Leaf"

    def __init__(self, encoder: FeatureEncoder) -> None:
        GenericTreeParser.__init__(self, encoder=encoder)

    def parse_n_nodes(self) -> int:
        n_leaves = int(self.base[self.NUM_LEAVES_KEY])
        self.leaf_offset = n_leaves
        return 2 * n_leaves - 1

    def parse_root(self) -> pd.Series:
        return self.base.xs(self.DEFAULT_ROOT_ID, level=self.NODE_KEY).iloc[0]

    def get_internal_node(self, node: pd.Series) -> tuple[int, float]:
        column_index = int(node[self.FEATURE_KEY])
        threshold = float(node[self.THRESHOLD_KEY])
        return column_index, threshold

    def get_children(
        self,
        node: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        left_id = str(node[self.LEFT_CHILD_KEY])
        right_id = str(node[self.RIGHT_CHILD_KEY])
        left = self.base.xs(left_id, level=self.ID_KEY).iloc[0]
        right = self.base.xs(right_id, level=self.ID_KEY).iloc[0]
        return left, right

    def get_leaf_value(self, node: pd.Series) -> Number:
        return node[self.VALUE_KEY]

    def is_leaf(self, node: pd.Series) -> bool:
        return str(node[self.FEATURE_KEY]) == self.IS_LEAF

    def _read_node_id(self, node: pd.Series) -> int:
        return self._read_node_id_static(node=node)

    @staticmethod
    def _read_node_id_static(node: pd.Series) -> int:
        return int(node.name)
