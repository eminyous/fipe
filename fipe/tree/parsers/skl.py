from abc import ABCMeta

import numpy as np

from ...feature import FeatureEncoder
from ...typing import MNumber, Number, SKLearnParsableTree
from ..parser import GenericTreeParser


class SKLearnTreeParser(GenericTreeParser[SKLearnParsableTree, int]):
    __metaclass__ = ABCMeta

    def __init__(self, encoder: FeatureEncoder) -> None:
        GenericTreeParser.__init__(self, encoder=encoder)

    def parse_n_nodes(self) -> int:
        return self.base.node_count

    def parse_root(self) -> int:
        return self.DEFAULT_ROOT_ID

    def get_internal_node(self, node: int) -> tuple[int, float]:
        column_index = int(self.base.feature[node])
        threshold = float(self.base.threshold[node])
        return column_index, threshold

    def get_children(self, node: int) -> tuple[int, int]:
        left = int(self.base.children_left[node])
        right = int(self.base.children_right[node])
        return left, right

    def is_leaf(self, node: int) -> bool:
        left = int(self.base.children_left[node])
        right = int(self.base.children_right[node])
        return left == right

    def read_node_id(self, node: int) -> int:
        return self._read_node_id_static(node=node)

    @staticmethod
    def _read_node_id_static(node: int) -> int:
        return node


class TreeParserCL(SKLearnTreeParser):
    use_hard_voting: bool

    def __init__(
        self,
        encoder: FeatureEncoder,
        *,
        use_hard_voting: bool,
    ) -> None:
        SKLearnTreeParser.__init__(self, encoder=encoder)
        self.use_hard_voting = use_hard_voting

    def get_leaf_value(self, node: int) -> MNumber:
        value = self.base.value[node].flatten()
        value = np.asarray(value)
        if self.use_hard_voting:
            k = value.size
            q = np.argmax(value)
            value = np.eye(k)[q]
        return value


class TreeParserRG(SKLearnTreeParser):
    def get_leaf_value(self, node: int) -> Number:
        return self.base.value[node].flatten()[0]
