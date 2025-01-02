from abc import ABCMeta

import numpy as np

from ...feature import FeatureEncoder
from ...typing import MNumber, Number, SKLearnParsableNode, SKLearnParsableTree
from ..parser import GenericTreeParser


class SKLearnTreeParser(
    GenericTreeParser[SKLearnParsableTree, SKLearnParsableNode],
):
    __metaclass__ = ABCMeta

    def __init__(self, encoder: FeatureEncoder) -> None:
        GenericTreeParser.__init__(self, encoder=encoder)

    def parse_n_nodes(self) -> int:
        return self.base.node_count

    def parse_root(self) -> SKLearnParsableNode:
        return self.DEFAULT_ROOT_ID

    def get_internal_node(self, node: int) -> tuple[int, float]:
        column_index = int(self.base.feature[node])
        threshold = float(self.base.threshold[node])
        return column_index, threshold

    def get_leaf_value(self, node: SKLearnParsableNode) -> Number:
        return Number(self.base.value[node].flatten()[0])

    def get_children(self, node: int) -> tuple[int, int]:
        left = int(self.base.children_left[node])
        right = int(self.base.children_right[node])
        return left, right

    def is_leaf(self, node: SKLearnParsableNode) -> bool:
        left = SKLearnParsableNode(self.base.children_left[node])
        right = SKLearnParsableNode(self.base.children_right[node])
        return left == right

    def read_node_id(self, node: SKLearnParsableNode) -> int:
        return self._read_node_id_static(node=node)

    @staticmethod
    def _read_node_id_static(node: SKLearnParsableNode) -> int:
        return node


class SKLearnTreeParserClassifier(SKLearnTreeParser):
    use_hard_voting: bool

    def __init__(
        self,
        encoder: FeatureEncoder,
        *,
        use_hard_voting: bool,
    ) -> None:
        super().__init__(encoder=encoder)
        self.use_hard_voting = use_hard_voting

    def get_leaf_value(self, node: SKLearnParsableNode) -> MNumber:
        value = self.base.value[node].flatten()
        value = np.asarray(value)
        if self.use_hard_voting:
            k = value.size
            q = np.argmax(value)
            value = np.eye(k)[q]
        return value
