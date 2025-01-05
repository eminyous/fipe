from typing import override

import numpy as np

from ...feature import FeatureEncoder
from ...typing import MNumber, Number, SKLearnParsableNode, SKLearnParsableTree
from .generic import GenericTreeParser


class SKLearnTreeParser(
    GenericTreeParser[SKLearnParsableTree, SKLearnParsableNode],
):
    _use_hard_voting: bool

    def __init__(
        self,
        encoder: FeatureEncoder,
        *,
        use_hard_voting: bool = False,
    ) -> None:
        GenericTreeParser.__init__(self, encoder=encoder)
        self._use_hard_voting = use_hard_voting

    @override
    def parse_n_nodes(self) -> int:
        return self.base.node_count

    @override
    def parse_root(self) -> SKLearnParsableNode:
        return self.DEFAULT_ROOT_ID

    @override
    def read_node(self, node: int) -> tuple[int, Number]:
        index = int(self.base.feature[node])  # pyright: ignore[reportAttributeAccessIssue]
        threshold = Number(self.base.threshold[node])  # pyright: ignore[reportAttributeAccessIssue]
        return index, threshold

    @override
    def read_leaf(self, node: SKLearnParsableNode) -> MNumber:
        value = np.array(self.base.value[node][0], dtype=Number).flatten()
        if value.size == 1:
            return value[0]

        if self._use_hard_voting:
            k = value.size
            q = np.argmax(value)
            value = np.eye(k)[q]
        return value

    @override
    def read_children(self, node: int) -> tuple[int, int]:
        left = int(self.base.children_left[node])  # pyright: ignore[reportAttributeAccessIssue]
        right = int(self.base.children_right[node])  # pyright: ignore[reportAttributeAccessIssue]
        return left, right

    @override
    def is_leaf(self, node: SKLearnParsableNode) -> bool:
        left = SKLearnParsableNode(self.base.children_left[node])  # pyright: ignore[reportAttributeAccessIssue]
        right = SKLearnParsableNode(self.base.children_right[node])  # pyright: ignore[reportAttributeAccessIssue]
        return left == right

    @override
    def read_node_id(self, node: SKLearnParsableNode) -> int:
        return self._read_node_id_static(node=node)

    @staticmethod
    def _read_node_id_static(node: SKLearnParsableNode) -> int:
        return node
