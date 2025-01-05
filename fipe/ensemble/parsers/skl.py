from typing import override

import numpy as np

from ...feature import FeatureEncoder
from ...typing import MNumber, Number, SKLearnNode, SKLearnTree
from .generic import GenericParser


class SKLearnParser(GenericParser[SKLearnTree, SKLearnNode]):
    _use_hard_voting: bool

    def __init__(
        self,
        encoder: FeatureEncoder,
        *,
        use_hard_voting: bool = False,
    ) -> None:
        super().__init__(encoder=encoder)
        self._use_hard_voting = use_hard_voting

    @override
    def parse_n_nodes(self) -> int:
        return self.base.node_count

    @override
    def parse_root(self) -> SKLearnNode:
        return self.DEFAULT_ROOT_ID

    @override
    def read_node(self, node: int) -> tuple[int, Number]:
        index = int(self.base.feature[node])  # pyright: ignore[reportAttributeAccessIssue]
        threshold = Number(self.base.threshold[node])  # pyright: ignore[reportAttributeAccessIssue]
        return index, threshold

    @override
    def read_leaf(self, node: SKLearnNode) -> MNumber:
        value = np.array(self.base.value[node][0], dtype=Number).flatten()
        if value.size == 1:
            return value[0]

        if self._use_hard_voting:
            k = value.size
            q = np.argmax(value)
            value = np.eye(k)[q]
        return value

    @override
    def read_children(self, node: int) -> tuple[SKLearnNode, SKLearnNode]:
        left = SKLearnNode(self.base.children_left[node])  # pyright: ignore[reportAttributeAccessIssue]
        right = SKLearnNode(self.base.children_right[node])  # pyright: ignore[reportAttributeAccessIssue]
        return left, right

    @override
    def is_leaf(self, node: SKLearnNode) -> bool:
        left = SKLearnNode(self.base.children_left[node])  # pyright: ignore[reportAttributeAccessIssue]
        right = SKLearnNode(self.base.children_right[node])  # pyright: ignore[reportAttributeAccessIssue]
        return left == right

    @override
    def read_node_id(self, node: SKLearnNode) -> int:
        return self._read_node_id_static(node=node)

    @staticmethod
    def _read_node_id_static(node: SKLearnNode) -> int:
        return node
