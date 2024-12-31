from collections.abc import Mapping
from typing import Generic

from ..feature.types import FeatureType
from ..typing import LV, Number, ParsableTree
from .base import BaseTree


class TreeContainer(Generic[LV]):
    _tree: BaseTree[LV, ParsableTree]

    def __init__(self, tree: BaseTree[LV, ParsableTree]) -> None:
        self._tree = tree

    @property
    def n_nodes(self) -> int:
        return self._tree.n_nodes

    @property
    def max_depth(self) -> int:
        return self._tree.max_depth

    @property
    def root_id(self) -> int:
        return self._tree.root_id

    @property
    def leaves(self) -> set[int]:
        return self._tree.leaves

    @property
    def internal_nodes(self) -> set[int]:
        return self._tree.internal_nodes

    @property
    def nodes(self) -> set[int]:
        return self._tree.nodes

    def nodes_at_depth(self, depth: int) -> set[int]:
        return self._tree.nodes_at_depth(depth)

    def nodes_split_on(self, feature: str) -> set[int]:
        return self._tree.nodes_split_on(feature)

    @property
    def left(self) -> Mapping[int, int]:
        return self._tree.left

    @property
    def right(self) -> Mapping[int, int]:
        return self._tree.right

    @property
    def node_value(self) -> Mapping[int, LV]:
        return self._tree.leaf_value

    @property
    def threshold(self) -> Mapping[int, Number]:
        return self._tree.threshold

    @property
    def category(self) -> Mapping[int, str]:
        return self._tree.category

    @property
    def types(self) -> Mapping[str, FeatureType]:
        return self._tree.types
