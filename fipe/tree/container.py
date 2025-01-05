from collections.abc import Mapping
from dataclasses import dataclass

from ..typing import FeatureType, MNumber, Number
from .tree import Tree


@dataclass
class TreeContainer:
    tree: Tree

    @property
    def n_nodes(self) -> int:
        return self.tree.n_nodes

    @property
    def max_depth(self) -> int:
        return self.tree.max_depth

    @property
    def root_id(self) -> int:
        return self.tree.root_id

    @property
    def leaves(self) -> set[int]:
        return self.tree.leaves

    @property
    def nodes(self) -> set[int]:
        return self.tree.nodes

    def nodes_at_depth(self, depth: int) -> set[int]:
        return self.tree.nodes_at_depth(depth)

    def nodes_split_on(self, feature: str) -> set[int]:
        return self.tree.nodes_split_on(feature)

    @property
    def left(self) -> Mapping[int, int]:
        return self.tree.left

    @property
    def right(self) -> Mapping[int, int]:
        return self.tree.right

    @property
    def leaf_value(self) -> Mapping[int, MNumber]:
        return self.tree.leaf_value

    @property
    def leaf_value_shape(self) -> tuple[int, ...]:
        return self.tree.leaf_value_shape

    @property
    def threshold(self) -> Mapping[int, Number]:
        return self.tree.threshold

    @property
    def category(self) -> Mapping[int, str]:
        return self.tree.category

    @property
    def types(self) -> Mapping[str, FeatureType]:
        return self.tree.types
