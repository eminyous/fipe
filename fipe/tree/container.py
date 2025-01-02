from collections.abc import Mapping
from dataclasses import dataclass

from ..typing import FeatureType, LeafValue, Number
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
    def internal_nodes(self) -> set[int]:
        return self.tree.internal_nodes

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
    def node_value(self) -> Mapping[int, LeafValue]:
        return self.tree.leaf_value

    @property
    def threshold(self) -> Mapping[int, Number]:
        return self.tree.threshold

    @property
    def category(self) -> Mapping[int, str]:
        return self.tree.category

    @property
    def types(self) -> Mapping[str, FeatureType]:
        return self.tree.types
