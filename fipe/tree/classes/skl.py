from abc import ABCMeta, abstractmethod
from typing import Generic

from ...feature import FeatureEncoder
from ...typing import LV, ParsableTreeSKL
from ..base import BaseTree


class TreeSKL(BaseTree[LV, ParsableTreeSKL], Generic[LV]):
    __metaclass__ = ABCMeta

    def __init__(
        self,
        tree: ParsableTreeSKL,
        encoder: FeatureEncoder,
    ) -> None:
        super().__init__(tree, encoder=encoder)

    def _parse_tree(self, tree: ParsableTreeSKL) -> None:
        self.root_id = 0
        self.n_leaves = (tree.node_count + 1) // 2
        self._parse_node(tree, self.root_id, 0)

    def _parse_node(
        self,
        tree: ParsableTreeSKL,
        node: int,
        depth: int,
    ) -> None:
        self.node_depth[node] = depth
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left == right:
            self._read_leaf(tree, node)
        else:
            self._read_internal(tree, node)
            children = (left, right)
            self._read_children(
                tree=tree,
                node=node,
                children=children,
                depth=depth,
            )

    def _read_internal(self, tree: ParsableTreeSKL, node: int) -> None:
        index = int(tree.feature[node])
        threshold = float(tree.threshold[node])
        self._set_internal_node(node=node, index=index, threshold=threshold)

    def _read_children(
        self,
        tree: ParsableTreeSKL,
        node: int,
        children: tuple[int, int],
        depth: int,
    ) -> None:
        whichs = (self.LEFT_NAME, self.RIGHT_NAME)
        for which, child in zip(whichs, children, strict=True):
            self._set_child(node=node, child=child, which=which)

        for child in children:
            self._parse_node(tree, child, depth + 1)

    @abstractmethod
    def _read_leaf(self, tree: ParsableTreeSKL, node: int) -> None:
        msg = "Leaf values should be implemented in a subclass."
        raise NotImplementedError(msg)
