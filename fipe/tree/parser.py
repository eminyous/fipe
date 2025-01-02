from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

from ..feature import FeatureContainer, FeatureEncoder
from ..typing import LeafValue, ParsableNode, ParsableTree
from .tree import Tree

PT = TypeVar("PT", bound=ParsableTree)
NT = TypeVar("NT", bound=ParsableNode)


class GenericTreeParser(FeatureContainer, Generic[PT, NT]):
    __metaclass__ = ABCMeta

    DEFAULT_ROOT_ID = 0
    leaf_offset: int

    __base: PT | None = None

    def __init__(self, encoder: FeatureEncoder) -> None:
        super().__init__(encoder=encoder)
        self.leaf_offset = 0

    @property
    def base(self) -> PT:
        if self.__base is None:
            msg = "Base not set. Call parse method first."
            raise ValueError(msg)
        return self.__base

    def init_tree(self) -> Tree:
        return Tree(encoder=self.encoder)

    @staticmethod
    def parse_root_id() -> int:
        return int(GenericTreeParser.DEFAULT_ROOT_ID)

    def parse(self, base: PT) -> Tree:
        self.__base = base
        self.leaf_offset = 0

        tree = self.init_tree()

        root_id = self.parse_root_id()
        n_nodes = self.parse_n_nodes()
        tree.n_leaves = (n_nodes + 1) // 2
        tree.leaf_offset = self.leaf_offset
        tree.root_id = root_id

        root = self.parse_root()
        self.parse_node(tree, node_id=root_id, node=root, depth=0)

        self.__base = None
        self.leaf_offset = 0
        return tree

    def parse_node(
        self,
        tree: Tree,
        node_id: int,
        node: NT,
        depth: int,
    ) -> None:
        tree.node_depth[node_id] = depth
        if self.is_leaf(node=node):
            value = self.get_leaf_value(node=node)
            tree.add_leaf(node=node_id, value=value)
        else:
            column_index, threshold = self.get_internal_node(node=node)
            tree.add_internal_node(
                node=node_id,
                column_index=column_index,
                threshold=threshold,
            )
            children = self.get_children(node=node)
            self.parse_children(
                tree=tree,
                node_id=node_id,
                children=children,
                depth=depth,
            )

    def parse_children(
        self,
        tree: Tree,
        node_id: int,
        children: tuple[NT, NT],
        depth: int,
    ) -> None:
        whichs = (tree.LEFT_NAME, tree.RIGHT_NAME)
        for which, child in zip(whichs, children, strict=True):
            child_id = self.read_node_id(node=child)
            tree.add_child(node=node_id, child=child_id, which=which)
            self.parse_node(
                tree=tree,
                node_id=child_id,
                node=child,
                depth=depth + 1,
            )

    @abstractmethod
    def parse_n_nodes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def parse_root(self) -> NT:
        raise NotImplementedError

    @abstractmethod
    def is_leaf(self, node: NT) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_leaf_value(self, node: NT) -> LeafValue:
        raise NotImplementedError

    @abstractmethod
    def get_internal_node(self, node: NT) -> tuple[int, float]:
        raise NotImplementedError

    @abstractmethod
    def get_children(self, node: NT) -> tuple[NT, NT]:
        raise NotImplementedError

    @abstractmethod
    def read_node_id(self, node: NT) -> int:
        raise NotImplementedError
