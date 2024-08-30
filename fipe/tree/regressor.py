import numpy as np

from ..feature.encoder import FeatureEncoder
from ..typing import BaseTree, Node
from .tree import Tree


class TreeRegressor(Tree):
    def __init__(self, tree: BaseTree, encoder: FeatureEncoder) -> None:
        Tree.__init__(self, tree, encoder)

    def read_leaf(self, tree: BaseTree, node: Node) -> None:
        # We assume for now that the value
        # is a scalar in the regression case
        v = tree.value[node].flatten()[0]
        self.value[node] = np.array([-v, v])
        if self.n_classes == -1:
            self.n_classes = 2
