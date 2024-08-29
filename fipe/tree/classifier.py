import numpy as np

from ..feature.encoder import FeatureEncoder
from ..typing import Node
from .tree import Tree


class TreeClassifier(Tree):
    def __init__(self, tree, encoder: FeatureEncoder):
        Tree.__init__(self, tree, encoder)

    def read_leaf(self, tree, node: Node):
        super().read_leaf(tree, node)
        q = np.argmax(self.value[node])
        self.value[node] = np.eye(self.n_classes)[q]
