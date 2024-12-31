from ...typing import Number, ParsableTreeSKL
from .skl import TreeSKL


class TreeGB(TreeSKL[Number]):
    def _read_leaf(self, tree: ParsableTreeSKL, node: int) -> None:
        value = tree.value[node].flatten()[0]
        self.leaf_value[node] = float(value)
