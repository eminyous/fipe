from ...typing import Number, ParsableTreeSKL
from .skl import TreeSKL


class TreeGB(TreeSKL[Number]):
    def _read_leaf(self, tree: ParsableTreeSKL, node: int) -> None:
        value = float(tree.value[node].flatten()[0])
        self._set_leaf(node=node, value=value)
