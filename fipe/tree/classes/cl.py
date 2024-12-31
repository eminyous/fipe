from typing import Generic

import numpy as np

from ...feature import FeatureEncoder
from ...typing import HV, MNumber, ParsableTreeSKL
from .skl import TreeSKL


class TreeCL(TreeSKL[MNumber], Generic[HV]):
    __voting__: HV

    def __init__(
        self,
        tree: ParsableTreeSKL,
        encoder: FeatureEncoder,
        voting: HV,
    ) -> None:
        self.__voting__ = voting
        super().__init__(tree=tree, encoder=encoder)

    def _read_leaf(self, tree: ParsableTreeSKL, node: int) -> None:
        value = tree.value[node].flatten()
        value = np.asarray(value)
        if self.__voting__:
            k = value.size
            q = np.argmax(value)
            value = np.eye(k)[q]
        self._set_leaf(node=node, value=value)
