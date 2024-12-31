import numpy as np

from ..typing import MNumber
from .vote import VoteOCEAN


class OCEAN(VoteOCEAN):
    _new_weights: MNumber

    @property
    def new_weights(self) -> MNumber:
        return self._new_weights

    @new_weights.setter
    def new_weights(self, new_weights: MNumber) -> None:
        self._new_weights = np.copy(new_weights)
