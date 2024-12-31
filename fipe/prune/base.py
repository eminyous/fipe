from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from ..ensemble import Ensemble, EnsembleContainer
from ..typing import MNumber, Number


class BasePruner(ABC, EnsembleContainer):
    def __init__(self, ensemble: Ensemble, weights: npt.ArrayLike) -> None:
        EnsembleContainer.__init__(self, ensemble=ensemble, weights=weights)

    @abstractmethod
    def prune(self, threshold: Number) -> None:
        msg = "Method 'prune' must be implemented in a child class"
        raise NotImplementedError(msg)

    @property
    @abstractmethod
    def _pruned_weights(self) -> MNumber:
        msg = "Property '_pruned_weights' must be implemented in a child class"
        raise NotImplementedError(msg)

    @property
    def weights(self) -> MNumber:
        return self._pruned_weights

    @property
    def active_estimators(self) -> set[int]:
        THRESHOLD = 1e-6
        return set(np.where(self.weights > THRESHOLD)[0])

    @property
    def n_active_estimators(self) -> int:
        return len(self.active_estimators)
