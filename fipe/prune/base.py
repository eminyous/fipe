from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.typing as npt

from ..ensemble import EnsembleContainer
from ..feature import FeatureEncoder
from ..typing import BaseEnsemble, MNumber


class BasePruner(EnsembleContainer):
    __metaclass__ = ABCMeta

    def __init__(
        self,
        base: BaseEnsemble,
        encoder: FeatureEncoder,
        weights: npt.ArrayLike,
    ) -> None:
        super().__init__(ensemble=(base, encoder), weights=weights)

    @abstractmethod
    def prune(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def _pruned_weights(self) -> MNumber:
        raise NotImplementedError

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
