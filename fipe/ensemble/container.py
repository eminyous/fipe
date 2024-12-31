from abc import ABCMeta

import numpy as np
import numpy.typing as npt

from ..typing import MNumber
from .ensemble import Ensemble


class EnsembleContainer:
    __metaclass__ = ABCMeta

    _ensemble: Ensemble
    _weights: MNumber

    def __init__(self, ensemble: Ensemble, weights: npt.ArrayLike) -> None:
        self._ensemble = ensemble
        self._weights = np.asarray(weights)

    @property
    def ensemble(self) -> Ensemble:
        return self._ensemble

    @property
    def n_estimators(self) -> int:
        return self._ensemble.n_estimators

    @property
    def n_classes(self) -> int:
        return self._ensemble.n_classes
