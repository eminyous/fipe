from abc import ABCMeta

import numpy as np
import numpy.typing as npt

from ..feature import FeatureEncoder
from ..typing import BaseEnsemble, MNumber
from .ensemble import Ensemble


class EnsembleContainer:
    __metaclass__ = ABCMeta

    _ensemble: Ensemble
    _weights: MNumber

    def __init__(
        self,
        *,
        ensemble: Ensemble | tuple[BaseEnsemble, FeatureEncoder],
        weights: npt.ArrayLike,
    ) -> None:
        if isinstance(ensemble, tuple):
            base, encoder = ensemble
            ensemble = Ensemble(base=base, encoder=encoder)

        self._ensemble = ensemble
        self._weights = np.asarray(weights)

    @property
    def ensemble(self) -> Ensemble:
        return self._ensemble

    @property
    def is_binary(self) -> bool:
        return self.ensemble.is_binary

    @property
    def n_estimators(self) -> int:
        return self.ensemble.n_estimators

    @property
    def n_classes(self) -> int:
        return self.ensemble.n_classes
