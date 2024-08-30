from abc import ABCMeta

import numpy as np
from numpy.typing import ArrayLike

from ..typing import Weights, numeric
from .ensemble import Ensemble


class EnsembleContainer:
    """
    Abstract class for ensemble containers.

    This class is a wrapper around the Ensemble class
    and provide a way to access the ensemble and the
    weights of the ensemble.
    """

    __metaclass__ = ABCMeta

    _ensemble: Ensemble
    _weights: dict[int, numeric]

    def __init__(
        self,
        ensemble: Ensemble,
        weights: Weights,
    ) -> None:
        self._ensemble = ensemble
        self._weights = self._to_dict(weights)

    @property
    def ensemble(self) -> Ensemble:
        return self._ensemble

    @property
    def n_estimators(self) -> int:
        return self.ensemble.n_estimators

    @property
    def n_classes(self) -> int:
        return self.ensemble.n_classes

    def _to_dict(self, w: Weights) -> dict[int, numeric]:
        if isinstance(w, dict):
            return self._to_dict_from_dict(w)
        return self._to_dict_from_array(w)

    def _to_dict_from_dict(self, w: dict[int, numeric]) -> dict[int, numeric]:
        wd = {}
        for t in range(self.n_estimators):
            wd[t] = w.get(t, 0.0)
        return wd

    def _to_dict_from_array(self, w: ArrayLike) -> dict[int, numeric]:
        w = np.array(w)
        return {t: w[t] for t in range(self.n_estimators)}

    def _to_array(self, w: Weights) -> np.ndarray:
        if isinstance(w, dict):
            return self._to_array_from_dict(w)
        return self._to_array_from_array(w)

    def _to_array_from_dict(self, w: dict[int, numeric]) -> np.ndarray:
        return np.array([w.get(t, 0.0) for t in range(self.n_estimators)])

    @staticmethod
    def _to_array_from_array(w: ArrayLike) -> np.ndarray:
        return np.array(w)
