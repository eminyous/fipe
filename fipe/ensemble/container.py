from abc import ABCMeta

import numpy as np

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
    _weights: dict[int, float]

    def __init__(self, ensemble: Ensemble, weights):
        self._ensemble = ensemble
        self._weights = self._to_dict(weights)

    @property
    def ensemble(self):
        return self._ensemble

    @property
    def n_estimators(self):
        return self.ensemble.n_estimators

    @property
    def n_classes(self):
        return self.ensemble.n_classes

    def _to_dict(self, w):
        wd = {}
        for t in range(self.n_estimators):
            try:
                wd[t] = w[t]
            except (KeyError, IndexError):
                wd[t] = 0.0
        return wd

    def _to_array(self, w):
        wa = np.zeros(self.n_estimators)
        for t in range(self.n_estimators):
            try:
                wa[t] = w[t]
            except (KeyError, IndexError):
                wa[t] = 0.0
        return wa
