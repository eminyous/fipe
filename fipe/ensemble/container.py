from abc import ABCMeta

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
        self._weights = {}
        for t, w in enumerate(weights):
            self._weights[t] = w

    @property
    def ensemble(self):
        return self._ensemble

    @property
    def n_estimators(self):
        return self.ensemble.n_estimators

    @property
    def n_classes(self):
        return self.ensemble.n_classes
