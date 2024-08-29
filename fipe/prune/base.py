from abc import ABC, abstractmethod

from ..ensemble.container import EnsembleContainer
from ..ensemble.ensemble import Ensemble


class BasePruner(ABC, EnsembleContainer):
    msg = "Subclasses must implement the {name} {method}"

    def __init__(self, ensemble: Ensemble, weights):
        EnsembleContainer.__init__(self, ensemble=ensemble, weights=weights)

    @abstractmethod
    def prune(self):
        msg = self.msg.format(name="prune", method="method")
        raise NotImplementedError(msg)

    @property
    @abstractmethod
    def _prune_weights(self) -> dict[int, float]:
        msg = self.msg.format(name="_prune_weights", method="property")
        raise NotImplementedError(msg)

    @property
    def weights(self):
        w = self._prune_weights
        return self._to_array(w)

    @property
    def activated(self) -> set[int]:
        w = self._prune_weights
        return {t for t, v in w.items() if v > 1e-6}

    @property
    def n_activated(self) -> int:
        return len(self.activated)
