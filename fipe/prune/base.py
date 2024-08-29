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
    def activated(self) -> list[int]:
        msg = self.msg.format(name="activated", method="property")
        raise NotImplementedError(msg)

    @property
    def n_activated(self) -> int:
        return len(self.activated)
