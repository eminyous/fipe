from abc import ABC, abstractmethod

from numpy.typing import ArrayLike

from ..ensemble.container import EnsembleContainer
from ..ensemble.ensemble import Ensemble
from ..typing import Weights, numeric


class BasePruner(ABC, EnsembleContainer):
    msg = "Subclasses must implement the {name} {method}"

    def __init__(self, ensemble: Ensemble, weights: Weights) -> None:
        EnsembleContainer.__init__(self, ensemble=ensemble, weights=weights)

    @abstractmethod
    def prune(self) -> None:
        msg = self.msg.format(name="prune", method="method")
        raise NotImplementedError(msg)

    @property
    @abstractmethod
    def _prune_weights(self) -> dict[int, numeric]:
        msg = self.msg.format(name="_prune_weights", method="property")
        raise NotImplementedError(msg)

    @property
    def weights(self) -> ArrayLike:
        w = self._prune_weights
        return self._to_array(w)

    @property
    def activated(self) -> set[int]:
        w = self._prune_weights
        THRESHOLD = 1e-6
        return {t for t, v in w.items() if v > THRESHOLD}

    @property
    def n_activated(self) -> int:
        return len(self.activated)
