import warnings

import numpy as np
import numpy.typing as npt

from .ensemble import Ensemble
from .feature import FeatureContainer, FeatureEncoder
from .oracle import Oracle
from .prune import Pruner
from .typing import ParsableEnsemble, SNumber


class FIPE(Pruner, FeatureContainer):
    oracle: Oracle

    _n_oracle_calls: int
    _max_oracle_calls: int
    _counter_factuals: list[list[SNumber]]

    def __init__(
        self,
        base: ParsableEnsemble,
        weights: npt.ArrayLike,
        encoder: FeatureEncoder,
        norm: int = 1,
        **kwargs,
    ) -> None:
        ensemble = Ensemble(base=base, encoder=encoder)
        Pruner.__init__(
            self,
            ensemble=ensemble,
            weights=weights,
            norm=norm,
            **kwargs,
        )
        FeatureContainer.__init__(self, encoder=encoder)
        self.oracle = Oracle(
            encoder=encoder,
            ensemble=ensemble,
            weights=weights,
            **kwargs,
        )
        self._counter_factuals = []
        self._n_oracle_calls = 0
        self._max_oracle_calls = kwargs.get("max_oracle_calls", 100)

    def build(self) -> None:
        Pruner.build(self)
        self.oracle.build()
        self._n_oracle_calls = 0

    def prune(self) -> None:
        while self._n_oracle_calls < self._max_oracle_calls:
            Pruner.prune(self)
            if self.SolCount == 0:
                msg = "No feasible solution in the pruning model."
                warnings.warn(msg, RuntimeWarning, stacklevel=1)
                break
            X = self._separate(self.weights)
            if len(X) > 0:
                self._save_counter_factuals(X=X)
                X = self.transform(X=X)
                self.add_samples(X=X)
            else:
                break

    @property
    def n_oracle_calls(self) -> int:
        return self._n_oracle_calls

    @property
    def counter_factuals(self) -> list[list[SNumber]]:
        return self._counter_factuals

    def _save_counter_factuals(self, X: list[SNumber]) -> None:
        self._counter_factuals.append(X)

    def _separate(self, weights: npt.ArrayLike) -> list[SNumber]:
        self._n_oracle_calls += 1
        weights = np.asarray(weights)
        return list(self.oracle.separate(new_weights=weights))
