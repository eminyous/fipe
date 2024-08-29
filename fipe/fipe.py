import warnings
from copy import deepcopy

from .ensemble import Ensemble
from .feature import FeatureContainer, FeatureEncoder
from .oracle import Oracle
from .prune import Pruner
from .typing import BaseEnsemble, Sample


class FIPE(Pruner, FeatureContainer):
    """Functionally-Identical Pruning of Ensemble (FIPE) algorithm."""

    oracle: Oracle
    _n_oracle_calls: int
    _max_oracle_calls: int
    _counterfactuals: list[list[Sample]]
    _history: list[dict[int, float]]

    def __init__(
        self, base: BaseEnsemble, weights, encoder: FeatureEncoder, **kwargs
    ):
        """Initialize parent classes and oracle component."""
        ensemble = Ensemble(base=base, encoder=encoder)
        Pruner.__init__(self, ensemble=ensemble, weights=weights, **kwargs)
        FeatureContainer.__init__(self, encoder=encoder)
        self.oracle = Oracle(
            encoder=encoder, ensemble=ensemble, weights=weights, **kwargs
        )
        # Initialize attributes
        self._counterfactuals = []
        self._history = []
        self._max_oracle_calls = kwargs.get("max_oracle_calls", 100)

    def build(self):
        """Build combinatorial optimization models."""
        Pruner.build(self)
        self.oracle.build()
        self._n_oracle_calls = 0

    def prune(self):
        """Iteratively prune the ensemble and call the oracle to separate."""
        while self._n_oracle_calls < self._max_oracle_calls:
            # Solve pruning problem
            self.optimize()
            if self.SolCount == 0:
                # No solution found.
                msg = "No solution found in the pruning model."
                warnings.warn(msg)
                break
            self._save_weights()

            # Solve separation problem
            X = self._separate(self.weights)
            if len(X) > 0:
                self._save_counterfactuals(X)
                X = self.transform(X)
                self.add_samples(X)
            else:
                # No counterfactuals found.
                # The pruning is complete.
                break

    @property
    def n_oracle_calls(self) -> int:
        """Return the number of times the oracle was called."""
        return self._n_oracle_calls

    @property
    def counterfactuals(self) -> list[list[Sample]]:
        """Return the counterfactual examples found during pruning."""
        return self._counterfactuals

    def _save_weights(self):
        weights = deepcopy(self.weights)
        self._history.append(weights)

    def _save_counterfactuals(self, counters: list[Sample]):
        self._counterfactuals.append(counters)

    def _separate(self, weights):
        """Call the oracle separation problem for th egiven weights."""
        self._n_oracle_calls += 1
        return list(self.oracle.separate(weights))
