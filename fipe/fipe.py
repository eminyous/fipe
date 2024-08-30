import warnings

from .ensemble import Ensemble
from .feature import FeatureContainer, FeatureEncoder
from .oracle import Oracle
from .prune import Pruner
from .typing import BaseEnsemble, Sample, Weights


class FIPE(Pruner, FeatureContainer):
    """Functionally-Identical Pruning of Ensemble (FIPE) algorithm."""

    oracle: Oracle

    _n_oracle_calls: int
    _max_oracle_calls: int
    _counterfactuals: list[list[Sample]]

    def __init__(
        self,
        base: BaseEnsemble,
        weights: Weights,
        encoder: FeatureEncoder,
        norm: int,
        **kwargs,
    ) -> None:
        """Initialize parent classes and oracle component."""
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
        # Initialize attributes
        self._counterfactuals = []
        self._max_oracle_calls = kwargs.get("max_oracle_calls", 100)

    def build(self) -> None:
        """Build combinatorial optimization models."""
        Pruner.build(self)
        self.oracle.build()
        self._n_oracle_calls = 0

    def prune(self) -> None:
        while self._n_oracle_calls < self._max_oracle_calls:
            # Solve pruning problem
            Pruner.prune(self)
            if self.SolCount == 0:
                # No solution found.
                msg = "No solution found in the pruning model."
                warnings.warn(msg, RuntimeWarning, stacklevel=1)
                break

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
        return self._n_oracle_calls

    @property
    def counterfactuals(self) -> list[list[Sample]]:
        return self._counterfactuals

    def _save_counterfactuals(self, counters: list[Sample]) -> None:
        self._counterfactuals.append(counters)

    def _separate(self, weights: Weights) -> list[Sample]:
        self._n_oracle_calls += 1
        return list(self.oracle.separate(weights))
