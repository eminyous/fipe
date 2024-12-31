import warnings

import gurobipy as gp
import numpy as np
import numpy.typing as npt

from .feature import FeatureContainer, FeatureEncoder
from .oracle import Oracle
from .prune import Pruner
from .typing import BaseEnsemble, SNumber


class FIPE(Pruner, FeatureContainer):
    PRUNER_NAME_FMT = "{name}_Pruner"
    ORACLE_NAME_FMT = "{name}_Oracle"

    oracle: Oracle

    _n_oracle_calls: int
    _max_oracle_calls: int
    _counter_factuals: list[list[SNumber]]

    def __init__(
        self,
        base: BaseEnsemble,
        encoder: FeatureEncoder,
        weights: npt.ArrayLike,
        norm: int = 1,
        *,
        name: str = "FIPE",
        env: gp.Env | None = None,
        eps: float = Oracle.DEFAULT_EPS,
        tol: float = Oracle.DEFAULT_TOL,
        max_oracle_calls: int = 100,
    ) -> None:
        Pruner.__init__(
            self,
            base=base,
            encoder=encoder,
            weights=weights,
            norm=norm,
            name=self.PRUNER_NAME_FMT.format(name=name),
            env=env,
        )
        FeatureContainer.__init__(self, encoder=encoder)
        self.oracle = Oracle(
            base=base,
            encoder=encoder,
            weights=weights,
            name=self.ORACLE_NAME_FMT.format(name=name),
            env=env,
            tol=tol,
            eps=eps,
        )
        self._counter_factuals = []
        self._n_oracle_calls = 0
        self._max_oracle_calls = max_oracle_calls

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
