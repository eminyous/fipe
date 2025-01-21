from collections.abc import Callable
from typing import Literal

import gurobipy as gp
import numpy.typing as npt

from .feature import FeatureContainer, FeatureEncoder
from .oracle import Oracle
from .prune import Pruner
from .typing import BaseEnsemble, MNumber, SNumber


class FIPE(Pruner, FeatureContainer):
    PRUNER_NAME_FMT = "{name}_Pruner"
    ORACLE_NAME_FMT = "{name}_Oracle"

    oracle: Oracle | Callable[[MNumber], list[SNumber]]

    _n_oracle_calls: int
    _max_oracle_calls: int
    _oracle_samples: list[list[SNumber]]

    def __init__(
        self,
        base: BaseEnsemble,
        encoder: FeatureEncoder,
        weights: npt.ArrayLike,
        norm: Literal[0, 1] = 1,
        *,
        name: str = "FIPE",
        env: gp.Env | None = None,
        eps: float = Oracle.DEFAULT_EPS,
        tol: float = Oracle.DEFAULT_TOL,
        oracle: Literal["auto"] | Callable[[MNumber], list[SNumber]] = "auto",
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
        if oracle == "auto":
            self.oracle = Oracle(
                base=base,
                encoder=encoder,
                weights=weights,
                name=self.ORACLE_NAME_FMT.format(name=name),
                env=env,
                tol=tol,
                eps=eps,
            )
        else:
            self.oracle = oracle
        self._oracle_samples = []
        self._n_oracle_calls = 0
        self._max_oracle_calls = max_oracle_calls

    def build(self) -> None:
        Pruner.build(self)
        if isinstance(self.oracle, Oracle):
            self.oracle.build()
        self._n_oracle_calls = 0

    def prune(self) -> None:
        while self._n_oracle_calls < self._max_oracle_calls:
            Pruner.prune(self)
            X = self._call_oracle(self.weights)
            if len(X) > 0:
                self._save_oracle_samples(X=X)
                X = self.transform(X=X)
                self.add_samples(X=X)
            else:
                break

    @property
    def n_oracle_calls(self) -> int:
        return self._n_oracle_calls

    @property
    def oracle_samples(self) -> list[list[SNumber]]:
        return self._oracle_samples

    def _save_oracle_samples(self, X: list[SNumber]) -> None:
        self._oracle_samples.append(X)

    def _call_oracle(self, weights: MNumber) -> list[SNumber]:
        self._n_oracle_calls += 1
        return list(self.oracle(weights))
