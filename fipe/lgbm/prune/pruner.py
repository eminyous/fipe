import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import ArrayLike, NDArray

from ...mip import MIP
from ..ensemble import Ensemble
from .base import BasePruner


class Pruner(BasePruner, MIP):
    _n_samples: int
    _norm: int
    _objective: gp.Var
    _weight_vars: gp.tupledict[int, gp.Var]
    _sample_constrs: gp.tupledict[tuple[int, int], gp.Constr]

    def __init__(
        self,
        ensemble: Ensemble,
        weights: ArrayLike,
        norm: int = 1,
        **kwargs,
    ) -> None:
        BasePruner.__init__(self, ensemble=ensemble, weights=weights)
        MIP.__init__(
            self, name=kwargs.get("name", "Pruner"), env=kwargs.get("env")
        )
        self._norm = norm
        self._weight_vars = gp.tupledict()
        self._sample_constrs = gp.tupledict()

    def build(self) -> None:
        self._add_weight_vars()
        self._add_objective()
        self._n_samples = 0

    def add_samples(
        self,
        X: ArrayLike,
    ) -> None:
        X = np.asarray(X)
        classes = self.ensemble.predict(X=X, w=self._weights)
        prob = self.ensemble.scores(X=X)
        X = np.asarray(X)
        n = X.shape[0]
        for i in range(n):
            self._add_sample(prob=prob[i], class_=classes[i])

    def prune(self) -> None:
        if self._n_samples == 0:
            msg = "Pruner has not been built yet."
            raise RuntimeError(msg)
        self.optimize()

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray[np.float64]:
        w = self.weights
        return self.ensemble.predict(X=X, w=w)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def _pruned_weights(self) -> NDArray[np.float64]:
        if self.SolCount == 0:
            return self._weights
        return np.array(
            [self._weight_vars[t].X for t in range(self.n_estimators)]
        )

    def _add_sample(
        self,
        prob: ArrayLike,
        class_: int,
    ) -> None:
        for c in range(self.ensemble.n_classes):
            if class_ == c:
                continue
            self._add_sample_constr(
                prob=prob,
                true_class=class_,
                class_=c,
            )
        self._n_samples += 1

    def _add_weight_vars(self) -> None:
        for t in range(self.n_estimators):
            self._weight_vars[t] = self.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                name=f"weight_{t}",
            )

    def _add_objective(self) -> None:
        self._objective = self.addVar(vtype=GRB.CONTINUOUS, name="objective")
        self.addGenConstrNorm(
            self._objective,
            self._weight_vars,
            self._norm,
        )

    def _add_sample_constr(
        self,
        prob: ArrayLike,
        true_class: int,
        class_: int,
    ) -> None:
        prob = np.asarray(prob)
        n = self._n_samples
        self._sample_constrs[n, class_] = self.addConstr(
            gp.quicksum(
                (prob[t, true_class] - prob[t, class_]) * self._weight_vars[t]
                for t in range(self.n_estimators)
            )
            >= 1.0,
            name=f"sample_{n}_class_{class_}",
        )
