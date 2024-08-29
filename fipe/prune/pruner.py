import gurobipy as gp
import numpy as np
from gurobipy import GRB

from ..ensemble import Ensemble
from ..mip import MIP
from .base import BasePruner


class Pruner(BasePruner, MIP):
    _weight_vars: gp.tupledict[int, gp.Var]
    _n_samples: int
    _sample_constrs: gp.tupledict[tuple[int, int], gp.Constr]

    def __init__(self, ensemble: Ensemble, weights, **kwargs):
        BasePruner.__init__(self, ensemble=ensemble, weights=weights)
        MIP.__init__(
            self, name=kwargs.get("name", ""), env=kwargs.get("env", None)
        )
        self._weight_vars = gp.tupledict()
        self._sample_constrs = gp.tupledict()

    def build(self):
        self._add_vars()
        self._n_samples = 0

    def add_samples(self, X):
        w = np.array([self._weights[t] for t in range(self.n_estimators)])
        y = self._ensemble.predict(X, w)
        p = self._ensemble.scores(X)
        n = X.shape[0]
        for i in range(n):
            self._add_sample_constrs(p[i], y[i])

    def prune(self):
        if self._n_samples == 0:
            msg = "No samples was added to the pruner."
            raise ValueError(msg)
        self.optimize()

    def predict(self, X):
        w = self.weights
        w = np.array([w[t] for t in range(self.n_estimators)])
        return self._ensemble.predict(X, w)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def weights(self) -> dict[int, float]:
        if self.SolCount == 0:
            return {t: self._weights[t] for t in range(self.n_estimators)}
        return {t: self._weight_vars[t].X for t in range(self.n_estimators)}

    @property
    def activated(self) -> list[int]:
        if self.SolCount == 0:
            return list(range(self.n_estimators))
        return [
            t for t in range(self.n_estimators) if self._weight_vars[t].X > 1e-6
        ]

    def _add_vars(self):
        for t in range(self.n_estimators):
            self._weight_vars[t] = self.addVar(
                vtype=GRB.CONTINUOUS, lb=0.0, name=f"weight_{t}"
            )

    def _add_sample_constrs(self, p, y: int):
        for c in range(self.n_classes):
            if c == y:
                continue
            self._add_sample_constr(p, y, c)
        self._n_samples += 1

    def _add_sample_constr(self, p, y: int, c: int):
        i = self._n_samples
        self._sample_constrs[i, c] = self.addConstr(
            gp.quicksum(
                self._weight_vars[t] * (p[t, y] - p[t, c])
                for t in range(self.n_estimators)
            )
            >= 1.0,
            name=f"sample_{i}_{c}",
        )

    def set_norm(self, norm: int = 0):
        obj = self.addVar(name="objective")
        self.addGenConstrNorm(obj, self._weight_vars, norm)
        self.setObjective(obj, GRB.MINIMIZE)
