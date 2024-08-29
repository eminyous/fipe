import gurobipy as gp
from gurobipy import GRB

from ..ensemble import Ensemble
from ..mip import MIP
from .base import BasePruner


class Pruner(BasePruner, MIP):
    _n_samples: int
    _norm: int
    _objective: gp.Var
    _weight_vars: gp.tupledict[int, gp.Var]
    _sample_constrs: gp.tupledict[tuple[int, int], gp.Constr]

    def __init__(self, ensemble: Ensemble, weights, norm: int, **kwargs):
        BasePruner.__init__(self, ensemble=ensemble, weights=weights)
        MIP.__init__(
            self, name=kwargs.get("name", ""), env=kwargs.get("env", None)
        )
        self._norm = norm
        self._weight_vars = gp.tupledict()
        self._sample_constrs = gp.tupledict()

    def build(self):
        self._add_vars()
        self._add_objective()
        self._n_samples = 0

    def add_samples(self, X):
        w = self._to_array(self._weights)
        y = self.ensemble.predict(X, w)
        p = self.ensemble.scores(X)
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
        return self.ensemble.predict(X, w)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def _prune_weights(self):
        if self.SolCount == 0:
            return self._weights
        return {t: v.X for t, v in self._weight_vars.items()}

    def _add_vars(self):
        for t in range(self.n_estimators):
            self._weight_vars[t] = self.addVar(
                vtype=GRB.CONTINUOUS, lb=0.0, name=f"weight_{t}"
            )

    def _add_objective(self):
        self._objective = self.addVar(name="objective")
        self.addGenConstrNorm(self._objective, self._weight_vars, self._norm)
        self.setObjective(self._objective, GRB.MINIMIZE)

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
