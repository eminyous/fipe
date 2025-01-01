import gurobipy as gp
import numpy as np
import numpy.typing as npt

from ..feature import FeatureEncoder
from ..mip import MIP
from ..typing import BaseEnsemble, MClass, MNumber
from .base import BasePruner


class Pruner(BasePruner, MIP):
    _n_samples: int
    _norm: int
    _objective: gp.Var
    _weight_vars: gp.tupledict[int, gp.Var]
    _sample_constrs: gp.tupledict[tuple[int, int], gp.Constr]

    def __init__(
        self,
        base: BaseEnsemble,
        encoder: FeatureEncoder,
        weights: npt.ArrayLike,
        norm: int = 1,
        *,
        name: str = "Pruner",
        env: gp.Env | None = None,
    ) -> None:
        BasePruner.__init__(
            self,
            base=base,
            encoder=encoder,
            weights=weights,
        )
        MIP.__init__(self, name=name, env=env)
        self._norm = norm
        self._weight_vars = gp.tupledict()
        self._sample_constrs = gp.tupledict()

    def build(self) -> None:
        self._add_weight_vars()
        self._add_objective()
        self._n_samples = 0

    def add_samples(self, X: npt.ArrayLike) -> None:
        X = np.asarray(X)
        classes = self.ensemble.predict(X=X, w=self._weights)
        prob = self.ensemble.scores(X=X)
        X = np.asarray(X)
        n = X.shape[0]
        for i in range(n):
            self._add_sample(prob=prob[i], class_=classes[i])

    def prune(self) -> None:
        if self._n_samples == 0:
            msg = "No samples have been added to the pruner."
            raise RuntimeError(msg)
        self.optimize()

    def predict(self, X: npt.ArrayLike) -> MClass:
        w = self.weights
        return self.ensemble.predict(X=X, w=w)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def _pruned_weights(self) -> MNumber:
        if self.SolCount == 0:
            return self._weights
        return np.array([
            self._weight_vars[t].X for t in range(self.n_estimators)
        ])

    def _add_sample(self, prob: npt.ArrayLike, class_: int) -> None:
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
                vtype=gp.GRB.CONTINUOUS,
                lb=0.0,
                name=f"weight_{t}",
            )

    def _add_objective(self) -> None:
        self._objective = self.addVar(vtype=gp.GRB.CONTINUOUS, name="norm")
        self.addGenConstrNorm(
            self._objective,
            self._weight_vars,
            self._norm,
        )
        self.setObjective(self._objective, gp.GRB.MINIMIZE)

    def _add_sample_constr(
        self,
        prob: npt.ArrayLike,
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
