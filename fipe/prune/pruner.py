from typing import Literal

import gurobipy as gp
import numpy as np
import numpy.typing as npt

from ..feature import FeatureEncoder
from ..mip import MIP
from ..typing import BaseEnsemble, MNumber, MProb
from .base import BasePruner


class Pruner(BasePruner, MIP):
    WEIGHT_VARS_NAME = "weights"
    OBJECTIVE_NAME = "norm"
    SAMPLE_CONSTR_NAME_FMT = "sample_{n}"
    VALID_NOMRS: tuple[int, ...] = (0, 1)

    _n_samples: int
    _norm: int
    _objective: gp.Var
    _weight_vars: gp.MVar
    _sample_constrs: gp.tupledict[int, gp.MConstr]

    def __init__(
        self,
        base: BaseEnsemble,
        encoder: FeatureEncoder,
        weights: npt.ArrayLike,
        norm: Literal[0, 1] = 1,
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
        self._validate_norm(norm=norm)
        self._norm = norm
        self._sample_constrs = gp.tupledict()

    def build(self) -> None:
        self._add_weight_vars()
        self._add_objective()
        self._n_samples = 0

    def add_samples(self, X: npt.ArrayLike) -> None:
        X = np.asarray(X)
        classes = self.ensemble.predict(X=X, w=self._weights)
        prob = self.ensemble.predict_proba(X=X)
        n = X.shape[0]
        for i in range(n):
            self._add_sample(prob=prob[i], class_=classes[i])

    def prune(self) -> None:
        if self._n_samples == 0:
            msg = "No samples have been added to the pruner."
            raise RuntimeError(msg)
        self.optimize()

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def _pruner_weights(self) -> MNumber:
        if self.SolCount == 0:
            return self._weights
        return np.array(self._weight_vars.X)

    def _add_sample(self, prob: MProb, class_: int) -> None:
        true_prob = prob[:, [class_]]
        prob = np.delete(arr=prob, obj=class_, axis=1)
        name = self.SAMPLE_CONSTR_NAME_FMT.format(n=self._n_samples)
        constr = self.addMConstr(
            A=(true_prob - prob).T,
            x=self._weight_vars,
            sense=gp.GRB.GREATER_EQUAL,
            b=np.ones(self.n_classes - 1),
            name=name,
        )
        self._sample_constrs[self._n_samples] = constr
        self._n_samples += 1

    def _add_weight_vars(self) -> None:
        self._weight_vars = self.addMVar(
            shape=self.n_estimators,
            lb=0.0,
            name=self.WEIGHT_VARS_NAME,
        )

    def _add_objective(self) -> None:
        self._objective = self.addVar(
            vtype=gp.GRB.CONTINUOUS,
            name=self.OBJECTIVE_NAME,
        )
        self.addGenConstrNorm(
            resvar=self._objective,
            vars=self._weight_vars,
            which=self._norm,
            name=self.OBJECTIVE_NAME,
        )
        self.setObjective(self._objective, gp.GRB.MINIMIZE)

    def _validate_norm(self, norm: int) -> None:
        if norm not in self.VALID_NOMRS:
            msg = "The norm must be either 0 or 1."
            raise ValueError(msg)
