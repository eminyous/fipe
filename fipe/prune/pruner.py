import re
from pathlib import Path
from typing import Literal

import gurobipy as gp
import numpy as np
import numpy.typing as npt
import pyscipopt as scip

from ..env import ENV, PrunerSolver
from ..feature import FeatureEncoder
from ..mip import MIP
from ..typing import BaseEnsemble, MNumber, MProb, Number
from .base import BasePruner


class Pruner(BasePruner, MIP):
    WEIGHT_VARS_NAME = "weights"
    OBJECTIVE_NAME = "norm"
    SAMPLE_CONSTR_NAME_FMT = "sample_{n}"
    VALID_NOMRS: tuple[int, ...] = (0, 1)

    # Cache
    CACHE = Path(".fipe_cache")
    MPS = CACHE / Path("pruner.mps")

    _n_samples: int
    _norm: int
    _objective: gp.Var
    _weight_vars: gp.MVar
    _sample_constrs: gp.tupledict[int, gp.MConstr]

    __weights: MNumber

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
        self._n_samples = 0

    def add_samples(self, X: npt.ArrayLike) -> None:
        X = np.asarray(X)
        w = self._weights
        classes = self.ensemble.predict(X=X, w=w)
        prob = self.ensemble.predict_proba(X=X)
        n = X.shape[0]
        for i in range(n):
            self._add_sample(prob=prob[i], class_=classes[i])

    def prune(self) -> None:
        if self._n_samples == 0:
            msg = "No samples have been added to the pruner."
            raise RuntimeError(msg)
        self._prune_l1()
        if self._norm == 0:
            self._prune_l0()

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def _pruner_weights(self) -> MNumber:
        return self.__weights

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

    def _validate_norm(self, norm: int) -> None:
        if norm not in self.VALID_NOMRS:
            msg = "The norm must be either 0 or 1."
            raise ValueError(msg)

    def _prune_l1(self) -> None:
        w = self._weight_vars
        self.setObjective(w.sum(), gp.GRB.MINIMIZE)
        self._optimize()

    def _prune_l0(self) -> None:
        W = Number(np.sum(self.__weights))
        n = self.n_estimators
        w = self._weight_vars
        u = self.addMVar(shape=n, vtype=gp.GRB.BINARY, name="u")
        contrs = self.addConstr(w <= W * u, name="bigM")
        self.setObjective(u.sum(), gp.GRB.MINIMIZE)
        self._optimize()
        self.remove(contrs)
        self.remove(u)

    def _optimize(self) -> None:
        if ENV.pruner_solver == PrunerSolver.GUROBI:
            self._optimize_gurobi()
        elif ENV.pruner_solver == PrunerSolver.SCIP:
            self._optimize_scip()
        else:
            msg = "The pruner solver is not supported."
            raise ValueError(msg)

    def _optimize_gurobi(self) -> None:
        self.optimize()
        if self.SolCount == 0:
            self.__weights = np.array(self._weights)
        else:
            self.__weights = np.array(self._weight_vars.X)

    def _optimize_scip(self) -> None:
        self.CACHE.mkdir(exist_ok=True)
        self.write(str(self.MPS))
        model = scip.Model()
        model.readProblem(str(self.MPS))
        model.optimize()
        self.__weights = self._get_weights_scip(model=model)
        self.MPS.unlink()
        self.CACHE.rmdir()

    def _get_weights_scip(self, model: scip.Model) -> MNumber:
        solution = model.getBestSol()
        pattern = rf"{self.WEIGHT_VARS_NAME}\[(\d+)\]"
        values = np.zeros(self.n_estimators)
        variables = model.getVars()
        for var in variables:
            matcher = re.match(pattern, var.name)
            if matcher:
                i = int(matcher.group(1))
                val = model.getSolVal(solution, var)
                values[i] = val
        return values
