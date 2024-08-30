from collections.abc import Callable, Generator

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .ensemble import Ensemble
from .feature import FeatureEncoder
from .ocean import OCEAN
from .typing import Sample, Weights


class Oracle(OCEAN):
    def __init__(
        self,
        encoder: FeatureEncoder,
        ensemble: Ensemble,
        weights: Weights,
        **kwargs,
    ) -> None:
        """Initialize oracle mip from parent class."""
        OCEAN.__init__(
            self,
            encoder=encoder,
            ensemble=ensemble,
            weights=weights,
            **kwargs,
        )

    def separate(self, new_weights: Weights) -> Generator[Sample, None, None]:
        self._set_new_weights(new_weights)
        for c in range(self.n_classes):
            yield from self._run_on_single_class(c)

    def _get_counterfactuals(
        self,
        c: int,
        k: int,
    ) -> Generator[Sample, None, None]:
        param = GRB.Param.SolutionNumber
        for i in range(self.SolCount):
            self.setParam(param, i)
            x = self._feature_vars.Xn

            if self.PoolObjVal < 0.0:
                continue

            if self.PoolObjVal == 0.0 and c < k:
                continue

            # This is to avoid the same prediction
            X = self.transform(x)
            # Read weights
            w = np.array([self._weights[t] for t in range(self.n_estimators)])
            new_w = np.array(
                [self._new_weights[t] for t in range(self.n_estimators)],
            )
            # Predict class according to two ensembles
            pred = self._ensemble.predict(X, w)
            new_pred = self._ensemble.predict(X, new_w)
            if np.all(pred == new_pred):
                continue
            yield x
        self.setParam(param, 0)

    def _run_on_single_class(self, c: int) -> Generator[Sample, None, None]:
        # Set as constraint that original mip
        # classifies as c
        self.set_majority_class(c)
        # For all other classes,
        # maximize misclassification
        for k in range(self.n_classes):
            if c == k:
                continue
            self._run_on_pair_of_classes(c, k)
            yield from self._get_counterfactuals(c, k)
        # Remove constraint that original
        # model classifies as c
        self.clear_majority_class()

    def _run_on_pair_of_classes(self, c1: int, c2: int) -> None:
        # Adapt objective: maximize the difference
        # between the two classes
        obj = gp.quicksum(
            self._new_weights[t] * self._flow_vars[t].value[c2]
            for t in range(self.n_estimators)
        ) - gp.quicksum(
            self._new_weights[t] * self._flow_vars[t].value[c1]
            for t in range(self.n_estimators)
        )
        self.setObjective(obj, sense=gp.GRB.MAXIMIZE)
        # Solve with early termination callback
        termination_callback = self._get_termination_callback()
        self.optimize(termination_callback)

    @staticmethod
    def _get_termination_callback() -> Callable[[gp.Model, int], None]:
        def callback(model: gp.Model, where: int) -> None:
            if where == gp.GRB.Callback.MIPSOL:
                val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
                if val <= 0.0:
                    model.terminate()

        return callback
