import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .ensemble import Ensemble
from .feature import FeatureEncoder
from .ocean import OCEAN


class Oracle(OCEAN):
    def __init__(
        self, encoder: FeatureEncoder, ensemble: Ensemble, weights, **kwargs
    ):
        """Initialize oracle mip from parent class."""
        OCEAN.__init__(
            self,
            encoder=encoder,
            ensemble=ensemble,
            weights=weights,
            **kwargs,
        )

    def separate(self, new_weights):
        """Run the separation mip using the given weights.

        The separation mip returns a list of counterfactual examples,
        i.e., points that have different class accordingo to the original
        ensemble and the pruned ensmble.
        When the separation mip returns an empty list, the pruning
        algorithm has converged to a functionally-identical model.
        """
        self._set_pruned_model_weights(new_weights)
        for c in range(self.n_classes):
            yield from self._run_on_single_class(c)

    def _get_counterfactuals(self, c, k, check: bool = True):
        param = GRB.Param.SolutionNumber
        for i in range(self.SolCount):
            self.setParam(param, i)
            x = self._feature_vars.Xn
            if check:
                self._check_counter_factual(x)

            if self.PoolObjVal < 0.0:
                continue

            if self.PoolObjVal == 0.0 and c < k:
                continue

            # This is to avoid the same prediction
            X = self.transform(x)
            # Read weights
            w = np.array([self._weights[t] for t in range(self.n_estimators)])
            new_w = np.array(
                [self._new_weights[t] for t in range(self.n_estimators)]
            )
            # Predict class according to two ensembles
            pred = self._ensemble.predict(X, w)
            new_pred = self._ensemble.predict(X, new_w)
            if np.all(pred == new_pred):
                continue
            yield x
        self.setParam(param, 0)

    def _run_on_single_class(self, c: int):
        """Run the separation mip for a single target class single class."""
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

    def _run_on_pair_of_classes(self, c1: int, c2: int):
        """
        Separation mip tries to classifiy as class c2
        whereas the original model classifies as c1.
        """
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

    def _get_termination_callback(self):
        """
        Interrupt solving process if the maximum misclassification
        score is less than zero.
        """

        def callback(model: gp.Model, where: int):
            if where == gp.GRB.Callback.MIPSOL:
                val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
                if val <= 0.0:
                    model.terminate()

        return callback
