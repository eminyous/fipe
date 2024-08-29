import gurobipy as gp

from ..ensemble import Ensemble
from ..feature.encoder import FeatureEncoder
from .base import BaseOCEAN


class VoteOCEAN(BaseOCEAN):
    _eps: float
    _majority_class_constrs: gp.tupledict[int, gp.Constr]

    def __init__(
        self, encoder: FeatureEncoder, ensemble: Ensemble, weights, **kwargs
    ):
        BaseOCEAN.__init__(self, encoder, ensemble, weights, **kwargs)
        self._eps = kwargs.get("eps", 1.0)

    def set_majority_class(self, c: int):
        self._majority_class_constrs = gp.tupledict()
        for k in range(self.n_classes):
            if k == c:
                continue
            self._add_majority_class_constr(c, k)

    def clear_majority_class(self):
        self.remove(self._majority_class_constrs)
        self._majority_class_constrs = gp.tupledict()

    def _add_majority_class_constr(self, c: int, k: int):
        rhs = self._eps if k < c else 0.0

        constr = self.addConstr(
            gp.quicksum(
                self._weights[t] * self._flow_vars[t].value[c]
                for t in range(self.n_estimators)
            )
            >= gp.quicksum(
                self._weights[t] * self._flow_vars[t].value[k]
                for t in range(self.n_estimators)
            )
            + rhs,
            name=f"majority_class_{c}_{k}",
        )
        self._majority_class_constrs[k] = constr
