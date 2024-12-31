import gurobipy as gp
import numpy.typing as npt

from ..ensemble import Ensemble
from ..feature import FeatureEncoder
from .base import BaseOCEAN


class VoteOCEAN(BaseOCEAN):
    _eps: float
    _majority_class_constrs: gp.tupledict[int, gp.Constr]

    def __init__(
        self,
        encoder: FeatureEncoder,
        ensemble: Ensemble,
        weights: npt.ArrayLike,
        **kwargs,
    ) -> None:
        BaseOCEAN.__init__(
            self,
            encoder=encoder,
            ensemble=ensemble,
            weights=weights,
            **kwargs,
        )
        self._eps = kwargs.get("eps", 1e-6)
        self._majority_class_constrs = gp.tupledict()

    def set_majority_class(self, class_: int) -> None:
        self._majority_class_constrs = gp.tupledict()
        for c in range(self.n_classes):
            if c == class_:
                continue
            self._add_majority_class_constr(majority_class=class_, class_=c)

    def clear_majority_class(self) -> None:
        self.remove(self._majority_class_constrs)
        self._majority_class_constrs = gp.tupledict()

    def _add_majority_class_constr(
        self,
        majority_class: int,
        class_: int,
    ) -> None:
        rhs = self._eps if majority_class > class_ else 0.0
        constr = self.addConstr(
            self.function(class_=majority_class)
            >= self.function(class_=class_) + rhs,
            name=f"majority_class_constr_{majority_class}_class_{class_}",
        )
        self._majority_class_constrs[class_] = constr
