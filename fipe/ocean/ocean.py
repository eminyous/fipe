import gurobipy as gp
import numpy as np
import numpy.typing as npt

from ..feature import FeatureEncoder
from ..typing import BaseEnsemble, MNumber
from .base import BaseOCEAN


class OCEAN(BaseOCEAN):
    DEFAULT_EPS = 1e-6

    _new_weights: MNumber
    _eps: float
    _maj_class_constrs: gp.tupledict[int, gp.Constr]

    def __init__(
        self,
        base: BaseEnsemble,
        encoder: FeatureEncoder,
        weights: npt.ArrayLike,
        *,
        name: str = "OCEAN",
        env: gp.Env | None = None,
        tol: float = BaseOCEAN.DEFAULT_TOL,
        eps: float = DEFAULT_EPS,
    ) -> None:
        BaseOCEAN.__init__(
            self,
            base=base,
            encoder=encoder,
            weights=weights,
            name=name,
            env=env,
            tol=tol,
        )
        self._eps = eps
        self._maj_class_constrs = gp.tupledict()

    @property
    def new_weights(self) -> MNumber:
        return self._new_weights

    @new_weights.setter
    def new_weights(self, new_weights: MNumber) -> None:
        self._new_weights = np.copy(new_weights)

    def set_maj_class(self, maj_class: int) -> None:
        self._maj_class_constrs = gp.tupledict()
        for class_ in range(self.n_classes):
            if class_ == maj_class:
                continue
            self._add_majority_class_constr(maj_class=maj_class, class_=class_)

    def clear_majority_class(self) -> None:
        self.remove(self._maj_class_constrs)
        self._maj_class_constrs = gp.tupledict()

    def _add_majority_class_constr(self, maj_class: int, class_: int) -> None:
        rhs = self._eps if maj_class > class_ else 0.0
        name = f"majority_class_constr_{maj_class}_class_{class_}"
        expr = self.function(maj_class) - self.function(class_)
        constr = self.addConstr(expr >= rhs, name=name)
        self._maj_class_constrs[class_] = constr
