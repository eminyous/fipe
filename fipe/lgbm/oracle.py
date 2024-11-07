from collections.abc import Callable, Generator

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import ArrayLike

from ..feature import FeatureEncoder
from ..typing import Sample
from .ensemble import Ensemble
from .ocean import OCEAN


class Oracle(OCEAN):
    def __init__(
        self,
        encoder: FeatureEncoder,
        ensemble: Ensemble,
        weights: ArrayLike,
        **kwargs,
    ) -> None:
        OCEAN.__init__(
            self,
            encoder=encoder,
            ensemble=ensemble,
            weights=weights,
            **kwargs,
        )

    def separate(self, new_weights: ArrayLike) -> Generator[Sample, None, None]:
        new_weights = np.asarray(new_weights)
        self._set_new_weights(new_weights=new_weights)
        yield from self._separate()

    def _separate(self) -> Generator[Sample, None, None]:
        for class_ in range(self.n_classes):
            yield from self._separate_class(class_=class_)

    def _separate_class(self, class_: int) -> Generator[Sample, None, None]:
        self.set_majority_class(class_=class_)
        for c in range(self.n_classes):
            if c == class_:
                continue
            self._separate_pair(majority_class=class_, class_=c)
            yield from self._get_samples(majority_class=class_, class_=c)
        self.clear_majority_class()

    def _get_samples(
        self,
        majority_class: int,
        class_: int,
    ) -> Generator[Sample, None, None]:
        param = GRB.Param.SolutionNumber
        for i in range(self.SolCount):
            self.setParam(param, i)
            x = self._feature_vars.Xn

            if self.PoolObjVal < 0.0:
                continue

            if self.PoolObjVal == 0.0 and majority_class < class_:
                continue

            X = self.transform(x)
            y_pred = self.ensemble.predict(X=X, w=self._new_weights)
            y_true = self.ensemble.predict(X=X, w=self._weights)
            if np.all(y_pred == y_true):
                continue
            yield x
        self.setParam(param, 0)

    def _separate_pair(self, majority_class: int, class_: int) -> None:
        obj = self.weighted_function(
            class_=class_,
            weights=self._new_weights,
        ) - self.weighted_function(
            class_=majority_class,
            weights=self._new_weights,
        )
        self.setObjective(obj, GRB.MAXIMIZE)
        _callback = self._optimize_callback()
        self.optimize(_callback)

    @staticmethod
    def _optimize_callback() -> Callable[[gp.Model, int], None]:
        def callback(model: gp.Model, where: int) -> None:
            if where == GRB.Callback.MIPSOL:
                val = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
                if val <= 0.0:
                    model.terminate()

        return callback
