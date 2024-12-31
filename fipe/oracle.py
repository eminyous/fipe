from collections.abc import Callable, Generator

import gurobipy as gp
import numpy as np
import numpy.typing as npt

from .ocean import OCEAN
from .typing import SNumber


class Oracle(OCEAN):
    def separate(
        self,
        new_weights: npt.ArrayLike,
    ) -> Generator[SNumber, None, None]:
        new_weights = np.copy(new_weights)
        self.new_weights = new_weights
        yield from self._separate()

    def _separate(self) -> Generator[SNumber, None, None]:
        for class_ in range(self.n_classes):
            yield from self._separate_class(majority_class=class_)

    def _separate_class(
        self,
        majority_class: int,
    ) -> Generator[SNumber, None, None]:
        self.set_majority_class(class_=majority_class)
        for class_ in range(self.n_classes):
            if class_ == majority_class:
                continue
            self._separate_pair(majority_class=majority_class, class_=class_)
            yield from self._extract_valid_samples(
                majority_class=majority_class,
                class_=class_,
            )
        self.clear_majority_class()

    def _extract_valid_samples(
        self,
        majority_class: int,
        class_: int,
    ) -> Generator[SNumber, None, None]:
        param = gp.GRB.Param.SolutionNumber
        for i in range(self.SolCount):
            self.setParam(param, i)

            if self.PoolObjVal < 0.0:
                continue

            if self.PoolObjVal == 0.0 and majority_class < class_:
                continue

            x = self._feature_vars.Xn
            X = self.transform(x)
            y_pred = self.ensemble.predict(X=X, w=self._new_weights)
            y_true = self.ensemble.predict(X=X, w=self._weights)
            if np.all(y_pred == y_true):
                continue
            yield x
        self.setParam(param, 0)

    def _separate_pair(self, majority_class: int, class_: int) -> None:
        class_score = self.weighted_function(
            class_=class_,
            weights=self._new_weights,
        )
        majority_class_score = self.weighted_function(
            class_=majority_class,
            weights=self._new_weights,
        )
        obj = class_score - majority_class_score
        self.setObjective(obj, gp.GRB.MAXIMIZE)
        cb = self._optimize_callback()
        self.optimize(cb)

    @staticmethod
    def _optimize_callback() -> Callable[[gp.Model, int], None]:
        def cb(model: gp.Model, where: int) -> None:
            if where == gp.GRB.Callback.MIPSOL:
                val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
                if val <= 0.0:
                    model.terminate()

        return cb
