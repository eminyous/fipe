from collections.abc import Generator
from typing import override

import numpy.typing as npt

from ...typing import (
    DecisionTreeRegressor,
    GradientBoostingClassifier,
    MProb,
)
from .skl import SKLearnBinder


class GradientBoostingBinder(
    SKLearnBinder[GradientBoostingClassifier, DecisionTreeRegressor],
):
    @property
    @override
    def n_classes(self) -> int:
        return self._base.n_classes_

    @property
    @override
    def n_estimators(self) -> int:
        return self._base.n_estimators_

    @property
    @override
    def base_estimators(self) -> Generator[DecisionTreeRegressor, None, None]:
        yield from self._base.estimators_.ravel()

    @override
    def _predict_proba_impl(self, X: npt.ArrayLike, *, probs: MProb) -> None:
        for e in range(self.n_estimators):
            self._predict_proba_e(e, X, scores=probs[:, e, :])

    def _predict_proba_e(
        self,
        e: int,
        X: npt.ArrayLike,
        *,
        scores: MProb,
    ) -> None:
        if self.is_binary:
            scores[:, 1] = self._base.estimators_[e, 0].predict(X)
            scores[:, 0] = -scores[:, 1]
            return

        n_classes = self.n_classes
        for k in range(n_classes):
            scores[:, k] = self._base.estimators_[e, k].predict(X)
