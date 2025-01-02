from collections.abc import Generator

import numpy as np
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
    def n_classes(self) -> int:
        return self._base.n_classes_

    @property
    def n_estimators(self) -> int:
        return self._base.n_estimators_

    @property
    def base_estimators(self) -> Generator[DecisionTreeRegressor, None, None]:
        yield from self._base.estimators_.ravel()

    def _scores_impl(self, X: npt.ArrayLike) -> MProb:
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_estimators = self.n_estimators
        n_classes = self.n_classes
        scores = np.zeros((n_samples, n_estimators, n_classes))
        for j in range(n_estimators):
            scores[:, j, :] = self._scores_estimator(j, X)

        return scores

    def _scores_estimator(
        self,
        index: int,
        X: npt.ArrayLike,
    ) -> MProb:
        X = np.asarray(X)
        n_classes = self.n_classes
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, n_classes))
        if self.is_binary:
            scores[:, 1] = self._base.estimators_[index, 0].predict(X)
            scores[:, 0] = -scores[:, 1]
        else:
            for j in range(n_classes):
                scores[:, j] = self._base.estimators_[index, j].predict(X)
        return scores
