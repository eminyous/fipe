from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor

from ...tree import TreeGB
from .skl import EnsembleSKL


class EnsembleGB(
    EnsembleSKL[TreeGB, GradientBoostingClassifier, DecisionTreeRegressor]
):
    __tree_class__ = TreeGB

    @property
    def n_classes(self) -> int:
        return self._base.n_classes_

    @property
    def n_estimators(self) -> int:
        return self._base.n_estimators_

    @property
    def _base_trees(self) -> Iterable[DecisionTreeRegressor]:
        return self._base.estimators_.ravel().tolist()

    def _scores_impl(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_estimators = self.n_estimators
        n_classes = self.n_classes
        scores = np.zeros((n_samples, n_estimators, n_classes))
        for j in range(n_estimators):
            scores[:, j, :] = self._compute_scores_estimator(j, X)

        return scores

    def _compute_scores_estimator(
        self,
        index: int,
        X: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
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


CLASSES = {GradientBoostingClassifier: EnsembleGB}
