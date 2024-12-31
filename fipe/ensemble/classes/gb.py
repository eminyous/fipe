import numpy as np
import numpy.typing as npt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor

from ...feature import FeatureEncoder
from ...tree import TreeGB
from ..base import BaseEnsemble


class EnsembleGB(BaseEnsemble[TreeGB, GradientBoostingClassifier]):
    @property
    def m_valued(self) -> bool:
        return True

    @property
    def n_classes(self) -> int:
        return self._base.n_classes_

    @property
    def n_estimators(self) -> int:
        return self._base.n_estimators_

    @property
    def _base_estimators(self) -> list[list[DecisionTreeRegressor]]:
        estimators = []
        for i in range(self.n_estimators):
            if self.n_classes == self.NUM_BINARY_CLASSES:
                estimator = [self._base.estimators_[i, 0]]
            else:
                estimator = list(self._base.estimators_[i])
            estimators.append(estimator)
        return estimators

    @property
    def _base_trees(self) -> list[DecisionTreeRegressor]:
        estimators = self._base_estimators
        return [tree for estimator in estimators for tree in estimator]

    def _parse_trees(self, encoder: FeatureEncoder) -> None:
        trees = self._base_trees

        def parse_tree(tree: DecisionTreeRegressor) -> TreeGB:
            return TreeGB(tree=tree.tree_, encoder=encoder)

        self._trees = list(map(parse_tree, trees))

    def _scores_impl(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_estimators = self.n_estimators
        n_classes = self.n_classes
        scores = np.zeros((n_samples, n_estimators, n_classes))
        for i in range(n_samples):
            scores[i] = self._compute_scores_sample(X[i])

        return scores

    def _compute_scores_sample(
        self,
        x: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        x = np.asarray(x)
        n_estimators = self.n_estimators
        n_classes = self.n_classes
        scores = np.zeros((n_estimators, n_classes))
        for i in range(n_estimators):
            scores[i] = self._compute_base_scores_estimator(i, x)

        return scores

    def _compute_base_scores_estimator(
        self,
        index: int,
        x: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        x = np.asarray(x).reshape(1, -1)
        n_classes = self.n_classes
        scores = np.zeros(n_classes)
        if n_classes == self.NUM_BINARY_CLASSES:
            scores[1] = self._base_estimators[index][0].predict(x)[0]
            scores[0] = -scores[1]
        else:
            for j in range(n_classes):
                scores[j] = self._base_estimators[index][j].predict(x)[0]
        return scores


CLASSES = {GradientBoostingClassifier: EnsembleGB}
