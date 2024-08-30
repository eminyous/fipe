from collections.abc import Iterable, Iterator

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor

from ..feature import FeatureEncoder
from ..tree import Tree, TreeClassifier, TreeRegressor
from ..typing import BaseEnsemble, BaseEstimator


class Ensemble(Iterable[Tree]):

    _base: BaseEnsemble
    _trees: list[Tree]

    def __init__(
        self,
        base: BaseEnsemble,
        encoder: FeatureEncoder,
    ) -> None:
        self._base = base
        self._parse_trees(encoder)

    def predict(self, X: ArrayLike, w: ArrayLike) -> np.ndarray:
        p = self.score(X, w)
        return np.argmax(p, axis=-1)

    def score(self, X: ArrayLike, w: ArrayLike) -> np.ndarray:
        w = np.asarray(w)
        p = self.scores(X)
        for i in range(len(self)):
            p[:, i, :] *= w[i]
        return np.sum(p, axis=1) / np.sum(w)

    def scores(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        p = np.array([self._scores(X, e) for e in self.estimators])
        return np.swapaxes(p, 0, 1)

    @property
    def estimators(self) -> list[BaseEstimator]:
        if isinstance(self._base, GradientBoostingClassifier):
            return self._base.estimators_[:, 0].ravel().tolist()
        if isinstance(self._base, AdaBoostClassifier):
            return list(self._base)
        return list(self._base.estimators_)

    @property
    def n_classes(self) -> int:
        if not isinstance(self._base.n_classes_, int):
            msg = "n_classes must be an integer."
            raise TypeError(msg)
        return self._base.n_classes_

    @property
    def n_estimators(self) -> int:
        return len(self._trees)

    @property
    def max_depth(self) -> int:
        return max(tree.max_depth for tree in self)

    def __getitem__(self, t: int) -> Tree:
        return self._trees[t]

    def __iter__(self) -> Iterator[Tree]:
        return iter(self._trees)

    def __len__(self) -> int:
        return self.n_estimators

    def _parse_trees(self, encoder: FeatureEncoder) -> None:
        self._trees = []
        if isinstance(self._base, AdaBoostClassifier):
            self._trees = []
            for e in self._base:
                self._trees.append(TreeClassifier(e.tree_, encoder))
            return
        if isinstance(self._base, GradientBoostingClassifier):
            self._trees = []
            for e in self._base.estimators_[:, 0].ravel():
                self._trees.append(TreeRegressor(e.tree_, encoder))
            return
        for e in self._base:
            self._trees.append(Tree(e.tree_, encoder))

    def _scores(self, X: ArrayLike, e: BaseEstimator) -> ArrayLike:
        if isinstance(e, DecisionTreeRegressor):
            q = e.predict(X)
            return np.array([-q, q]).T
        if isinstance(self._base, AdaBoostClassifier):
            q = e.predict(X)
            k = self.n_classes
            return np.eye(k)[q]
        return e.predict_proba(X)
