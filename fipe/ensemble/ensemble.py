from collections.abc import Iterable, Iterator

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from ..feature import FeatureEncoder
from ..tree import Tree, TreeClassifier, TreeRegressor
from ..typing import BaseEnsemble


class Ensemble(Iterable[Tree]):
    _base: BaseEnsemble
    _trees: list[Tree]

    def __init__(
        self,
        base: BaseEnsemble,
        encoder: FeatureEncoder,
    ):
        self._base = base
        self._parse_trees(encoder)

    def predict(self, X, w):
        """Return class of input points for given weights."""
        p = self.score(X, w)
        return np.argmax(p, axis=-1)

    def score(self, X, w):
        p = self.scores(X)
        for i in range(len(self)):
            p[:, i, :] *= w[i]
        return np.sum(p, axis=1) / np.sum(w)

    def scores(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        scores = []
        for e in self.estimators:
            scores.append(self._scores(X, e))
        p = np.array(scores)
        p = np.swapaxes(p, 0, 1)
        return p

    @property
    def estimators(self):
        if isinstance(self._base, GradientBoostingClassifier):
            return self._base.estimators_[:, 0].ravel()
        return self._base.estimators_

    @property
    def n_classes(self) -> int:
        assert isinstance(self._base.n_classes_, int)
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

    def _parse_trees(self, encoder: FeatureEncoder):
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

    def _scores(self, X, e):
        if isinstance(self._base, AdaBoostClassifier):
            q = e.predict(X)
            k = e.n_classes_
            p = np.eye(k)[q]
            return p
        if isinstance(self._base, GradientBoostingClassifier):
            q = e.predict(X)
            p = np.array([-q, q]).T
            return p
        return e.predict_proba(X)
