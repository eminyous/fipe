from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import ArrayLike, NDArray

from ...feature import FeatureEncoder
from ..tree import Tree

TREE_INFO_KEY = "tree_info"

NUM_BINARY_CLASSES = 2


class Ensemble(Iterable[Tree]):
    _base: LGBMClassifier
    _trees: list[Tree]

    def __init__(self, base: LGBMClassifier, encoder: FeatureEncoder) -> None:
        self._base = base
        self._parse_trees(encoder=encoder)

    def predict(self, X: ArrayLike, w: ArrayLike) -> NDArray[np.intp]:
        p = self.score(X=X, w=w)
        return np.argmax(p, axis=-1)

    def score(self, X: ArrayLike, w: ArrayLike) -> NDArray[np.float64]:
        w = np.asarray(w)
        p = self.scores(X=X)
        for e in range(self.n_estimators):
            p[:, e, :] *= w[e]
        return np.sum(p, axis=1) / np.sum(w)

    def scores(self, X: ArrayLike) -> NDArray[np.float64]:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._base_scores(X=X)

    @property
    def n_classes(self) -> int:
        return self._base.n_classes_

    @property
    def n_estimators(self) -> int:
        return self._base.n_estimators_

    @property
    def max_depth(self) -> int:
        return max(tree.max_depth for tree in self)

    def __iter__(self) -> Iterator[Tree]:
        return iter(self._trees)

    def __len__(self) -> int:
        return len(self._trees)

    def __getitem__(self, index: int) -> Tree:
        return self._trees[index]

    def _parse_trees(self, encoder: FeatureEncoder) -> None:
        model = self._base.booster_.dump_model()
        trees = model["tree_info"]

        def parse_tree(tree: dict[str, Any]) -> Tree:
            return Tree(tree=tree, encoder=encoder)

        self._trees = list(map(parse_tree, trees))

    def _base_scores(self, X: ArrayLike) -> NDArray[np.float64]:
        leaf_indices = self._base.predict_proba(X, pred_leaf=True)
        if self.n_classes == NUM_BINARY_CLASSES:
            return self._binary_base_scores(X=X, leaf_indices=leaf_indices)
        return self._multi_base_scores(X=X, leaf_indices=leaf_indices)

    def _binary_base_scores(
        self,
        X: ArrayLike,
        leaf_indices: NDArray[np.int32],
    ) -> NDArray[np.float64]:
        n_samples = int(X.shape[0])
        n_classes = self.n_classes
        n_estimators = self.n_estimators
        scores = np.zeros((n_samples, n_estimators, n_classes))
        for i in range(n_samples):
            for j in range(n_estimators):
                leaf_index = int(leaf_indices[i, j])
                value = self[j].predict(leaf_index)
                scores[i, j, 1] = value
                scores[i, j, 0] = -value
        return scores

    def _multi_base_scores(
        self,
        X: ArrayLike,
        leaf_indices: NDArray[np.int32],
    ) -> NDArray[np.float64]:
        n_samples = int(X.shape[0])
        n_classes = self.n_classes
        n_estimators = self.n_estimators
        scores = np.zeros((n_samples, n_estimators, n_classes))
        for i in range(n_samples):
            for j in range(n_estimators):
                for k in range(n_classes):
                    e = j * n_classes + k
                    leaf_index = int(leaf_indices[i, e])
                    value = self[e].predict(leaf_index)
                    scores[i, j, k] = value
        return scores
