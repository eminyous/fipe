from collections.abc import Generator

import numpy as np
import numpy.typing as npt
import xgboost as xgb

from ...tree import XGBoostTreeParser
from ...typing import MProb, XGBoostBooster, XGBoostParsableTree
from ..binder import GenericBinder


class XGBoostBinder(GenericBinder[XGBoostBooster, XGBoostParsableTree]):
    TREE_KEY = "Tree"

    INDEX = (
        TREE_KEY,
        XGBoostTreeParser.NODE_KEY,
        XGBoostTreeParser.ID_KEY,
    )

    __n_trees: int | None = None

    @property
    def n_trees(self) -> int:
        if self.__n_trees is None:
            trees = self._base.trees_to_dataframe()[self.TREE_KEY].unique()
            self.__n_trees = len(trees)
        return self.__n_trees

    @property
    def n_classes(self) -> int:
        n_trees = self.n_trees
        return (n_trees // self.n_estimators) + int(
            self.n_estimators == n_trees
        )

    @property
    def n_estimators(self) -> int:
        return self._base.num_boosted_rounds()

    @property
    def base_trees(self) -> Generator[XGBoostParsableTree, None, None]:
        data = self._base.trees_to_dataframe().set_index(list(self.INDEX))
        for _, tree in data.groupby(level=self.TREE_KEY):
            yield tree.reset_index(level=self.TREE_KEY, drop=True)

    def _scores_impl(
        self,
        X: npt.ArrayLike,
    ) -> MProb:
        dX = xgb.DMatrix(X)
        leaf_indices = self._base.predict(dX, pred_leaf=True)
        leaf_indices = leaf_indices.astype(int)
        return self._scores_leaf(leaf_indices)

    def _scores_leaf(
        self,
        leaf_indices: npt.NDArray[np.int64],
    ) -> MProb:
        n_samples = int(leaf_indices.shape[0])
        n_classes = self.n_classes
        n_estimators = self.n_estimators
        scores = np.zeros((n_samples, n_estimators, n_classes))
        for i in range(n_samples):
            scores[i] = self._scores_sample(leaf_indices[i])

        return scores

    def _scores_sample(
        self,
        leaf_indices: npt.NDArray[np.int64],
    ) -> MProb:
        n_classes = self.n_classes
        n_estimators = self.n_estimators
        scores = np.zeros((n_estimators, n_classes))
        for j in range(n_estimators):
            scores[j] = self._scores_estimator(j, leaf_indices)

        return scores

    def _scores_estimator(
        self,
        index: int,
        leaf_indices: npt.NDArray[np.int64],
    ) -> MProb:
        n_classes = self.n_classes
        scores = np.zeros(n_classes)
        if self.is_binary:
            leaf_index = int(leaf_indices[index])
            scores[1] = self._predict_leaf(leaf_index, index)
            scores[0] = -scores[1]
            return scores

        for k in range(n_classes):
            e = index * n_classes + k
            leaf_index = int(leaf_indices[e])
            scores[k] = self._predict_leaf(leaf_index, e)
        return scores

    def _predict_leaf(
        self,
        leaf_index: int,
        index: int,
    ) -> float:
        return self.callback.predict_leaf(leaf_index=leaf_index, index=index)
