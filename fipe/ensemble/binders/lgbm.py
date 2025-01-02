from collections.abc import Generator

import numpy as np
import numpy.typing as npt

from ...typing import LightGBMBooster, LightGBMParsableTree, MProb, Prob
from ..binder import GenericBinder


class LightGBMBinder(GenericBinder[LightGBMBooster, LightGBMParsableTree]):
    TREE_INFO_KEY = "tree_info"

    @property
    def n_classes(self) -> int:
        n_per_iter = self._base.num_model_per_iteration()
        return n_per_iter + int(n_per_iter == 1)

    @property
    def n_estimators(self) -> int:
        n_trees = self._base.num_trees()
        n_per_iter = self._base.num_model_per_iteration()
        return n_trees // n_per_iter

    @property
    def base_trees(self) -> Generator[LightGBMParsableTree, None, None]:
        model = self._base.dump_model()
        yield from model[self.TREE_INFO_KEY]

    def _predict_leaf(self, leaf_index: int, index: int) -> Prob:
        return self.callback.predict_leaf(leaf_index=leaf_index, index=index)

    def _scores_impl(self, X: npt.ArrayLike) -> MProb:
        leaf_indices = self._base.predict(X, pred_leaf=True)
        leaf_indices = np.asarray(
            leaf_indices,
            dtype=np.int32,
        )
        return self._scores_leaf(leaf_indices=leaf_indices)

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
        n_estimators = self.n_estimators
        n_classes = self.n_classes
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
