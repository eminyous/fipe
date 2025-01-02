from abc import ABCMeta
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
import xgboost as xgb

from ...typing import (
    LightGBMBooster,
    LightGBMParsableTree,
    MProb,
    Prob,
    XGBoostBooster,
    XGBoostParsableTree,
)
from ..binder import GenericBinder

Booster = LightGBMBooster | XGBoostBooster
ParsableBoosterTree = LightGBMParsableTree | XGBoostParsableTree
Data = npt.ArrayLike | xgb.DMatrix

BT = TypeVar("BT", bound=Booster)
PT = TypeVar("PT", bound=ParsableBoosterTree)


class BoosterBinder(GenericBinder[BT, PT], Generic[BT, PT]):
    __metaclass__ = ABCMeta

    def _predict_leaf(self, leaf_index: int, index: int) -> Prob:
        return self.callback.predict_leaf(leaf_index=leaf_index, index=index)

    def _scores_impl(self, X: npt.ArrayLike) -> MProb:
        x = self._transform(X)
        leaf_indices = self._base.predict(x, pred_leaf=True)
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

    @staticmethod
    def _transform(X: npt.ArrayLike) -> Data:
        return X
