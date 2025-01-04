from abc import ABCMeta, abstractmethod
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

    @property
    def n_classes(self) -> int:
        n_per_iter = self.n_trees_per_iter
        return n_per_iter + int(n_per_iter == 1)

    @property
    def n_estimators(self) -> int:
        n_trees = self.n_trees
        n_per_iter = self.n_trees_per_iter
        return n_trees // n_per_iter

    @property
    @abstractmethod
    def n_trees(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_trees_per_iter(self) -> int:
        raise NotImplementedError

    def _predict_leaf(self, leaf_index: int, index: int) -> Prob:
        return self.callback.predict_leaf(leaf_index=leaf_index, index=index)

    def _scores_impl(
        self,
        X: npt.ArrayLike,
        *,
        scores: MProb,
    ) -> None:
        x = self._transform(X)
        leaf_indices = self._base.predict(x, pred_leaf=True)
        leaf_indices = np.asarray(
            leaf_indices,
            dtype=np.int32,
        )
        self._scores_leaf(leaf_indices=leaf_indices, scores=scores)

    def _scores_leaf(
        self,
        leaf_indices: npt.NDArray[np.int64],
        *,
        scores: MProb,
    ) -> None:
        for i, _ in enumerate(leaf_indices):
            self._scores_sample(
                leaf_indices=leaf_indices[i],
                scores=scores[i],
            )

    def _scores_sample(
        self,
        leaf_indices: npt.NDArray[np.int64],
        *,
        scores: MProb,
    ) -> None:
        for j in range(self.n_estimators):
            self._scores_estimator(j, leaf_indices, scores=scores[j])

    def _scores_estimator(
        self,
        index: int,
        leaf_indices: npt.NDArray[np.int64],
        *,
        scores: MProb,
    ) -> None:
        if self.is_binary:
            leaf_index = int(leaf_indices[index])
            scores[1] = self._predict_leaf(leaf_index, index)
            scores[0] = -scores[1]
        else:
            n_classes = self.n_classes
            for k in range(n_classes):
                e = index * n_classes + k
                leaf_index = int(leaf_indices[e])
                scores[k] = self._predict_leaf(leaf_index, e)

    @staticmethod
    def _transform(X: npt.ArrayLike) -> Data:
        return X
