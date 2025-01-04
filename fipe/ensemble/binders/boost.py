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

    def _predict_leaf(self, e: int, index: int) -> Prob:
        return self.callback.predict_leaf(e=e, index=index)

    def _base_predict(self, X: Data) -> npt.NDArray[np.int32]:
        return np.asarray(self._base.predict(X, pred_leaf=True))

    def _scores_impl(
        self,
        X: npt.ArrayLike,
        *,
        scores: MProb,
    ) -> None:
        dX = self._transform(X)
        indices = self._base_predict(dX)
        self._scores_leaf(indices=indices, scores=scores)

    def _scores_leaf(
        self,
        indices: npt.NDArray[np.int64],
        *,
        scores: MProb,
    ) -> None:
        for i, _ in enumerate(indices):
            self._scores_sample(indices=indices[i], scores=scores[i])

    def _scores_sample(
        self,
        indices: npt.NDArray[np.int64],
        *,
        scores: MProb,
    ) -> None:
        for e in range(self.n_estimators):
            self._scores_estimator(e=e, indices=indices, scores=scores[e])

    def _scores_estimator(
        self,
        e: int,
        indices: npt.NDArray[np.int64],
        *,
        scores: MProb,
    ) -> None:
        if self.is_binary:
            index = int(indices[e])
            scores[1] = self._predict_leaf(e=e, index=index)
            scores[0] = -scores[1]
            return

        n_classes = self.n_classes
        for k in range(n_classes):
            j = e * n_classes + k
            index = int(indices[j])
            scores[k] = self._predict_leaf(e=j, index=index)

    @staticmethod
    def _transform(X: npt.ArrayLike) -> Data:
        return X
