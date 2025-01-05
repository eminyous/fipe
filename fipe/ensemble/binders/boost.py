from abc import abstractmethod
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
import xgboost as xgb

from ...typing import (
    LightGBMBooster,
    LightGBMTree,
    MProb,
    Prob,
    XGBoostBooster,
    XGBoostTree,
)
from .generic import GenericBinder

Booster = LightGBMBooster | XGBoostBooster
ParsableBoosterTree = LightGBMTree | XGBoostTree

BT = TypeVar("BT", bound=Booster)
PT = TypeVar("PT", bound=ParsableBoosterTree)


class BoosterBinder(GenericBinder[BT, PT], Generic[BT, PT]):
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

    def _predict_base(self, X: npt.ArrayLike) -> npt.NDArray[np.int64]:
        if isinstance(self._base, XGBoostBooster):
            dX = xgb.DMatrix(X)
            indices = self._base.predict(dX, pred_leaf=True)
        else:
            indices = self._base.predict(X, pred_leaf=True)
        return np.asarray(indices, dtype=np.int64)

    def _predict_proba_impl(
        self,
        X: npt.ArrayLike,
        *,
        probs: MProb,
    ) -> None:
        indices = self._predict_base(X)
        self._predict_proba_leaf(indices=indices, probs=probs)

    def _predict_proba_leaf(
        self,
        indices: npt.NDArray[np.int64],
        *,
        probs: MProb,
    ) -> None:
        for i, _ in enumerate(indices):
            self._predict_prob_s(indices=indices[i], probs=probs[i])

    def _predict_prob_s(
        self,
        indices: npt.NDArray[np.int64],
        *,
        probs: MProb,
    ) -> None:
        for e in range(self.n_estimators):
            self._predict_proba_e(e=e, indices=indices, probs=probs[e])

    def _predict_proba_e(
        self,
        e: int,
        indices: npt.NDArray[np.int64],
        *,
        probs: MProb,
    ) -> None:
        if self.is_binary:
            index = int(indices[e])
            probs[1] = self._predict_leaf(e=e, index=index)
            probs[0] = -probs[1]
            return

        n_classes = self.n_classes
        for k in range(n_classes):
            ee = e * n_classes + k
            index = int(indices[ee])
            probs[k] = self._predict_leaf(e=ee, index=index)
