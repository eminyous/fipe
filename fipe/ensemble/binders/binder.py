from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from ...typing import BaseEnsemble, MClass, MProb, ParsableTree, Prob
from .callback import BinderCallback

BE = TypeVar("BE", bound=BaseEnsemble)
PT = TypeVar("PT", bound=ParsableTree)


class Binder(ABC, Generic[BE, PT]):
    NUM_BINARY_CLASSES = 2

    # Protected attributes
    _base: BE

    # Private attributes
    __callback: BinderCallback

    def __init__(
        self,
        base: BE,
        *,
        callback: BinderCallback,
    ) -> None:
        self._base = base
        self.__callback = callback

    @property
    def callback(self) -> BinderCallback:
        return self.__callback

    def predict(self, X: npt.ArrayLike, w: npt.ArrayLike) -> MClass:
        p = self.predict_weighted_proba(X=X, w=w)
        return np.argmax(p, axis=-1)

    def predict_weighted_proba(
        self,
        X: npt.ArrayLike,
        w: npt.ArrayLike,
    ) -> MProb:
        w = np.asarray(w)
        p = self.predict_proba(X=X)
        for e in range(self.n_estimators):
            p[:, e, :] *= w[e]
        return np.sum(p, axis=1) / np.sum(w)

    def predict_proba(self, X: npt.ArrayLike) -> MProb:
        X = np.array(X, ndmin=2)
        n_samples = X.shape[0]
        n_classes = self.n_classes
        n_estimators = self.n_estimators
        shape = (n_samples, n_estimators, n_classes)
        scores = np.empty(shape=shape, dtype=Prob)
        self._predict_proba_impl(X=X, probs=scores)
        return scores

    @property
    def is_binary(self) -> bool:
        return self.n_classes == self.NUM_BINARY_CLASSES

    @property
    @abstractmethod
    def n_classes(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_estimators(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def base_trees(self) -> Generator[PT, None, None]:
        raise NotImplementedError

    @abstractmethod
    def _predict_proba_impl(self, X: npt.ArrayLike, *, probs: MProb) -> None:
        raise NotImplementedError
