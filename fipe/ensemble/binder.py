from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from ..typing import BaseEnsemble, MClass, MProb, ParsableTree, Prob

BE = TypeVar("BE", bound=BaseEnsemble)
PT = TypeVar("PT", bound=ParsableTree)


class BinderCallback:
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict_leaf(self, leaf_index: int, index: int) -> Prob:
        raise NotImplementedError


class GenericBinder(Generic[BE, PT]):
    __metaclass__ = ABCMeta

    NUM_BINARY_CLASSES = 2

    _base: BE
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
        p = self.score(X=X, w=w)
        return np.argmax(p, axis=-1)

    def score(self, X: npt.ArrayLike, w: npt.ArrayLike) -> MProb:
        w = np.asarray(w)
        p = self.scores(X=X)
        for e in range(self.n_estimators):
            p[:, e, :] *= w[e]
        return np.sum(p, axis=1) / np.sum(w)

    def scores(self, X: npt.ArrayLike) -> MProb:
        X = np.array(X, ndmin=2)
        return self._scores_impl(X=X)

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
    def _scores_impl(self, X: npt.ArrayLike) -> MProb:
        raise NotImplementedError
