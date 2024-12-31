from abc import ABCMeta, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Generic

import numpy as np
import numpy.typing as npt

from ..feature import FeatureEncoder
from ..tree import BT
from ..typing import BE


class EnsembleParser(Sequence[BT], Generic[BT, BE]):
    __metaclass__ = ABCMeta

    NUM_BINARY_CLASSES = 2

    _base: BE
    _trees: list[BT]

    def __init__(self, base: BE, encoder: FeatureEncoder) -> None:
        self._base = base
        self._parse_trees(encoder=encoder)

    def predict(
        self,
        X: npt.ArrayLike,
        w: npt.ArrayLike,
    ) -> npt.NDArray[np.intp]:
        p = self.score(X=X, w=w)
        return np.argmax(p, axis=-1)

    def score(
        self,
        X: npt.ArrayLike,
        w: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        w = np.asarray(w)
        p = self.scores(X=X)
        for e in range(self.n_estimators):
            p[:, e, :] *= w[e]
        return np.sum(p, axis=1) / np.sum(w)

    def scores(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._scores_impl(X=X)

    def __iter__(self) -> Iterator[BT]:
        return iter(self._trees)

    def __len__(self) -> int:
        return len(self._trees)

    def __getitem__(self, index: int) -> BT:
        return self._trees[index]

    @property
    def max_depth(self) -> int:
        return max(tree.max_depth for tree in self)

    @property
    def is_binary(self) -> bool:
        return self.n_classes == self.NUM_BINARY_CLASSES

    @property
    @abstractmethod
    def n_classes(self) -> int:
        msg = "n_classes property must be implemented in subclass."
        raise NotImplementedError(msg)

    @property
    @abstractmethod
    def n_estimators(self) -> int:
        msg = "n_estimators property must be implemented in subclass."
        raise NotImplementedError(msg)

    @abstractmethod
    def _parse_trees(self, encoder: FeatureEncoder) -> None:
        msg = "_parse_trees method must be implemented in subclass."
        raise NotImplementedError(msg)

    @abstractmethod
    def _scores_impl(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        msg = "_scores_impl method must be implemented in subclass."
        raise NotImplementedError(msg)
