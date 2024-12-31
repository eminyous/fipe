from collections.abc import Iterable, Iterator
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ..feature import FeatureEncoder
from ..tree import BaseTree
from ..typing import LeafValue, ParsableEnsemble, ParsableTree
from .base import BaseEnsemble
from .classes import CLASSES

Tree = BaseTree[LeafValue, ParsableTree]


class Ensemble(Iterable[Tree]):
    NUM_BINARY_CLASSES = BaseEnsemble.NUM_BINARY_CLASSES
    CLASSES: ClassVar[dict[type, type]] = CLASSES

    _base: BaseEnsemble[Tree, ParsableEnsemble]

    def __init__(self, base: ParsableEnsemble, encoder: FeatureEncoder) -> None:
        cls = self.fetch_cls(base=base)
        self._base = cls(base=base, encoder=encoder)

    def predict(
        self,
        X: npt.ArrayLike,
        w: npt.ArrayLike,
    ) -> npt.NDArray[np.intp]:
        return self._base.predict(X=X, w=w)

    def score(
        self,
        X: npt.ArrayLike,
        w: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        return self._base.score(X=X, w=w)

    def scores(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return self._base.scores(X=X)

    @property
    def n_classes(self) -> int:
        return self._base.n_classes

    @property
    def n_estimators(self) -> int:
        return self._base.n_estimators

    @property
    def max_depth(self) -> int:
        return self._base.max_depth

    @property
    def m_valued(self) -> bool:
        return self._base.m_valued

    def __getitem__(self, t: int) -> BaseTree[LeafValue, ParsableTree]:
        return self._base[t]

    def __iter__(self) -> Iterator[BaseTree[LeafValue, ParsableTree]]:
        return iter(self._base)

    def __len__(self) -> int:
        return len(self._base)

    @staticmethod
    def fetch_cls(base: ParsableEnsemble) -> type[BaseEnsemble]:
        for base_cls, cls in Ensemble.CLASSES.items():
            if isinstance(base, base_cls):
                return cls
        msg = f"Unknown ensemble class: {type(base).__name__}"
        raise ValueError(msg)
