from collections.abc import Iterable, Iterator
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ..feature import FeatureEncoder
from ..tree import Tree
from ..typing import BaseEnsemble
from .classes import CLASSES
from .parser import EnsembleParser


class Ensemble(Iterable[Tree]):
    NUM_BINARY_CLASSES = EnsembleParser.NUM_BINARY_CLASSES
    CLASSES: ClassVar[dict[type, type]] = CLASSES

    _parser: EnsembleParser[Tree, BaseEnsemble]

    def __init__(
        self,
        base: BaseEnsemble,
        encoder: FeatureEncoder,
    ) -> None:
        cls = self.fetch_cls(base=base)
        self._parser = cls(base=base, encoder=encoder)

    def predict(
        self,
        X: npt.ArrayLike,
        w: npt.ArrayLike,
    ) -> npt.NDArray[np.intp]:
        return self._parser.predict(X=X, w=w)

    def score(
        self,
        X: npt.ArrayLike,
        w: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        return self._parser.score(X=X, w=w)

    def scores(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return self._parser.scores(X=X)

    @property
    def n_classes(self) -> int:
        return self._parser.n_classes

    @property
    def n_estimators(self) -> int:
        return self._parser.n_estimators

    @property
    def max_depth(self) -> int:
        return self._parser.max_depth

    @property
    def m_valued(self) -> bool:
        return self._parser.m_valued

    def __getitem__(self, t: int) -> Tree:
        return self._parser[t]

    def __iter__(self) -> Iterator[Tree]:
        return iter(self._parser)

    def __len__(self) -> int:
        return len(self._parser)

    @staticmethod
    def fetch_cls(base: BaseEnsemble) -> type[EnsembleParser]:
        for parsable_cls, cls in Ensemble.CLASSES.items():
            if isinstance(base, parsable_cls):
                return cls
        msg = f"Unknown base ensemble class: {type(base).__name__}"
        raise ValueError(msg)
