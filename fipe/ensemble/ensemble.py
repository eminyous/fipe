from collections.abc import Iterable, Iterator, Sequence
from typing import override

import numpy.typing as npt

from ..feature import FeatureEncoder
from ..tree import Tree
from ..typing import (
    BaseEnsemble,
    MClass,
    MProb,
    Prob,
)
from .binders.callback import BinderCallback
from .builder import Binder, Builder, create_builder


class Ensemble(BinderCallback, Iterable[Tree]):
    _builder: Builder
    _trees: Sequence[Tree]

    def __init__(self, base: BaseEnsemble, encoder: FeatureEncoder) -> None:
        self._builder = create_builder(
            base=base,
            encoder=encoder,
            callback=self,
        )
        self._trees = self._builder.parse_trees()

    def predict(self, X: npt.ArrayLike, w: npt.ArrayLike) -> MClass:
        return self._binder.predict(X=X, w=w)

    def predict_weighted_proba(
        self,
        X: npt.ArrayLike,
        w: npt.ArrayLike,
    ) -> MProb:
        return self._binder.predict_weighted_proba(X=X, w=w)

    def predict_proba(self, X: npt.ArrayLike) -> MProb:
        return self._binder.predict_proba(X=X)

    @override
    def predict_leaf(self, e: int, index: int) -> Prob:
        return Prob(self[e].predict(index))

    @property
    def is_binary(self) -> bool:
        return self._binder.is_binary

    @property
    def n_classes(self) -> int:
        return self._binder.n_classes

    @property
    def n_estimators(self) -> int:
        return self._binder.n_estimators

    @property
    def max_depth(self) -> int:
        return max(tree.max_depth for tree in self)

    def __getitem__(self, t: int) -> Tree:
        return self._trees[t]

    @override
    def __iter__(self) -> Iterator[Tree]:
        return iter(self._trees)

    def __len__(self) -> int:
        return len(self._trees)

    @property
    def _binder(self) -> Binder:
        return self._builder.binder
