from collections.abc import Iterable, Iterator, Sequence
from typing import override

import numpy.typing as npt

from ..feature import FeatureEncoder
from ..tree import Tree, TreeParser, create_parser
from ..typing import BaseEnsemble, MClass, MProb, ParsableTree, Prob
from .binders import Binder, create_binder
from .binders.generic import BinderCallback


class Ensemble(BinderCallback, Iterable[Tree]):
    _binder: Binder
    _parser: TreeParser
    _trees: Sequence[Tree]

    def __init__(self, base: BaseEnsemble, encoder: FeatureEncoder) -> None:
        self._init_binder(base=base)
        self._init_parser(base=base, encoder=encoder)
        self._parse_trees()

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

    def _init_binder(self, base: BaseEnsemble) -> None:
        self._binder = create_binder(base=base, callback=self)

    def _init_parser(self, base: BaseEnsemble, encoder: FeatureEncoder) -> None:
        self._parser = create_parser(base=base, encoder=encoder)

    def _parse_trees(self) -> None:
        def _parse(tree: ParsableTree) -> Tree:
            return self._parser.parse(tree)

        base_trees = self._binder.base_trees
        self._trees = list(map(_parse, base_trees))
