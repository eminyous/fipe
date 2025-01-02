from collections.abc import Iterator, Sequence

import numpy.typing as npt

from ..feature import FeatureEncoder
from ..tree import Tree, TreeParser, create_parser
from ..typing import BaseEnsemble, MClass, MProb, Prob
from .binder import EnsembleBinderCallback
from .binders import EnsembleBinder, create_binder


class Ensemble(Sequence[Tree], EnsembleBinderCallback):
    _binder: EnsembleBinder
    _tree_parser: TreeParser
    _trees: Sequence[Tree]

    def __init__(self, base: BaseEnsemble, encoder: FeatureEncoder) -> None:
        self._binder = self.init_ensemble_binder(base=base, callback=self)
        self._tree_parser = self.init_tree_parser(base=base, encoder=encoder)
        parse = self._tree_parser.parse
        base_trees = self._binder.base_trees
        self._trees = list(map(parse, base_trees))

    def predict(self, X: npt.ArrayLike, w: npt.ArrayLike) -> MClass:
        return self._binder.predict(X=X, w=w)

    def score(self, X: npt.ArrayLike, w: npt.ArrayLike) -> MProb:
        return self._binder.score(X=X, w=w)

    def scores(self, X: npt.ArrayLike) -> MProb:
        return self._binder.scores(X=X)

    def predict_leaf(self, leaf_index: int, index: int) -> Prob:
        return Prob(self[index].predict(leaf_index))

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

    def __iter__(self) -> Iterator[Tree]:
        return iter(self._trees)

    def __len__(self) -> int:
        return len(self._trees)

    @staticmethod
    def init_ensemble_binder(
        base: BaseEnsemble, callback: EnsembleBinderCallback
    ) -> EnsembleBinder:
        return create_binder(base=base, callback=callback)

    @staticmethod
    def init_tree_parser(
        base: BaseEnsemble,
        encoder: FeatureEncoder,
    ) -> TreeParser:
        return create_parser(base=base, encoder=encoder)
