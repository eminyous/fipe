from collections.abc import Iterator, Sequence

import numpy.typing as npt

from ..feature import FeatureEncoder
from ..tree import Tree, TreeParser, create_parser
from ..typing import BaseEnsemble, MClass, MProb, Prob
from .classes import Ens, create_ensemble
from .generic import Callback


class Ensemble(Sequence[Tree], Callback):
    ensemble: Ens
    tree_parser: TreeParser
    trees: Sequence[Tree]

    def __init__(self, base: BaseEnsemble, encoder: FeatureEncoder) -> None:
        self.ensemble = self.init_ensemble(base=base, callback=self)
        self.tree_parser = self.init_tree_parser(base=base, encoder=encoder)
        parse = self.tree_parser.parse
        base_trees = self.ensemble.base_trees
        self.trees = list(map(parse, base_trees))

    def predict(self, X: npt.ArrayLike, w: npt.ArrayLike) -> MClass:
        return self.ensemble.predict(X=X, w=w)

    def score(self, X: npt.ArrayLike, w: npt.ArrayLike) -> MProb:
        return self.ensemble.score(X=X, w=w)

    def scores(self, X: npt.ArrayLike) -> MProb:
        return self.ensemble.scores(X=X)

    def predict_leaf(self, leaf_index: int, index: int) -> Prob:
        return Prob(self[index].predict(leaf_index))

    @property
    def is_binary(self) -> bool:
        return self.ensemble.is_binary

    @property
    def n_classes(self) -> int:
        return self.ensemble.n_classes

    @property
    def n_estimators(self) -> int:
        return self.ensemble.n_estimators

    @property
    def max_depth(self) -> int:
        return max(tree.max_depth for tree in self)

    def __getitem__(self, t: int) -> Tree:
        return self.trees[t]

    def __iter__(self) -> Iterator[Tree]:
        return iter(self.trees)

    def __len__(self) -> int:
        return len(self.trees)

    @staticmethod
    def init_ensemble(base: BaseEnsemble, callback: Callback) -> Ens:
        return create_ensemble(base=base, callback=callback)

    @staticmethod
    def init_tree_parser(
        base: BaseEnsemble,
        encoder: FeatureEncoder,
    ) -> TreeParser:
        return create_parser(base=base, encoder=encoder)
