from abc import ABCMeta, abstractmethod
from collections.abc import Iterator, Sequence
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import Booster

from ..feature import FeatureEncoder
from ..tree import (
    Tree,
    TreeParser,
    TreeParserCL,
    TreeParserLGBM,
    TreeParserRG,
    TreeParserXGB,
)
from ..typing import BaseEnsemble, MClass, ParsableTree

Param = bool | int | str | float
Args = dict[str, Param]
TreeParserArgs = tuple[type[TreeParser], Args]


class EnsembleParser(Sequence[Tree]):
    __metaclass__ = ABCMeta

    NUM_BINARY_CLASSES = 2

    _base: BaseEnsemble
    _trees: list[Tree]
    _tree_parser: TreeParser

    TREE_PARSER_MAPPING: ClassVar[dict[type[BaseEnsemble], TreeParserArgs]] = {
        RandomForestClassifier: (
            TreeParserCL,
            {"use_hard_voting": False},
        ),
        AdaBoostClassifier: (
            TreeParserCL,
            {"use_hard_voting": True},
        ),
        GradientBoostingClassifier: (TreeParserRG, {}),
        LGBMClassifier: (TreeParserLGBM, {}),
        Booster: (TreeParserXGB, {}),
    }

    def __init__(self, base: BaseEnsemble, encoder: FeatureEncoder) -> None:
        self._base = base
        parser_cls, args = self.fetch_tree_parser(base=base)
        self._tree_parser = parser_cls(encoder=encoder, **args)
        self._trees = list(map(self._tree_parser.parse, self.base_trees))

    def predict(
        self,
        X: npt.ArrayLike,
        w: npt.ArrayLike,
    ) -> MClass:
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

    def __iter__(self) -> Iterator[Tree]:
        return iter(self._trees)

    def __len__(self) -> int:
        return len(self._trees)

    def __getitem__(self, index: int) -> Tree:
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
        raise NotImplementedError

    @property
    @abstractmethod
    def n_estimators(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def base_trees(self) -> list[ParsableTree]:
        raise NotImplementedError

    @abstractmethod
    def _parse_trees(self, encoder: FeatureEncoder) -> None:
        raise NotImplementedError

    @abstractmethod
    def _scores_impl(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @staticmethod
    def fetch_tree_parser(base: BaseEnsemble) -> TreeParserArgs:
        for cls, tup in EnsembleParser.TREE_PARSER_MAPPING.items():
            if isinstance(base, cls):
                return tup
        msg = f"Unknown base ensemble class: {type(base).__name__}"
        raise ValueError(msg)
