from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from functools import partial
from typing import ClassVar, Generic, TypeVar

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ...feature import FeatureEncoder
from ...tree import TreeSKL
from ..parser import EnsembleParser

Classifier = (
    RandomForestClassifier | AdaBoostClassifier | GradientBoostingClassifier
)
CL = TypeVar("CL", bound=Classifier)
TP = TypeVar("TP", bound=TreeSKL)
DT = TypeVar("DT", bound=DecisionTreeClassifier | DecisionTreeRegressor)


class EnsembleSKL(EnsembleParser[TP, CL], Generic[TP, CL, DT]):
    __metaclass__ = ABCMeta

    DEFAULT_TREE_ARGS: ClassVar[dict] = {}
    __tree_cls__: type[TP]

    def _parse_trees(self, encoder: FeatureEncoder) -> None:
        pt = partial(self._parse_tree, encoder=encoder)
        self._trees = list(map(pt, self._base_trees))

    @property
    @abstractmethod
    def _base_trees(self) -> Iterable[DT]:
        msg = "This property must be implemented in a subclass."
        raise NotImplementedError(msg)

    @property
    def n_classes(self) -> int:
        if not isinstance(self._base.n_classes_, int):
            msg = "n_classes must be an integer."
            raise TypeError(msg)
        return self._base.n_classes_

    def _get_tree_args(self) -> dict[str, int | bool]:
        return self.DEFAULT_TREE_ARGS

    def _parse_tree(self, tree: DT, encoder: FeatureEncoder) -> TP:
        args = self._get_tree_args()
        return self.__tree_cls__(tree=tree.tree_, encoder=encoder, **args)
