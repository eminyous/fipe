from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from typing import Generic, TypeVar

from ...typing import (
    AdaBoostClassifier,
    DecisionTree,
    GradientBoostingClassifier,
    ParsableTreeSKL,
    RandomForestClassifier,
)
from ..generic import GenericEnsemble

Classifier = (
    RandomForestClassifier | AdaBoostClassifier | GradientBoostingClassifier
)

CL = TypeVar("CL", bound=Classifier)
DT = TypeVar("DT", bound=DecisionTree)


class EnsembleSKL(GenericEnsemble[CL, ParsableTreeSKL], Generic[CL, DT]):
    __metaclass__ = ABCMeta

    @property
    def n_classes(self) -> int:
        if not isinstance(self._base.n_classes_, int):
            msg = "n_classes must be an integer."
            raise TypeError(msg)
        return self._base.n_classes_

    @property
    def base_trees(self) -> Generator[ParsableTreeSKL, None, None]:
        for tree in self.base_estimators:
            yield tree.tree_

    @property
    @abstractmethod
    def base_estimators(self) -> Generator[DT, None, None]:
        raise NotImplementedError