from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from typing import Generic, TypeVar, override

from ...typing import (
    BaseDecisionTree,
    SKLearnClassifier,
    SKLearnParsableTree,
)
from .generic import GenericBinder

CL = TypeVar("CL", bound=SKLearnClassifier)
DT = TypeVar("DT", bound=BaseDecisionTree)


class SKLearnBinder(GenericBinder[CL, SKLearnParsableTree], Generic[CL, DT]):
    __metaclass__ = ABCMeta

    @property
    @override
    def n_classes(self) -> int:
        if not isinstance(self._base.n_classes_, int):
            msg = "n_classes must be an integer."
            raise TypeError(msg)
        return self._base.n_classes_

    @property
    @override
    def base_trees(self) -> Generator[SKLearnParsableTree, None, None]:
        for tree in self.base_estimators:
            yield tree.tree_

    @property
    @abstractmethod
    def base_estimators(self) -> Generator[DT, None, None]:
        raise NotImplementedError
