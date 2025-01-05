from abc import abstractmethod
from collections.abc import Generator
from typing import Generic, TypeVar, override

from ...typing import (
    DecisionTree,
    SKLearnClassifier,
    SKLearnTree,
)
from .binder import Binder

C = TypeVar("C", bound=SKLearnClassifier)
T = TypeVar("T", bound=DecisionTree)


class SKLearnBinder(Binder[C, SKLearnTree], Generic[C, T]):
    @property
    @override
    def n_classes(self) -> int:
        if not isinstance(self._base.n_classes_, int):
            msg = "n_classes must be an integer."
            raise TypeError(msg)
        return self._base.n_classes_

    @property
    @override
    def base_trees(self) -> Generator[SKLearnTree, None, None]:
        for tree in self.base_estimators:
            yield tree.tree_

    @property
    @abstractmethod
    def base_estimators(self) -> Generator[T, None, None]:
        raise NotImplementedError
