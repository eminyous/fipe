from abc import abstractmethod
from collections.abc import Generator
from typing import Generic, TypeVar, override

import numpy as np
import numpy.typing as npt

from ...typing import (
    AdaBoostClassifier,
    DecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    GradientBoostingClassifier,
    MProb,
    RandomForestClassifier,
    SKLearnClassifier,
    SKLearnTree,
)
from .binder import Binder, BinderCallback

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


Classifier = RandomForestClassifier | AdaBoostClassifier


class SKLearnBinderClassifier(
    SKLearnBinder[Classifier, DecisionTreeClassifier]
):
    _use_hard_voting: bool

    def __init__(
        self,
        base: Classifier,
        *,
        callback: BinderCallback,
        use_hard_voting: bool,
    ) -> None:
        super().__init__(base=base, callback=callback)
        self._use_hard_voting = use_hard_voting

    @property
    @override
    def n_estimators(self) -> int:
        return len(self._base.estimators_)

    @property
    @override
    def base_estimators(self) -> Generator[DecisionTreeClassifier, None, None]:
        yield from self._base

    @override
    def _predict_proba_impl(self, X: npt.ArrayLike, *, probs: MProb) -> None:
        for i, e in enumerate(self.base_estimators):
            self._predict_proba_e(e=e, X=X, probs=probs[:, i, :])

    def _predict_proba_e(
        self,
        e: DecisionTreeClassifier,
        X: npt.ArrayLike,
        *,
        probs: MProb,
    ) -> None:
        X = np.asarray(X)
        prob = np.asarray(e.predict_proba(X))
        probs[:] = self._cast_proba(prob)

    def _cast_proba(self, prob: MProb) -> MProb:
        if self._use_hard_voting:
            k = prob.shape[-1]
            q = np.argmax(prob, axis=-1)
            return np.eye(k)[q]
        return prob


class GradientBoostingBinder(
    SKLearnBinder[GradientBoostingClassifier, DecisionTreeRegressor],
):
    @property
    @override
    def n_classes(self) -> int:
        return self._base.n_classes_

    @property
    @override
    def n_estimators(self) -> int:
        return self._base.n_estimators_

    @property
    @override
    def base_estimators(self) -> Generator[DecisionTreeRegressor, None, None]:
        yield from self._base.estimators_.ravel()

    @override
    def _predict_proba_impl(self, X: npt.ArrayLike, *, probs: MProb) -> None:
        for e in range(self.n_estimators):
            self._predict_proba_e(e, X, scores=probs[:, e, :])

    def _predict_proba_e(
        self,
        e: int,
        X: npt.ArrayLike,
        *,
        scores: MProb,
    ) -> None:
        if self.is_binary:
            scores[:, 1] = self._base.estimators_[e, 0].predict(X)
            scores[:, 0] = -scores[:, 1]
            return

        n_classes = self.n_classes
        for k in range(n_classes):
            scores[:, k] = self._base.estimators_[e, k].predict(X)
