from collections.abc import Generator
from typing import override

import numpy as np
import numpy.typing as npt

from ...typing import (
    AdaBoostClassifier,
    DecisionTreeClassifier,
    MProb,
    RandomForestClassifier,
)
from .binder import BinderCallback
from .skl import SKLearnBinder

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
