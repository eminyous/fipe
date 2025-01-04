from collections.abc import Generator

import numpy as np
import numpy.typing as npt

from ...typing import (
    AdaBoostClassifier,
    DecisionTreeClassifier,
    MProb,
    RandomForestClassifier,
)
from ..binder import BinderCallback
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
    def n_estimators(self) -> int:
        return len(self._base.estimators_)

    @property
    def base_estimators(self) -> Generator[DecisionTreeClassifier, None, None]:
        yield from self._base

    def _scores_impl(self, X: npt.ArrayLike, *, scores: MProb) -> None:
        for j, e in enumerate(self.base_estimators):
            self._scores_estimator(e, X, scores=scores[:, j, :])

    def _scores_estimator(
        self,
        e: DecisionTreeClassifier,
        X: npt.ArrayLike,
        *,
        scores: MProb,
    ) -> None:
        X = np.asarray(X)
        p = e.predict_proba(X)
        scores[:] = self._scores_proba(p)

    def _scores_proba(self, p: MProb) -> MProb:
        if self._use_hard_voting:
            k = p.shape[-1]
            q = np.argmax(p, axis=-1)
            return np.eye(k)[q]
        return p
