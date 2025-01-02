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

    def _scores_impl(self, X: npt.ArrayLike) -> MProb:
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_estimators = self.n_estimators
        n_classes = self.n_classes
        scores = np.zeros((n_samples, n_estimators, n_classes))
        for j, e in enumerate(self.base_estimators):
            scores[:, j, :] = self._scores_estimator(e, X)
        return scores

    def _scores_estimator(
        self,
        e: DecisionTreeClassifier,
        X: npt.ArrayLike,
    ) -> MProb:
        X = np.asarray(X)
        p = e.predict_proba(X)
        return self._scores_proba(p)

    def _scores_proba(self, p: MProb) -> MProb:
        if self._use_hard_voting:
            k = p.shape[-1]
            q = np.argmax(p, axis=-1)
            return np.eye(k)[q]
        return p
