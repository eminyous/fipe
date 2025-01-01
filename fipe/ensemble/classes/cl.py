from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from ...typing import (
    AdaBoostClassifier,
    DecisionTreeClassifier,
    MProb,
    RandomForestClassifier,
)
from .skl import EnsembleSKL

Classifier = RandomForestClassifier | AdaBoostClassifier
CL = TypeVar("CL", bound=Classifier)


class EnsembleCL(EnsembleSKL[CL, DecisionTreeClassifier], Generic[CL]):
    __metaclass__ = ABCMeta

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
    ) -> npt.NDArray[np.float64]:
        X = np.asarray(X)
        p = e.predict_proba(X)
        return self._scores_proba(p)

    @staticmethod
    @abstractmethod
    def _scores_proba(p: MProb) -> MProb:
        raise NotImplementedError
