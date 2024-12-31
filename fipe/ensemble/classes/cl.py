from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ...tree import TreeCL
from ...typing import HV
from .skl import EnsembleSKL

Classifier = RandomForestClassifier | AdaBoostClassifier
CL = TypeVar("CL", bound=Classifier)


class EnsembleCL(
    EnsembleSKL[TreeCL[HV], CL, DecisionTreeClassifier], Generic[CL, HV]
):
    __metaclass__ = ABCMeta
    __tree_cls__ = TreeCL
    __voting__: HV

    @property
    def n_classes(self) -> int:
        if not isinstance(self._base.n_classes_, int):
            msg = "n_classes must be an integer."
            raise TypeError(msg)
        return self._base.n_classes_

    @property
    def n_estimators(self) -> int:
        return len(self._base)

    def _get_tree_args(self) -> dict[str, int | bool]:
        return {"voting": self.__voting__}

    def _scores_impl(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_estimators = self.n_estimators
        n_classes = self.n_classes
        scores = np.zeros((n_samples, n_estimators, n_classes))
        for j, e in enumerate(self._base_trees):
            scores[:, j, :] = self._scores_estimator(e, X)
        return scores

    def _scores_estimator(
        self,
        e: DecisionTreeClassifier,
        X: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        X = np.asarray(X)
        p = e.predict_proba(X)
        return self._scores_from_proba(p)

    @staticmethod
    @abstractmethod
    def _scores_from_proba(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
        msg = "This method must be implemented in a subclass."
        raise NotImplementedError(msg)
