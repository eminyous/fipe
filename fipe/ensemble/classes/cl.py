from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ...feature import FeatureEncoder
from ...tree import TreeCL
from ...typing import HV
from ..base import BaseEnsemble

Classifier = RandomForestClassifier | AdaBoostClassifier
CL = TypeVar("CL", bound=Classifier)


class EnsembleCL(BaseEnsemble[TreeCL[HV], CL], Generic[CL, HV]):
    __metaclass__ = ABCMeta

    _voting: HV

    def _parse_trees(self, encoder: FeatureEncoder) -> None:
        self._trees = []

        def parse_tree(tree: DecisionTreeClassifier) -> TreeCL:
            return TreeCL(tree=tree.tree_, encoder=encoder, voting=self._voting)

        self._trees = list(map(parse_tree, self._base_estimators))

    @property
    @abstractmethod
    def _base_estimators(self) -> list[DecisionTreeClassifier]:
        msg = "This property must be implemented in a subclass."
        raise NotImplementedError(msg)

    @property
    def n_classes(self) -> int:
        if not isinstance(self._base.n_classes_, int):
            msg = "n_classes must be an integer."
            raise TypeError(msg)
        return self._base.n_classes_

    @property
    def n_estimators(self) -> int:
        return len(self._base)

    @property
    def m_valued(self) -> bool:
        return False

    def _scores_impl(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_estimators = self.n_estimators
        n_classes = self.n_classes
        scores = np.zeros((n_samples, n_estimators, n_classes))
        for i in range(n_samples):
            scores[i] = self._compute_scores_sample(X[i])
        return scores

    def _compute_scores_sample(
        self,
        x: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        x = np.asarray(x)
        n_estimators = self.n_estimators
        n_classes = self.n_classes
        scores = np.zeros((n_estimators, n_classes))
        for i, e in enumerate(self._base_estimators):
            scores[i] = self._compute_base_scores_estimator(e, x)
        return scores

    def _compute_base_scores_estimator(
        self,
        estimator: DecisionTreeClassifier,
        x: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        x = np.asarray(x).reshape(1, -1)
        p = estimator.predict_proba(x).ravel()
        return self._compute_base_scores_estimator_from_proba(p)

    @staticmethod
    @abstractmethod
    def _compute_base_scores_estimator_from_proba(
        p: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        msg = "This method must be implemented in a subclass."
        raise NotImplementedError(msg)
