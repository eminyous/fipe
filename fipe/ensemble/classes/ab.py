from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from .cl import EnsembleCL


class EnsembleAB(EnsembleCL[AdaBoostClassifier, True]):
    __voting__ = True

    @property
    def _base_trees(self) -> Iterable[DecisionTreeClassifier]:
        return self._base

    @staticmethod
    def _scores_from_proba(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
        k = p.shape[-1]
        q = np.argmax(p, axis=-1)
        return np.eye(k)[q]


CLASSES = {AdaBoostClassifier: EnsembleAB}
