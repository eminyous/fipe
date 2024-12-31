from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .cl import EnsembleCL


class EnsembleRF(EnsembleCL[RandomForestClassifier, False]):
    __voting__ = False

    @property
    def _base_trees(self) -> Iterable[DecisionTreeClassifier]:
        return self._base.estimators_

    @staticmethod
    def _scores_from_proba(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(p)


CLASSES = {RandomForestClassifier: EnsembleRF}
