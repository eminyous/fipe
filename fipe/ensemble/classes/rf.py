import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ...feature import FeatureEncoder
from .cl import EnsembleCL


class EnsembleRF(EnsembleCL[RandomForestClassifier, False]):
    def __init__(
        self,
        base: RandomForestClassifier,
        encoder: FeatureEncoder,
    ) -> None:
        self._voting = False
        super().__init__(base=base, encoder=encoder)

    @property
    def _base_estimators(self) -> list[DecisionTreeClassifier]:
        return self._base.estimators_

    @staticmethod
    def _compute_scores_from_proba(
        p: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        return np.asarray(p)


CLASSES = {RandomForestClassifier: EnsembleRF}
