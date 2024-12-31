import numpy as np
import numpy.typing as npt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ...feature import FeatureEncoder
from .cl import EnsembleCL


class EnsembleAB(EnsembleCL[AdaBoostClassifier, True]):
    def __init__(
        self,
        base: AdaBoostClassifier,
        encoder: FeatureEncoder,
    ) -> None:
        self._voting = True
        super().__init__(base=base, encoder=encoder)

    @property
    def _base_estimators(self) -> list[DecisionTreeClassifier]:
        return list(self._base)

    @staticmethod
    def _compute_scores_from_proba(
        p: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        k = p.shape[-1]
        q = np.argmax(p, axis=-1)
        return np.eye(k)[q]


CLASSES = {AdaBoostClassifier: EnsembleAB}
