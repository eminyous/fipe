import numpy as np
import numpy.typing as npt

from ...typing import RandomForestClassifier
from .cl import EnsembleCL


class EnsembleRF(EnsembleCL[RandomForestClassifier]):
    @staticmethod
    def _scores_proba(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(p)
