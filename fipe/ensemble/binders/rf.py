import numpy as np
import numpy.typing as npt

from ...typing import RandomForestClassifier
from .cl import SKLearnBinderClassifier


class RandomForestBinder(SKLearnBinderClassifier[RandomForestClassifier]):
    @staticmethod
    def _scores_proba(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(p)
