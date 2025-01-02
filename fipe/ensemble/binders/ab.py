import numpy as np

from ...typing import AdaBoostClassifier, MProb
from .cl import EnsembleBinderClassifier


class AdaBoostBinder(EnsembleBinderClassifier[AdaBoostClassifier]):
    @staticmethod
    def _scores_proba(p: MProb) -> MProb:
        k = int(p.shape[-1])
        q = np.argmax(p, axis=-1)
        return np.eye(k)[q]
