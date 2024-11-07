import numpy as np
from numpy.typing import NDArray

from ...feature import FeatureEncoder
from ..ensemble import Ensemble
from .vote import VoteOCEAN


class OCEAN(VoteOCEAN):
    _new_weights: NDArray[np.float64]

    def __init__(
        self,
        encoder: FeatureEncoder,
        ensemble: Ensemble,
        weights: NDArray[np.float64],
        **kwargs,
    ) -> None:
        VoteOCEAN.__init__(
            self,
            encoder=encoder,
            ensemble=ensemble,
            weights=weights,
            **kwargs,
        )

    def _set_new_weights(self, new_weights: NDArray[np.float64]) -> None:
        self._new_weights = np.copy(new_weights)
