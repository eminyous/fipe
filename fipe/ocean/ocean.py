from ..ensemble import Ensemble
from ..feature.encoder import FeatureEncoder
from ..typing import Sample
from .vote import VoteOCEAN


class OCEAN(VoteOCEAN):
    _new_weights: dict[int, float]

    def __init__(
        self, encoder: FeatureEncoder, ensemble: Ensemble, weights, **kwargs
    ):
        VoteOCEAN.__init__(self, encoder, ensemble, weights, **kwargs)

    def _set_new_weights(self, weights):
        self._new_weights = self._to_dict(weights)

    def _check_counter_factual(self, x: Sample):
        pass
