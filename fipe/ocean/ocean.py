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

    def _set_pruned_model_weights(self, weights):
        self._new_weights = {}
        for t in range(self.n_estimators):
            try:
                self._new_weights[t] = weights[t]
            except (KeyError, IndexError):
                self._new_weights[t] = 0.0

    def _check_counter_factual(self, x: Sample):
        pass
