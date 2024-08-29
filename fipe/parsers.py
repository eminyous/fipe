import warnings

import numpy as np

from .ensemble import Ensemble
from .feature.encoder import FeatureEncoder
from .typing import numeric


class LevelParser:
    _levels: dict[str, list[numeric]]
    _tol: float

    def __init__(self, **kwargs):
        self._levels = {}
        self._tol = kwargs.get("tol", 1e-6)

    def parse_levels(self, ensembles: list[Ensemble], encoder: FeatureEncoder):
        for feature in encoder.continuous:
            levels = set()
            for ensemble in ensembles:
                levels |= self._get_levels(feature, ensemble)
            levels = list(sorted(levels))
            if len(levels) >= 2 and np.diff(levels).min() < self._tol:
                msg = (
                    f"The levels of the feature {feature}"
                    " are too close to each other."
                )
                warnings.warn(msg)
            self._levels[feature] = levels

    @property
    def levels(self):
        return self._levels

    def _get_levels(self, feature: str, ensemble: Ensemble):
        levels = set()
        for tree in ensemble:
            for n in tree.nodes_split_on(feature):
                levels.add(tree.threshold[n])
        return levels
