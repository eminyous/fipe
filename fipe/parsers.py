import warnings

import numpy as np

from .ensemble import Ensemble
from .feature import FeatureEncoder
from .typing import MNumber, Number


class LevelParser:
    DEFAULT_TOL = 1e-6

    _levels: dict[str, MNumber]
    _tol: float

    def __init__(self, tol: float = DEFAULT_TOL) -> None:
        self._levels = {}
        self._tol = tol

    def parse_levels(
        self,
        *ensembles: Ensemble,
        encoder: FeatureEncoder,
    ) -> None:
        for feature in encoder.continuous:
            levels = set()
            for ensemble in ensembles:
                levels |= self._get_levels(feature=feature, ensemble=ensemble)
            levels = sorted(levels)
            MIN_NUM_LEVELS = 2
            if len(levels) > MIN_NUM_LEVELS and np.any(
                np.diff(levels) < self._tol
            ):
                msg = f"Feature '{feature}' has duplicate levels"
                warnings.warn(msg, stacklevel=2)
            self._levels[feature] = np.array(levels)

    @property
    def levels(self) -> dict[str, MNumber]:
        return self._levels

    @staticmethod
    def _get_levels(feature: str, ensemble: Ensemble) -> set[Number]:
        levels = set()
        for tree in ensemble:
            tree_levels = {
                tree.threshold[node]
                for node in tree.nodes_split_on(feature=feature)
            }
            levels.update(tree_levels)
        return levels
