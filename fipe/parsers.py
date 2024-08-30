import warnings

import numpy as np

from .ensemble import Ensemble
from .feature.encoder import FeatureEncoder
from .typing import numeric


class LevelParser:
    _levels: dict[str, list[numeric]]
    _tol: float

    def __init__(self, **kwargs) -> None:
        self._levels = {}
        self._tol = kwargs.get("tol", 1e-6)

    def parse_levels(
        self,
        ensembles: list[Ensemble],
        encoder: FeatureEncoder,
    ) -> None:
        for feature in encoder.continuous:
            levels = set()
            for ensemble in ensembles:
                levels |= self._get_levels(feature, ensemble)
            levels = sorted(levels)
            MIN_NUM_LEVELS = 2
            if (
                len(levels) >= MIN_NUM_LEVELS
                and np.diff(levels).min() < self._tol
            ):
                msg = (
                    f"The levels of the feature {feature}"
                    " are too close to each other."
                )
                warnings.warn(msg, stacklevel=2)
            self._levels[feature] = levels

    @property
    def levels(self) -> dict[str, list[numeric]]:
        return self._levels

    @staticmethod
    def _get_levels(feature: str, ensemble: Ensemble) -> set[numeric]:
        levels = set()
        for tree in ensemble:
            tree_levels = {
                tree.threshold[n] for n in tree.nodes_split_on(feature)
            }
            levels.update(tree_levels)
        return levels
