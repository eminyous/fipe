import warnings
from functools import partial

import numpy as np

from ..ensemble import Ensemble
from ..feature import FeatureEncoder
from ..tree import Tree
from ..typing import MNumber, Number


class LevelParser:
    DEFAULT_TOL = 1e-6
    MIN_NUM_LEVELS = 2

    _tol: float

    def __init__(self, tol: float = DEFAULT_TOL) -> None:
        self._tol = tol

    def parse_levels(
        self,
        *ensembles: Ensemble,
        encoder: FeatureEncoder,
    ) -> dict[str, MNumber]:
        return {
            feature: self._parse_levels(*ensembles, feature=feature)
            for feature in encoder.continuous
        }

    def _parse_levels(
        self,
        *ensembles: Ensemble,
        feature: str,
    ) -> MNumber:
        f = partial(self._get_levels, feature=feature)
        levels = set[Number]().union(*map(f, ensembles))
        levels = sorted(levels)
        if len(levels) > self.MIN_NUM_LEVELS and np.any(
            np.diff(levels) < self._tol
        ):
            msg = f"Feature '{feature}' has duplicate levels"
            warnings.warn(msg, stacklevel=2)
        return np.array(levels)

    @staticmethod
    def _get_levels(ensemble: Ensemble, feature: str) -> set[Number]:
        f = partial(LevelParser._get_tree_levels, feature=feature)
        return set[Number]().union(*map(f, ensemble))

    @staticmethod
    def _get_tree_levels(tree: Tree, feature: str) -> set[Number]:
        return {
            tree.threshold[node]
            for node in tree.nodes_split_on(feature=feature)
        }
