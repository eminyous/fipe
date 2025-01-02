from collections.abc import Generator

import numpy.typing as npt

from ...typing import LightGBMBooster, LightGBMParsableTree
from .boost import BoosterBinder


class LightGBMBinder(BoosterBinder[LightGBMBooster, LightGBMParsableTree]):
    TREE_INFO_KEY = "tree_info"

    @property
    def n_classes(self) -> int:
        n_per_iter = self._base.num_model_per_iteration()
        return n_per_iter + int(n_per_iter == 1)

    @property
    def n_estimators(self) -> int:
        n_trees = self._base.num_trees()
        n_per_iter = self._base.num_model_per_iteration()
        return n_trees // n_per_iter

    @property
    def base_trees(self) -> Generator[LightGBMParsableTree, None, None]:
        model = self._base.dump_model()
        yield from model[self.TREE_INFO_KEY]

    @staticmethod
    def _transform(X: npt.ArrayLike) -> npt.ArrayLike:
        return X
