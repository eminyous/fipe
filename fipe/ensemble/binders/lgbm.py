from collections.abc import Generator
from typing import override

from ...typing import LightGBMBooster, LightGBMParsableTree
from .boost import BoosterBinder


class LightGBMBinder(BoosterBinder[LightGBMBooster, LightGBMParsableTree]):
    TREE_INFO_KEY = "tree_info"

    @property
    @override
    def n_trees(self) -> int:
        return self._base.num_trees()

    @property
    @override
    def n_trees_per_iter(self) -> int:
        return self._base.num_model_per_iteration()

    @property
    @override
    def base_trees(self) -> Generator[LightGBMParsableTree, None, None]:
        model = self._base.dump_model()
        yield from model[self.TREE_INFO_KEY]
