from collections.abc import Generator

import numpy.typing as npt
import xgboost as xgb

from ...typing import XGBoostBooster, XGBoostParsableTree
from .boost import BoosterBinder


class XGBoostBinder(BoosterBinder[XGBoostBooster, XGBoostParsableTree]):
    TREE_KEY = "Tree"
    NODE_KEY = "Node"
    ID_KEY = "ID"

    INDEX = (
        TREE_KEY,
        NODE_KEY,
        ID_KEY,
    )

    __n_trees: int | None = None

    @property
    def n_trees(self) -> int:
        if self.__n_trees is None:
            trees = self._base.trees_to_dataframe()[self.TREE_KEY].unique()
            self.__n_trees = len(trees)
        return self.__n_trees

    @property
    def n_trees_per_iter(self) -> int:
        n_rounds = self._base.num_boosted_rounds()
        return self.n_trees // n_rounds

    @property
    def base_trees(self) -> Generator[XGBoostParsableTree, None, None]:
        data = self._base.trees_to_dataframe().set_index(list(self.INDEX))
        for _, tree in data.groupby(level=self.TREE_KEY):
            yield tree.reset_index(level=self.TREE_KEY, drop=True)

    @staticmethod
    def _transform(X: npt.ArrayLike) -> xgb.DMatrix:
        return xgb.DMatrix(X)
