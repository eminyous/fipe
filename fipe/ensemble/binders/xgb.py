from collections.abc import Generator

import numpy.typing as npt
import xgboost as xgb

from ...tree import XGBoostTreeParser
from ...typing import XGBoostBooster, XGBoostParsableTree
from .boost import BoosterBinder


class XGBoostBinder(BoosterBinder[XGBoostBooster, XGBoostParsableTree]):
    TREE_KEY = "Tree"

    INDEX = (
        TREE_KEY,
        XGBoostTreeParser.NODE_KEY,
        XGBoostTreeParser.ID_KEY,
    )

    __n_trees: int | None = None

    @property
    def n_trees(self) -> int:
        if self.__n_trees is None:
            trees = self._base.trees_to_dataframe()[self.TREE_KEY].unique()
            self.__n_trees = len(trees)
        return self.__n_trees

    @property
    def n_classes(self) -> int:
        n_trees = self.n_trees
        return (n_trees // self.n_estimators) + int(
            self.n_estimators == n_trees
        )

    @property
    def n_estimators(self) -> int:
        return self._base.num_boosted_rounds()

    @property
    def base_trees(self) -> Generator[XGBoostParsableTree, None, None]:
        data = self._base.trees_to_dataframe().set_index(list(self.INDEX))
        for _, tree in data.groupby(level=self.TREE_KEY):
            yield tree.reset_index(level=self.TREE_KEY, drop=True)

    @staticmethod
    def _transform(X: npt.ArrayLike) -> xgb.DMatrix:
        return xgb.DMatrix(X)
