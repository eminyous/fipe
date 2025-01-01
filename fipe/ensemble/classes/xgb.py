from ...tree import XGBoostTreeParser
from ...typing import Booster, XGBoostParsableTree
from ..generic import GenericEnsemble


class XGBoostBinder(GenericEnsemble[Booster, XGBoostParsableTree]):
    TREE_KEY = "Tree"

    INDEX = (
        "Tree",
        XGBoostTreeParser.NODE_KEY,
        XGBoostTreeParser.ID_KEY,
    )

    @property
    def n_classes(self) -> int:
        n_trees = len(list(self.base_trees))
        return (n_trees // self.n_estimators) + int(
            self.n_estimators == n_trees
        )

    @property
    def n_estimators(self) -> int:
        return self._base.num_boosted_rounds()
