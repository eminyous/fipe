from ...tree import TreeParserXGB
from ...typing import Booster, ParsableTreeXGB
from ..generic import GenericEnsemble


class EnsembleXGB(GenericEnsemble[Booster, ParsableTreeXGB]):
    TREE_KEY = "Tree"

    INDEX = (
        "Tree",
        TreeParserXGB.NODE_KEY,
        TreeParserXGB.ID_KEY,
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
