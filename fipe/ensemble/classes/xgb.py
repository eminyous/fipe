from xgboost import Booster

from ...feature import FeatureEncoder
from ...tree import TreeXGB
from ..parser import EnsembleParser


class EnsembleXGB(EnsembleParser[TreeXGB, Booster]):
    TREE_KEY = "Tree"

    INDEX = (
        "Tree",
        TreeXGB.NODE_KEY,
        TreeXGB.ID_KEY,
    )

    @property
    def n_classes(self) -> int:
        n_trees = len(self._trees)
        n_base = self.n_estimators // n_trees
        n_add = 1 if self.n_estimators == n_trees else 0
        return n_base + n_add

    @property
    def n_estimators(self) -> int:
        return self._base.num_boosted_rounds()

    def _parse_trees(self, encoder: FeatureEncoder) -> None:
        model_data = self._base.trees_to_dataframe().set_index(list(self.INDEX))
        trees = []
        tree_ids = model_data.index.get_level_values(self.TREE_KEY).unique()
        for tree_id in tree_ids:
            tree_data = model_data.xs(tree_id, level=self.TREE_KEY).sort_index()
            tree = TreeXGB(tree=tree_data, encoder=encoder)
            trees.append(tree)

        self._trees = trees


CLASSES = {Booster: EnsembleXGB}
