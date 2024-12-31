import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import ArrayLike, NDArray

from ...feature import FeatureEncoder
from ...tree import TreeLGBM
from ...typing import ParsableTreeLGBM
from ..base import BaseEnsemble


class EnsembleLGBM(BaseEnsemble[TreeLGBM, LGBMClassifier]):
    TREE_INFO_KEY = "tree_info"

    @property
    def m_valued(self) -> bool:
        return self._base.boosting_type != "rf"

    @property
    def n_classes(self) -> int:
        return self._base.n_classes_

    @property
    def n_estimators(self) -> int:
        return self._base.n_estimators_

    def _parse_trees(self, encoder: FeatureEncoder) -> None:
        model = self._base.booster_.dump_model()
        trees = model[self.TREE_INFO_KEY]

        def parse_tree(tree: ParsableTreeLGBM) -> TreeLGBM:
            return TreeLGBM(tree=tree, encoder=encoder)

        self._trees = list(map(parse_tree, trees))

    def _scores_impl(self, X: ArrayLike) -> NDArray[np.float64]:
        leaf_indices = self._base.predict_proba(X, pred_leaf=True)
        leaf_indices = np.asarray(
            leaf_indices,
            dtype=np.int32,
        )
        return self._compute_scores(leaf_indices=leaf_indices)

    def _compute_scores(
        self,
        leaf_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        n_samples = int(leaf_indices.shape[0])
        n_classes = self.n_classes
        n_estimators = self.n_estimators
        scores = np.zeros((n_samples, n_estimators, n_classes))
        for i in range(n_samples):
            scores[i] = self._compute_scores_sample(leaf_indices[i])

        return scores

    def _compute_scores_sample(
        self,
        leaf_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        n_estimators = self.n_estimators
        n_classes = self.n_classes
        scores = np.zeros((n_estimators, n_classes))
        for j in range(n_estimators):
            scores[j] = self._compute_base_scores_estimator(
                j,
                leaf_indices,
            )
        return scores

    def _compute_base_scores_estimator(
        self,
        index: int,
        leaf_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        n_classes = self.n_classes
        scores = np.zeros(n_classes)
        if n_classes == self.NUM_BINARY_CLASSES:
            leaf_index = leaf_indices[index]
            scores[1] = self[index].predict(leaf_index)
            scores[0] = -scores[1]
            return scores

        for k in range(n_classes):
            e = index * n_classes + k
            leaf_index = leaf_indices[e]
            scores[k] = self[e].predict(leaf_index)
        return scores


CLASSES = {LGBMClassifier: EnsembleLGBM}
