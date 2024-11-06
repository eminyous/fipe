import warnings

import numpy as np
import pytest
from numpy.typing import ArrayLike
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from utils import DATASETS, gb_skip, separate, train_sklearn

from fipe import Ensemble, FeatureEncoder, Oracle
from fipe.typing import Sample, Weights


@pytest.mark.parametrize(
    "dataset",
    [d.name for d in DATASETS.iterdir() if d.is_dir()],
)
@pytest.mark.parametrize("n_estimators", [50])
@pytest.mark.parametrize("seed", [42, 60])
@pytest.mark.parametrize(
    "model_cls, options",
    [
        (RandomForestClassifier, {"max_depth": 2}),
        (AdaBoostClassifier, {"algorithm": "SAMME"}),
        (GradientBoostingClassifier, {"max_depth": 2, "init": "zero"}),
    ],
)
class TestOracle:
    @staticmethod
    def _separate(
        new_weights: Weights,
        encoder: FeatureEncoder,
        ensemble: Ensemble,
        weights: ArrayLike,
        eps: float = 1e-6,
    ) -> tuple[list[Sample], Oracle]:
        oracle = Oracle(encoder, ensemble, weights, eps=eps)
        oracle.build()
        X = []
        separate(oracle, new_weights, X)
        return X, oracle

    def test_oracle_cannot_separate_with_all_active(
        self,
        dataset: str,
        n_estimators: int,
        seed: int,
        model_cls: type,
        options: dict[str, int | str | None],
    ) -> None:
        """Call oracle with all trees set to active."""
        gb_skip(dataset, model_cls)
        model, encoder, ensemble, weights, _ = train_sklearn(
            dataset=dataset,
            model_cls=model_cls,
            n_estimators=n_estimators,
            seed=seed,
            options=options,
        )
        weights = np.asarray(weights)
        active_weights = {t: weights[t] for t in range(len(model))}
        X, oracle = self._separate(
            active_weights,
            encoder,
            ensemble,
            weights,
            eps=1e-6,
        )
        if len(X) > 0:
            # Due to numerical precision,
            # the oracle may return points
            # but they should be very few
            # and the predictions should be the same
            X = oracle.transform(X)
            warnings.warn(
                "The oracle returned points:",
                UserWarning,
                stacklevel=1,
            )
            for item in X:
                x = item.reshape(1, -1)
                model.predict(x)
                ensemble.predict(x, weights)
                ensemble.score(x, weights)
        assert len(X) == 0

    def test_oracle_separate(
        self,
        dataset: str,
        n_estimators: int,
        seed: int,
        model_cls: type,
        options: dict[str, int | str | None],
    ) -> None:
        """Call oracle with a single tree set to inactive."""
        gb_skip(dataset, model_cls)
        model, encoder, ensemble, weights, _ = train_sklearn(
            dataset=dataset,
            model_cls=model_cls,
            n_estimators=n_estimators,
            seed=seed,
            options=options,
        )
        weights = np.asarray(weights)
        new_weights = {t: weights[t] for t in range(len(model))}
        # Change the weight of some trees
        new_weights[0] = 0
        new_weights[3] = 30
        new_weights[4] = 2
        new_weights = np.array([new_weights[t] for t in range(len(model))])
        X, oracle = self._separate(new_weights, encoder, ensemble, weights)
        assert len(X) > 0
        # Check that the new points have the same predictions
        # according to my predict and sklearn both using the initial ensemble
        X = oracle.transform(X)
        for item in X:
            x = item.reshape(1, -1)
            sk_pred = model.predict(x)
            pred = ensemble.predict(x, weights)
            new_pred = ensemble.predict(x, new_weights)
            assert not np.any(new_pred == pred)
            assert np.all(sk_pred == pred)
