import warnings

import numpy as np
import pytest
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from utils import DATASETS, gb_skip, separate, train

from fipe import Oracle


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
    def _separate(new_weights, encoder, ensemble, weights, eps=1e-6):
        """Call the oracle and return the new points."""
        oracle = Oracle(encoder, ensemble, weights, eps=eps)
        oracle.build()
        X = []
        separate(oracle, new_weights, X)
        return X, oracle

    def test_oracle_cannot_separate_with_all_active(
        self, dataset, n_estimators, seed, model_cls, options
    ):
        """Call oracle with all trees set to active."""
        gb_skip(dataset, model_cls)
        model, encoder, ensemble, weights, _ = train(
            dataset=dataset,
            model_cls=model_cls,
            n_estimators=n_estimators,
            seed=seed,
            options=options,
        )

        active_weights = {t: weights[t] for t in range(len(model))}
        X, oracle = self._separate(
            active_weights, encoder, ensemble, weights, eps=1e-6
        )
        if len(X) > 0:
            # Due to numerical precision,
            # the oracle may return points
            # but they should be very few
            # and the predictions should be the same
            X = oracle.transform(X)
            warnings.warn("The oracle returned points:")
            for x in X:
                x = x.reshape(1, -1)
                sk_pred = model.predict(x)
                pred = ensemble.predict(x, weights)
                prob = ensemble.score(x, weights)
                print(f"It returns {x}")
                print(f"scikit-learn's prediction: {sk_pred}")
                print(f"My prediction: {pred}")
                print(f"Probabilities: {prob}")
        assert len(X) == 0

    def test_oracle_separate(
        self, dataset, n_estimators, seed, model_cls, options
    ):
        """Call oracle with a single tree set to inactive."""
        gb_skip(dataset, model_cls)
        model, encoder, ensemble, weights, _ = train(
            dataset=dataset,
            model_cls=model_cls,
            n_estimators=n_estimators,
            seed=seed,
            options=options,
        )
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
        for x in X:
            x = x.reshape(1, -1)
            sk_pred = model.predict(x)
            pred = ensemble.predict(x, weights)
            new_pred = ensemble.predict(x, new_weights)
            try:
                assert not np.any(new_pred == pred)
                assert np.all(sk_pred == pred)
            except AssertionError:
                print(f"Failed for {x}")
                print(f"scikit-learn's prediction: {sk_pred}")
                print(f"My prediction: {pred}")
                print(f"My new prediction: {new_pred}")
                prob = ensemble.score(x, weights)
                print(f"Probabilities: {prob}")
                new_prob = ensemble.score(x, new_weights)
                print(f"New Probabilities: {new_prob}")
                raise
