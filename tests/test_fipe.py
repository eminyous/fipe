from itertools import chain

import numpy as np
import pytest
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from utils import DATASETS, gb_skip, prune, train

from fipe import FIPE


def _test_predictions(X, pruner, weights):
    """
    Check that the predictions of the
    initial and pruned ensemble are the same.
    """
    ensemble = pruner.ensemble
    pruner_weights = np.array([pruner.weights[t] for t in range(len(ensemble))])
    pred = ensemble.predict(X, weights)
    pruner_pred = ensemble.predict(X, pruner_weights)
    assert np.all(pred == pruner_pred)


def _test_fidelity(model, pruner, weights):
    """
    Check that the pruned ensemble has 100% fidelity
    on the training data.
    """
    ensemble = pruner.ensemble
    pruner_weights = np.array([pruner.weights[t] for t in range(len(ensemble))])
    for xd in chain(*pruner.counterfactuals):
        x = pruner.transform(xd)
        x = x.reshape(1, -1)
        sk_pred = model.predict(x)
        pred = ensemble.predict(x, weights)
        pruner_pred = ensemble.predict(x, pruner_weights)
        try:
            assert np.all(pruner_pred == pred)
            assert np.all(sk_pred == pred)
        except AssertionError:
            print(f"Failed for {x}")
            print(f"scikit-learn's prediction: {sk_pred}")
            print(f"My prediction: {pred}")
            print(f"My new prediction: {pruner_pred}")
            # Show score prediction
            prob = ensemble.score(x, weights)
            print(f"Probabilities: {prob}")
            pruner_prob = ensemble.score(x, pruner_weights)
            print(f"New Probabilities: {pruner_prob}")
            raise


@pytest.mark.parametrize(
    "dataset",
    [d.name for d in DATASETS.iterdir() if d.is_dir()],
)
@pytest.mark.parametrize("n_estimators", [10])
@pytest.mark.parametrize("seed", [41, 56, 78])
@pytest.mark.parametrize(
    "model_cls, options",
    [
        (RandomForestClassifier, {"max_depth": 1}),
        (AdaBoostClassifier, {"algorithm": "SAMME"}),
        (GradientBoostingClassifier, {"max_depth": 1, "init": "zero"}),
    ],
)
@pytest.mark.parametrize("norm", [0, 1])
def test_prune(dataset, n_estimators, model_cls, options, seed, norm):
    gb_skip(dataset, model_cls)
    model, encoder, _, weights, (X_train, X_test, _, _) = train(
        dataset=dataset,
        model_cls=model_cls,
        n_estimators=n_estimators,
        seed=seed,
        options=options,
    )

    # Test that FIPE pruning runs without error
    pruner = FIPE(model, weights, encoder, eps=1e-4)
    pruner.build()
    pruner.setParam("OutputFlag", 0)
    pruner.oracle.setParam("OutputFlag", 0)
    pruner.set_norm(norm=norm)
    pruner.add_samples(X_train)
    prune(pruner)

    activated = pruner.activated
    assert len(activated) > 0

    _test_predictions(X_test, pruner, weights)

    # - Test that FIPE has 100% fidelity -
    #    Verify fidelity on new points
    _test_fidelity(model, pruner, weights)
