from itertools import chain

import numpy as np
import pytest
from lightgbm import LGBMClassifier
from numpy.typing import ArrayLike
from utils import DATASETS, gb_skip, prune, train_lgbm

from fipe.lgbm import FIPE


def _test_predictions(X: ArrayLike, pruner: FIPE, weights: ArrayLike) -> None:
    ensemble = pruner.ensemble
    pred = ensemble.predict(X, weights)
    pruner_pred = pruner.predict(X)
    idx = np.where(pred != pruner_pred)[0]
    if len(idx) == 0:
        return
    assert np.all(pred == pruner_pred)


def _test_fidelity(
    model: LGBMClassifier,
    pruner: FIPE,
    weights: ArrayLike,
) -> None:
    ensemble = pruner.ensemble
    pruner_weights = pruner.weights
    for xd in chain(*pruner.counter_factuals):
        x = pruner.transform(xd)
        x = x.reshape(1, -1)
        lgbm_pred = model.predict(x)
        pred = ensemble.predict(x, weights)
        pruner_pred = pruner.predict(x)
        try:
            assert np.all(pruner_pred == pred)
            assert np.all(lgbm_pred == pred)
        except AssertionError:
            # Show score prediction
            ensemble.score(x, weights)
            ensemble.score(x, pruner_weights)
            raise


@pytest.mark.parametrize(
    "dataset",
    [d.name for d in DATASETS.iterdir() if d.is_dir()],
)
@pytest.mark.parametrize("n_estimators", [10])
@pytest.mark.parametrize("seed", [41])
@pytest.mark.parametrize(
    ("model_cls", "options"),
    [
        (LGBMClassifier, {"max_depth": 2}),
    ],
)
@pytest.mark.parametrize("norm", [1])
def test_prune(
    dataset: str,
    n_estimators: int,
    seed: int,
    model_cls: type,
    options: dict[str, int | str | None],
    norm: int,
) -> None:
    gb_skip(dataset, model_cls)
    model, encoder, _, weights, (X_train, X_test, _, _) = train_lgbm(
        dataset=dataset,
        model_cls=model_cls,
        n_estimators=n_estimators,
        seed=seed,
        options=options,
    )

    # Test that FIPE pruning runs without error
    pruner = FIPE(model, weights, encoder, norm=norm, eps=1e-6)
    pruner.build()
    pruner.setParam("OutputFlag", 0)
    pruner.oracle.setParam("OutputFlag", 0)
    pruner.add_samples(X_train)
    prune(pruner)

    active_estimators = pruner.active_estimators
    assert len(active_estimators) > 0

    _test_predictions(X_train, pruner, weights)
    _test_predictions(X_test, pruner, weights)

    # - Test that FIPE has 100% fidelity -
    #    Verify fidelity on new points
    _test_fidelity(model, pruner, weights)
