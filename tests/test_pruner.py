import numpy as np
import pytest
from numpy.typing import ArrayLike
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from utils import DATASETS, gb_skip, prune, train_sklearn

from fipe import Ensemble, Pruner


def _test_predictions(
    X: ArrayLike,
    ensemble: Ensemble,
    weights: ArrayLike,
    pruner: Pruner,
) -> None:
    pred = ensemble.predict(X, weights)
    prune_pred = pruner.predict(X)
    assert len(pruner.activated) > 0
    assert np.all(pred == prune_pred)


@pytest.mark.parametrize(
    "dataset",
    [d.name for d in DATASETS.iterdir() if d.is_dir()],
)
@pytest.mark.parametrize("norm", [0, 1])
@pytest.mark.parametrize("n_estimators", [25])
@pytest.mark.parametrize("seed", [42, 44, 46, 60])
@pytest.mark.parametrize(
    ("model_cls", "options"),
    [
        (RandomForestClassifier, {"max_depth": 1}),
        (AdaBoostClassifier, {"algorithm": "SAMME"}),
        (GradientBoostingClassifier, {"max_depth": 1, "init": "zero"}),
    ],
)
def test_pruner_norm(
    dataset: str,
    n_estimators: int,
    model_cls: type,
    options: dict[str, int | str | None],
    seed: int,
    norm: int,
) -> None:
    gb_skip(dataset, model_cls)
    model, encoder, ensemble, weights, (X_train, _, _, _) = train_sklearn(
        dataset=dataset,
        model_cls=model_cls,
        n_estimators=n_estimators,
        seed=seed,
        options=options,
    )
    # Test that pruner runs without error
    ensemble = Ensemble(model, encoder)
    pruner = Pruner(ensemble, weights, norm=norm)
    pruner.build()
    pruner.setParam("OutputFlag", 0)
    pruner.add_samples(X_train)
    prune(pruner)
    # - Check that prediction of initial and pruned ensemble are the same -
    # Check on train data
    _test_predictions(X_train, ensemble, weights, pruner)
