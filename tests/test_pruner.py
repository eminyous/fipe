from pathlib import Path

import pytest
from utils import DATASETS, ENV, prune, train, validate_prune_pred

from fipe import Pruner
from fipe.typing import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    LightGBMBooster,
    RandomForestClassifier,
    XGBoostBooster,
)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("norm", [0, 1])
@pytest.mark.parametrize("n_estimators", [25])
@pytest.mark.parametrize("seed", [42, 44])
@pytest.mark.parametrize(
    ("model_cls", "options"),
    [
        (RandomForestClassifier, {"max_depth": 2}),
        (AdaBoostClassifier, {}),
        (GradientBoostingClassifier, {"max_depth": 2, "init": "zero"}),
        (LightGBMBooster, {"max_depth": 2}),
        (XGBoostBooster, {"max_depth": 2, "base_score": 0.5}),
    ],
)
def test_pruner_norm(
    dataset: Path,
    n_estimators: int,
    model_cls: type,
    options: dict[str, int | str | None],
    seed: int,
    norm: int,
) -> None:
    model, encoder, _, weights, (X_train, _, _, _) = train(
        dataset=dataset,
        model_cls=model_cls,
        n_estimators=n_estimators,
        seed=seed,
        options=options,
    )
    # Test that pruner runs without error
    pruner = Pruner(model, encoder, weights, norm=norm, env=ENV)
    pruner.build()
    pruner.add_samples(X_train)
    prune(pruner)
    # - Check that prediction of initial and pruned ensemble are the same -
    # Check on train data
    assert pruner.n_active_estimators > 0
    validate_prune_pred(X_train, pruner, weights)
