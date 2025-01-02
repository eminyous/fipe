from pathlib import Path

import pytest
from utils import (
    DATASETS,
    ENV,
    prune,
    train,
    validate_fidelity,
    validate_prune_pred,
)

from fipe import FIPE
from fipe.typing import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    LightGBMBooster,
    RandomForestClassifier,
    XGBoostBooster,
)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize(
    ("model_cls", "options"),
    [
        (RandomForestClassifier, {"max_depth": 1}),
        (AdaBoostClassifier, {}),
        (GradientBoostingClassifier, {"max_depth": 1, "init": "zero"}),
        (LightGBMBooster, {"max_depth": 1}),
        (XGBoostBooster, {"max_depth": 1, "base_score": 0.5}),
    ],
)
@pytest.mark.parametrize("n_estimators", [10])
@pytest.mark.parametrize("seed", [41, 56])
@pytest.mark.parametrize("norm", [0, 1])
def test_prune(
    dataset: Path,
    n_estimators: int,
    seed: int,
    model_cls: type,
    options: dict[str, int | str | None],
    norm: int,
) -> None:
    model, encoder, _, weights, (X_train, X_test, _, _) = train(
        dataset=dataset,
        model_cls=model_cls,
        n_estimators=n_estimators,
        seed=seed,
        options=options,
    )

    # Test that FIPE pruning runs without error
    pruner = FIPE(
        model,
        encoder,
        weights,
        norm=norm,
        env=ENV,
        eps=1e-6,
        tol=1e-4,
    )
    pruner.build()
    pruner.add_samples(X_train)
    prune(pruner)

    assert pruner.n_active_estimators > 0
    validate_prune_pred(X_test, pruner, weights)

    # - Test that FIPE has 100% fidelity -
    #    Verify fidelity on new points
    validate_fidelity(model, pruner, weights)
