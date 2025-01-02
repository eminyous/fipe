from pathlib import Path

import numpy as np
import pytest
from utils import DATASETS, train, validate_base_pred

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
        (RandomForestClassifier, {"max_depth": 5}),
        (AdaBoostClassifier, {}),
        (GradientBoostingClassifier, {"max_depth": 3, "init": "zero"}),
        (LightGBMBooster, {"max_depth": 2}),
        (XGBoostBooster, {"max_depth": 2, "base_score": 0.5}),
    ],
)
@pytest.mark.parametrize("n_estimators", [40, 80])
@pytest.mark.parametrize("seed", [42, 60])
class TestPredict:
    @staticmethod
    def test_predict_vs_base(
        dataset: Path,
        model_cls: type,
        n_estimators: int,
        seed: int,
        options: dict[str, int | str | None],
    ) -> None:
        model, _, ensemble, weights, (X_train, X_test, _, _) = train(
            dataset=dataset,
            model_cls=model_cls,
            n_estimators=n_estimators,
            seed=seed,
            options=options,
        )

        validate_base_pred(X_train, model, ensemble, weights)
        validate_base_pred(X_test, model, ensemble, weights)

    @staticmethod
    def test_predict_classes(
        dataset: Path,
        model_cls: type,
        n_estimators: int,
        seed: int,
        options: dict[str, int | str | None],
    ) -> None:
        _, _, ensemble, _, (X_train, _, _, _) = train(
            dataset=dataset,
            model_cls=model_cls,
            n_estimators=n_estimators,
            seed=seed,
            options=options,
        )

        classes = list(range(ensemble.n_classes))
        generator = np.random.default_rng()
        weights = generator.uniform(size=(len(ensemble),)) * 100
        actual_pred = ensemble.predict(X_train, weights)
        X_train = np.asarray(X_train)
        assert len(actual_pred) == len(X_train)
        assert set(actual_pred).issubset(set(classes))
