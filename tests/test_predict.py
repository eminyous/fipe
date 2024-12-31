import numpy as np
import numpy.typing as npt
import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from utils import DATASETS, train

from fipe import Ensemble


@pytest.mark.parametrize(
    ("model_cls", "options"),
    [
        (RandomForestClassifier, {"max_depth": 5}),
        (AdaBoostClassifier, {"algorithm": "SAMME"}),
        (GradientBoostingClassifier, {"max_depth": 3, "init": "zero"}),
        (LGBMClassifier, {"max_depth": 2}),
    ],
)
@pytest.mark.parametrize(
    "dataset",
    [d.name for d in DATASETS.iterdir() if d.is_dir()],
)
@pytest.mark.parametrize("n_estimators", [40, 80])
@pytest.mark.parametrize("seed", [42, 60])
class TestPredict:
    @staticmethod
    def _compare_predicts(
        model: (
            AdaBoostClassifier
            | GradientBoostingClassifier
            | RandomForestClassifier
            | LGBMClassifier
        ),
        ensemble: Ensemble,
        X: npt.ArrayLike,
        weights: npt.ArrayLike,
    ) -> None:
        expected_pred = model.predict(X)
        actual_pred = ensemble.predict(X, weights)
        assert (expected_pred == actual_pred).all()
        assert actual_pred.shape == expected_pred.shape

    def test_predict_vs_sklearn(
        self,
        dataset: str,
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

        # Test on training data
        self._compare_predicts(model, ensemble, X_train, weights)
        # Test on test data
        self._compare_predicts(model, ensemble, X_test, weights)

    @staticmethod
    def test_predict_classes(
        dataset: str,
        model_cls: type,
        n_estimators: int,
        seed: int,
        options: dict[str, int | str | None],
    ) -> None:
        model, _, ensemble, _, (X_train, _, _, _) = train(
            dataset=dataset,
            model_cls=model_cls,
            n_estimators=n_estimators,
            seed=seed,
            options=options,
        )

        classes = model.classes_
        generator = np.random.default_rng()
        weights = generator.uniform(size=(len(ensemble),)) * 100
        actual_pred = ensemble.predict(X_train, weights)
        X_train = np.asarray(X_train)
        assert len(actual_pred) == len(X_train)
        assert set(actual_pred).issubset(set(classes))
