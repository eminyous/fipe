import numpy as np
import pytest
from lightgbm import LGBMClassifier
from numpy.typing import ArrayLike
from utils import DATASETS, gb_skip, train_lgbm

from fipe.lgbm import Ensemble


@pytest.mark.parametrize(
    "model_cls, options",
    [
        (LGBMClassifier, {"max_depth": 2}),
    ],
)
@pytest.mark.parametrize(
    "dataset",
    [d.name for d in DATASETS.iterdir() if d.is_dir()],
)
@pytest.mark.parametrize("n_estimators", [20, 50])
@pytest.mark.parametrize("seed", [42])
class TestPredict:
    @staticmethod
    def _compare_predicts(
        model: LGBMClassifier,
        ensemble: Ensemble,
        X: ArrayLike,
        weights: ArrayLike,
    ) -> None:
        expected_pred = model.predict(X)
        actual_pred = ensemble.predict(X, weights)
        assert (expected_pred == actual_pred).all()
        assert actual_pred.shape == expected_pred.shape

    def test_predict_vs_lgbm(
        self,
        dataset: str,
        model_cls: type,
        n_estimators: int,
        seed: int,
        options: dict[str, int | str | None],
    ) -> None:
        gb_skip(dataset, model_cls)
        model, _, ensemble, weights, (X_train, X_test, _, _) = train_lgbm(
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
        gb_skip(dataset, model_cls)
        model, _, ensemble, _, (X_train, _, _, _) = train_lgbm(
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
