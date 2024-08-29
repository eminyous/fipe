import numpy as np
import pytest
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from utils import DATASETS, gb_skip, train


@pytest.mark.parametrize(
    "model_cls, options",
    [
        (RandomForestClassifier, {"max_depth": 5}),
        (AdaBoostClassifier, {"algorithm": "SAMME"}),
        (GradientBoostingClassifier, {"max_depth": 3, "init": "zero"}),
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
    def _compare_predicts(model, ensemble, X, weights):
        expected_pred = model.predict(X)
        actual_pred = ensemble.predict(X, weights)
        assert (expected_pred == actual_pred).all()
        assert actual_pred.shape == expected_pred.shape

    def test_predict_vs_sklearn(
        self, dataset, model_cls, n_estimators, seed, options
    ):
        gb_skip(dataset, model_cls)
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

    def test_predict_classes(
        self, dataset, model_cls, n_estimators, seed, options
    ):
        gb_skip(dataset, model_cls)
        model, _, ensemble, _, (X_train, _, _, _) = train(
            dataset=dataset,
            model_cls=model_cls,
            n_estimators=n_estimators,
            seed=seed,
            options=options,
        )

        classes = model.classes_
        weights = np.random.uniform(size=(len(ensemble),)) * 100
        actual_pred = ensemble.predict(X_train, weights)
        assert len(actual_pred) == len(X_train)
        assert set(actual_pred).issubset(set(classes))
