import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from utils import DATASETS, ENV, predict, separate, train

from fipe import FeatureEncoder, Oracle
from fipe.typing import (
    AdaBoostClassifier,
    BaseEnsemble,
    GradientBoostingClassifier,
    LightGBMBooster,
    MNumber,
    RandomForestClassifier,
    SNumber,
    XGBoostBooster,
)


@pytest.mark.parametrize("dataset", DATASETS)
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
@pytest.mark.parametrize("n_estimators", [25])
@pytest.mark.parametrize("seed", [42, 60])
class TestOracle:
    @staticmethod
    def _separate(
        new_weights: MNumber,
        encoder: FeatureEncoder,
        model: BaseEnsemble,
        weights: npt.ArrayLike,
        eps: float = 1e-6,
    ) -> tuple[list[SNumber], Oracle]:
        oracle = Oracle(model, encoder, weights, env=ENV, eps=eps)
        oracle.build()
        X = []
        separate(oracle, new_weights, X)
        return X, oracle

    def test_oracle_cannot_separate_with_all_active(
        self,
        dataset: Path,
        n_estimators: int,
        seed: int,
        model_cls: type,
        options: dict[str, int | str | None],
    ) -> None:
        """Call oracle with all trees set to active."""
        model, encoder, ensemble, weights, _ = train(
            dataset=dataset,
            model_cls=model_cls,
            n_estimators=n_estimators,
            seed=seed,
            options=options,
        )
        weights = np.asarray(weights)
        active_weights = np.copy(weights)
        X, oracle = self._separate(
            active_weights,
            encoder,
            model,
            weights,
            eps=1e-6,
        )
        if len(X) > 0:
            # Due to numerical precision,
            # the oracle may return points
            # but they should be very few
            # and the predictions should be the same
            X = oracle.transform(X)
            warnings.warn(
                "The oracle returned points:",
                UserWarning,
                stacklevel=1,
            )
            for item in X:
                x = item.reshape(1, -1)
                predict(model, x)
                ensemble.predict(x, w=weights)
                ensemble.score(x, w=weights)
        assert len(X) == 0

    def test_oracle_separate(
        self,
        dataset: str,
        n_estimators: int,
        seed: int,
        model_cls: type,
        options: dict[str, int | str | None],
    ) -> None:
        """Call oracle with a single tree set to inactive."""
        model, encoder, ensemble, weights, _ = train(
            dataset=dataset,
            model_cls=model_cls,
            n_estimators=n_estimators,
            seed=seed,
            options=options,
        )
        weights = np.asarray(weights)
        new_weights = np.copy(weights)
        # Change the weight of some trees
        new_weights[0] = 0
        new_weights[3] = 30
        new_weights[4] = 2
        X, oracle = self._separate(new_weights, encoder, model, weights)
        assert len(X) > 0
        # Check that the new points have the same predictions
        # according to my predict and sklearn both using the initial ensemble
        X = oracle.transform(X)
        model_pred = predict(model, X)
        pred = ensemble.predict(X, w=weights)
        new_pred = ensemble.predict(X, w=new_weights)
        assert np.all(model_pred == pred)
        assert not np.any(pred == new_pred)
