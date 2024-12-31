import warnings

import numpy as np
import numpy.typing as npt
import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from utils import DATASETS, ENV, separate, train

from fipe import FeatureEncoder, Oracle
from fipe.typing import BaseEnsemble, MNumber, SNumber


@pytest.mark.parametrize(
    "dataset",
    [d.name for d in DATASETS.iterdir() if d.is_dir()],
)
@pytest.mark.parametrize("n_estimators", [50])
@pytest.mark.parametrize("seed", [42, 60])
@pytest.mark.parametrize(
    ("model_cls", "options"),
    [
        (RandomForestClassifier, {"max_depth": 2}),
        (AdaBoostClassifier, {"algorithm": "SAMME"}),
        (GradientBoostingClassifier, {"max_depth": 2, "init": "zero"}),
        (LGBMClassifier, {"max_depth": 2}),
    ],
)
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
        dataset: str,
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
                model.predict(x)
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
        for item in X:
            x = item.reshape(1, -1)
            model_pred = model.predict(x)
            pred = ensemble.predict(x, w=weights)
            new_pred = ensemble.predict(x, w=new_weights)
            assert not np.any(new_pred == pred)
            assert np.all(model_pred == pred)
