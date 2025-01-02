from pathlib import Path

import gurobipy as gp
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from gurobipy import GurobiError
from lightgbm import LGBMClassifier
from numpy.typing import ArrayLike
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, XGBClassifier

from fipe import FIPE, Ensemble, FeatureEncoder, Oracle, Pruner
from fipe.typing import (
    BaseEnsemble,
    LightGBMBooster,
    MClass,
    MNumber,
    MProb,
    SNumber,
    XGBoostBooster,
)

ROOT = Path(__file__).parent
DATASETS_PATH = ROOT / "datasets-for-tests"
DATASETS = [
    dataset_path
    for dataset_path in DATASETS_PATH.iterdir()
    if dataset_path.is_dir()
]
ENV = gp.Env(empty=True)
ENV.setParam("OutputFlag", 0)
ENV.start()


def load(
    dataset_path: Path,
) -> tuple[pd.DataFrame, MClass, list[str]]:
    dataset_name = dataset_path.name
    data = pd.read_csv(dataset_path / f"{dataset_name}.full.csv")
    labels = data.iloc[:, -1]
    y = labels.astype("category").cat.codes.to_numpy().ravel()
    data = data.iloc[:, :-1]
    feature_path = dataset_path / f"{dataset_name}.featurelist.csv"
    with feature_path.open("r") as f:
        features = f.read().split(",")[:-1]
        f.close()
    return data, y, features


def predict(
    model: BaseEnsemble,
    X: ArrayLike,
) -> MClass:
    if isinstance(model, XGBoostBooster):
        dx = DMatrix(X)
        prob = model.predict(dx)
        if prob.ndim == 1:
            return np.array(prob - 0.5 > 0, dtype=np.intp)
        return np.argmax(prob, axis=-1)
    if isinstance(model, LightGBMBooster):
        prob = np.array(model.predict(X))
        if prob.ndim == 1:
            return np.array(prob - 0.5 > 0, dtype=np.intp)
        return np.argmax(model.predict(X), axis=-1)
    return np.array(model.predict(X))


def predict_proba(
    model: BaseEnsemble,
    X: ArrayLike,
) -> MProb:
    if isinstance(model, XGBoostBooster):
        dx = DMatrix(X)
        return model.predict(dx)
    if isinstance(model, LightGBMBooster):
        return np.array(model.predict(X))
    return model.predict_proba(X)


def train(
    dataset: str,
    model_cls: type[BaseEnsemble],
    options: dict[str, int | str | None],
    n_estimators: int = 50,
    seed: int = 42,
    test_size: float = 0.2,
) -> tuple[
    BaseEnsemble,
    FeatureEncoder,
    Ensemble,
    MNumber,
    tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
]:
    data, y, _ = load(dataset)

    encoder = FeatureEncoder(data)
    X = encoder.X.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    if model_cls not in {
        RandomForestClassifier,
        AdaBoostClassifier,
        GradientBoostingClassifier,
        LightGBMBooster,
        XGBoostBooster,
    }:
        msg = "Ensemble not supported"
        raise ValueError(msg)

    if model_cls == LightGBMBooster:
        cls = LGBMClassifier
    elif model_cls == XGBoostBooster:
        cls = XGBClassifier
    else:
        cls = model_cls

    if cls == LGBMClassifier:
        options["verbose"] = -1

    clf = cls(
        n_estimators=n_estimators,
        random_state=seed,
        **options,
    )
    clf.fit(X_train, y_train)

    if isinstance(clf, LGBMClassifier):
        model = clf.booster_
    elif isinstance(clf, XGBClassifier):
        model = clf.get_booster()
    else:
        model = clf

    ensemble = Ensemble(base=model, encoder=encoder)

    if isinstance(model, AdaBoostClassifier):
        weights = model.estimator_weights_
    else:
        weights = np.ones(n_estimators)

    weights /= weights.max()
    weights *= 1e5

    return (
        model,
        encoder,
        ensemble,
        weights,
        (X_train, X_test, y_train, y_test),
    )


def separate(oracle: Oracle, weights: MNumber, out: list[SNumber]) -> None:
    try:
        X = list(oracle.separate(weights))
        out.extend(X)
    except GurobiError:
        msg = "Gurobi license is not available"
        pytest.skip(f"Skipping test: {msg}")
    except Exception:
        raise


def prune(pruner: Pruner) -> None:
    try:
        pruner.prune()
    except GurobiError:
        msg = "Gurobi license is not available"
        pytest.skip(f"Skipping test: {msg}")
    except Exception:
        raise


def validate_prune_pred(
    X: npt.ArrayLike,
    pruner: Pruner,
    weights: npt.ArrayLike,
) -> None:
    ensemble = pruner.ensemble
    pred = ensemble.predict(X, weights)
    pruner_pred = pruner.predict(X)
    assert np.all(pred == pruner_pred)


def validate_base_pred(
    X: npt.ArrayLike,
    model: BaseEnsemble,
    ensemble: Ensemble,
    weights: npt.ArrayLike,
) -> None:
    expected_pred = predict(model, X)
    actual_pred = ensemble.predict(X, weights)
    assert actual_pred.shape == expected_pred.shape
    diff = np.abs(actual_pred - expected_pred) > 0

    assert (expected_pred == actual_pred).all(), (
        actual_pred[diff],
        expected_pred[diff],
        ensemble.score(X, weights)[diff],
        predict_proba(model, X)[diff],
    )


def validate_fidelity(
    model: BaseEnsemble,
    pruner: FIPE,
    weights: npt.ArrayLike,
) -> None:
    ensemble = pruner.ensemble
    pruner_weights = pruner.weights
    for xd in pruner.counter_factuals:
        x = pruner.transform(xd)
        model_pred = predict(model, x)
        pred = ensemble.predict(x, weights)
        pruner_pred = pruner.predict(x)
        try:
            assert np.all(pruner_pred == pred)
            assert np.all(model_pred == pred)
        except AssertionError:
            ensemble.score(x, weights)
            ensemble.score(x, pruner_weights)
            raise
