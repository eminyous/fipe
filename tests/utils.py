import warnings
from pathlib import Path

import gurobipy as gp
import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.model_selection import train_test_split

from fipe import FIPE, Ensemble, FeatureEncoder, Oracle, Pruner
from fipe.typing import (
    AdaBoostClassifier,
    BaseEnsemble,
    GradientBoostingClassifier,
    LightGBMBooster,
    MClass,
    MNumber,
    MProb,
    RandomForestClassifier,
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
NUM_BINARY_CLASS = 2
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
    X: npt.ArrayLike,
) -> MClass:
    if isinstance(model, XGBoostBooster):
        dx = xgb.DMatrix(X)
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
    X: npt.ArrayLike,
) -> MProb:
    if isinstance(model, XGBoostBooster):
        dx = xgb.DMatrix(X)
        return model.predict(dx)
    if isinstance(model, LightGBMBooster):
        return np.array(model.predict(X))
    return model.predict_proba(X)


def train_sklearn(
    model_cls: type[
        RandomForestClassifier | AdaBoostClassifier | GradientBoostingClassifier
    ],
    X_train: npt.ArrayLike,
    y_train: MClass,
    n_estimators: int,
    seed: int,
    **options,
) -> BaseEnsemble:
    model = model_cls(
        n_estimators=n_estimators,
        random_state=seed,
        **options,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: npt.ArrayLike,
    y_train: MClass,
    n_estimators: int,
    seed: int,
    **options,
) -> XGBoostBooster:
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        **options,
    )
    clf.fit(X_train, y_train)
    return clf.get_booster()


def train_lgbm(
    X_train: npt.ArrayLike,
    y_train: MClass,
    n_estimators: int,
    seed: int,
    **options,
) -> LightGBMBooster:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            verbose=-1,
            **options,
        )
        clf.fit(X_train, y_train)
    return clf.booster_


def train(
    dataset: Path,
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
    tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike],
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
        model = train_lgbm(
            X_train,
            y_train,
            n_estimators,
            seed,
            **options,
        )
    elif model_cls == XGBoostBooster:
        model = train_xgboost(
            X_train,
            y_train,
            n_estimators,
            seed,
            **options,
        )
    else:
        model = train_sklearn(
            model_cls,
            X_train,
            y_train,
            n_estimators,
            seed,
            **options,
        )
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
    except gp.GurobiError:
        msg = "Gurobi license is not available"
        pytest.skip(f"Skipping test: {msg}")
    except Exception:
        raise


def prune(pruner: Pruner) -> None:
    try:
        pruner.prune()
    except gp.GurobiError:
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
    assert (expected_pred == actual_pred).all()


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
