from pathlib import Path

import numpy as np
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

from fipe import Ensemble, FeatureEncoder, Oracle, Pruner, lgbm
from fipe.typing import Sample, Weights

ROOT = Path(__file__).parent
DATASETS = ROOT / "datasets-for-tests"


def load(dataset_name: str) -> tuple[pd.DataFrame, ArrayLike, list[str]]:
    dataset_path = DATASETS / f"{dataset_name}"
    data = pd.read_csv(dataset_path / f"{dataset_name}.full.csv")
    labels = data.iloc[:, -1:]
    y = labels.apply(lambda x: x.astype("category").cat.codes)
    y = y.to_numpy().ravel()

    data = data.iloc[:, :-1]
    with (dataset_path / f"{dataset_name}.featurelist.csv").open("r") as f:
        featurelist = f.read().split(",")[:-1]
        featurelist = f.read().split(",")[:-1]
        f.close()
    return data, y, featurelist


def train_sklearn(
    dataset: str,
    model_cls: type,
    options: dict[str, int | str | None],
    n_estimators: int = 50,
    seed: int = 42,
    test_size: float = 0.2,
) -> tuple[
    AdaBoostClassifier | GradientBoostingClassifier | RandomForestClassifier,
    FeatureEncoder,
    Ensemble,
    ArrayLike,
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
    }:
        msg = "Ensemble not supported"
        raise ValueError(msg)

    model = model_cls(n_estimators=n_estimators, random_state=seed, **options)

    model.fit(X_train, y_train)
    ensemble = Ensemble(model, encoder)

    if model_cls == AdaBoostClassifier:
        weights = model.estimator_weights_
    else:
        weights = np.ones(n_estimators)

    weights /= weights.max()
    weights *= 10000.0

    return (
        model,
        encoder,
        ensemble,
        weights,
        (X_train, X_test, y_train, y_test),
    )


def train_lgbm(
    dataset: str,
    model_cls: type,
    options: dict[str, int | str | None],
    n_estimators: int = 50,
    seed: int = 42,
    test_size: float = 0.2,
) -> tuple[
    LGBMClassifier,
    FeatureEncoder,
    lgbm.Ensemble,
    ArrayLike,
    tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
]:
    if model_cls != LGBMClassifier:
        msg = "Only LGBMClassifier supported"
        raise ValueError(msg)

    data, y, _ = load(dataset)

    encoder = FeatureEncoder(data)
    X = encoder.X.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    model = model_cls(n_estimators=n_estimators, random_state=seed, **options)
    model.fit(X_train, y_train)
    ensemble = lgbm.Ensemble(model, encoder)

    weights = np.ones(n_estimators)
    weights *= 10000.0

    return (
        model,
        encoder,
        ensemble,
        weights,
        (X_train, X_test, y_train, y_test),
    )


def separate(oracle: Oracle, weights: Weights, out: list[Sample]) -> None:
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


def gb_skip(dataset: str, model_cls: type) -> None:
    if model_cls == GradientBoostingClassifier and dataset == "Seeds":
        pytest.skip(
            "GradientBoostingClassifier not supported for more than 2 classes",
        )
