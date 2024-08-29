from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gurobipy import GurobiError
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split

from fipe import Ensemble, FeatureEncoder, Oracle, Pruner
from fipe.typing import Sample

ROOT = Path(__file__).parent
DATASETS = ROOT / "datasets-for-tests"


def load(dataset_name: str):
    dataset_path = DATASETS / f"{dataset_name}"
    data = pd.read_csv(dataset_path / f"{dataset_name}.full.csv")
    labels = data.iloc[:, -1]
    y = labels.astype("category").cat.codes
    y = y.values

    data = data.iloc[:, :-1]
    with open(
        file=dataset_path / f"{dataset_name}.featurelist.csv",
        mode="r",
        encoding="utf-8",
    ) as f:
        featurelist = f.read().split(",")[:-1]
        featurelist = f.read().split(",")[:-1]
        f.close()
    return data, y, featurelist


def train(
    dataset: str,
    model_cls,
    options: dict,
    n_estimators: int = 50,
    seed: int = 42,
    test_size: float = 0.2,
):
    data, y, _ = load(dataset)

    encoder = FeatureEncoder(data)
    X = encoder.X.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    if model_cls not in [
        RandomForestClassifier,
        AdaBoostClassifier,
        GradientBoostingClassifier,
    ]:
        raise ValueError("Ensemble not supported")

    model = model_cls(n_estimators=n_estimators, random_state=seed, **options)

    model.fit(X_train, y_train)
    ensemble = Ensemble(model, encoder)

    if model_cls == AdaBoostClassifier:
        weights = model.estimator_weights_
    else:
        weights = np.ones(n_estimators)

    weights = weights / weights.max()
    weights = weights * 1e4

    return (
        model,
        encoder,
        ensemble,
        weights,
        (X_train, X_test, y_train, y_test),
    )


def separate(oracle: Oracle, weights, out: list[Sample]):
    try:
        X = list(oracle.separate(weights))
        out.extend(X)
    except GurobiError:
        msg = "Gurobi license is not available"
        pytest.skip(f"Skipping test: {msg}")
    except Exception as e:
        raise e


def prune(pruner: Pruner):
    try:
        pruner.prune()
    except GurobiError:
        msg = "Gurobi license is not available"
        pytest.skip(f"Skipping test: {msg}")
    except Exception as e:
        raise e


def gb_skip(dataset, model_cls):
    if model_cls == GradientBoostingClassifier and dataset == "Seeds":
        pytest.skip(
            "GradientBoostingClassifier not supported for more than 2 classes"
        )
