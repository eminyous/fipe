from collections.abc import Mapping
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import Booster as LightGBMBooster
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import Tree  # noqa: PLC2701
from xgboost import Booster as XGBoostBooster


class FeatureType(Enum):
    CAT = "categorical"
    CON = "continuous"
    BIN = "binary"


SKLearnParsableNode = int
LightGBMParsableNode = Mapping[str, Any]
XGBoostParsableNode = pd.Series
ParsableNode = SKLearnParsableNode | LightGBMParsableNode | XGBoostParsableNode

SKLearnParsableTree = Tree
LightGBMParsableTree = Mapping[str, Any]
XGBoostParsableTree = pd.DataFrame
ParsableTree = SKLearnParsableTree | LightGBMParsableTree | XGBoostParsableTree


BaseEnsemble = (
    RandomForestClassifier
    | AdaBoostClassifier
    | GradientBoostingClassifier
    | LightGBMBooster
    | XGBoostBooster
)

BaseDecisionTree = DecisionTreeClassifier | DecisionTreeRegressor


Class = np.intp
MClass = npt.NDArray[Class]

Prob = np.float64
MProb = npt.NDArray[Prob]

Number = np.float64
MNumber = npt.NDArray[Number]
SNumber = pd.Series
DNumber = pd.DataFrame

Categories = set[str]

LeafValue = Number | MNumber
Transformable = SNumber | list[SNumber] | DNumber
