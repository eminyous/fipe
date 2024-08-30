from enum import Enum

from numpy.typing import ArrayLike
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import Tree  # noqa: PLC2701

numeric = int | float
Node = int
Sample = dict[str, numeric]
BaseEnsemble = (
    RandomForestClassifier | AdaBoostClassifier | GradientBoostingClassifier
)
BaseEstimator = DecisionTreeClassifier | DecisionTreeRegressor
Weights = ArrayLike | dict[int, numeric]
BaseTree = Tree


class FeatureType(Enum):
    CAT = 1
    CON = 2
    BIN = 3
