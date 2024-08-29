from enum import Enum

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

numeric = int | float
Node = int
Sample = dict[str, numeric]
BaseEnsemble = (
    RandomForestClassifier | AdaBoostClassifier | GradientBoostingClassifier
)


class FeatureType(Enum):
    CATEGORICAL = 1
    CONTINUOUS = 2
    BINARY = 3
