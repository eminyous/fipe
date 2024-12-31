from collections.abc import Mapping
from enum import Enum
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree._tree import Tree  # noqa: PLC2701
from xgboost import Booster


class FeatureType(Enum):
    CAT = "categorical"
    CON = "continuous"
    BIN = "binary"


ParsableTreeSKL = Tree
ParsableTreeLGBM = Mapping[str, Any]
ParsableTreeXGB = pd.DataFrame
ParsableTree = ParsableTreeSKL | ParsableTreeLGBM | ParsableTreeXGB

BaseEnsemble = (
    RandomForestClassifier
    | AdaBoostClassifier
    | GradientBoostingClassifier
    | LGBMClassifier
    | Booster
)

Number = np.float64
MNumber = npt.NDArray[Number]
SNumber = pd.Series
DNumber = pd.DataFrame

Categories = set[str]

LeafValue = Number | MNumber
Variable = Number | MNumber | SNumber
Transformable = SNumber | list[SNumber] | DNumber

BE = TypeVar("BE", bound=BaseEnsemble)
PT = TypeVar("PT", bound=ParsableTree)
LV = TypeVar("LV", bound=LeafValue)
VT = TypeVar("VT", bound=Variable)
HV = TypeVar("HV", bound=bool)
