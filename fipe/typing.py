from collections.abc import Mapping
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

ParsableTreeSKL = Tree
ParsableTreeLGBM = Mapping[str, Any]
ParsableTreeXGB = pd.DataFrame
ParsableTree = ParsableTreeSKL | ParsableTreeLGBM | ParsableTreeXGB

ParsableEnsemble = (
    RandomForestClassifier
    | AdaBoostClassifier
    | GradientBoostingClassifier
    | LGBMClassifier
    | Booster
)

Number = np.number
MNumber = npt.NDArray[Number]
SNumber = pd.Series

LeafValue = Number | MNumber
Variable = Number | MNumber | SNumber

PT = TypeVar("PT", bound=ParsableTree)
LV = TypeVar("LV", bound=LeafValue)
VT = TypeVar("VT", bound=Variable)
PE = TypeVar("PE", bound=ParsableEnsemble)
HV = TypeVar("HV", bound=bool)
