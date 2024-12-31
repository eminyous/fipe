from enum import Enum

from ..typing import MNumber

Levels = MNumber
Categories = set[str]


class FeatureType(Enum):
    CAT = "categorical"
    CON = "continuous"
    BIN = "binary"
