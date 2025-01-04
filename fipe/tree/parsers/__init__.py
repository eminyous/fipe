from ...feature import FeatureEncoder
from ...typing import (
    AdaBoostClassifier,
    BaseEnsemble,
    GradientBoostingClassifier,
    LightGBMBooster,
    RandomForestClassifier,
    XGBoostBooster,
)
from .lgbm import LightGBMTreeParser
from .skl import (
    SKLearnTreeMParser,
    SKLearnTreeParser,
)
from .xgb import XGBoostTreeParser

TreeParser = (
    SKLearnTreeMParser
    | SKLearnTreeParser
    | LightGBMTreeParser
    | XGBoostTreeParser
)


def create_parser(base: BaseEnsemble, encoder: FeatureEncoder) -> TreeParser:
    if isinstance(base, RandomForestClassifier):
        return SKLearnTreeMParser(
            encoder=encoder,
            use_hard_voting=False,
        )
    if isinstance(base, AdaBoostClassifier):
        return SKLearnTreeMParser(
            encoder=encoder,
            use_hard_voting=True,
        )
    if isinstance(base, GradientBoostingClassifier):
        return SKLearnTreeParser(encoder=encoder)
    if isinstance(base, LightGBMBooster):
        return LightGBMTreeParser(encoder=encoder)
    if isinstance(base, XGBoostBooster):
        return XGBoostTreeParser(encoder=encoder)
    msg = f"Unsupported base ensemble: {type(base).__name__}"
    raise TypeError(msg)


__all__ = [
    "TreeParser",
    "create_parser",
]
