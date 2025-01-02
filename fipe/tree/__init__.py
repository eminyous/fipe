from ..feature import FeatureEncoder
from ..typing import (
    AdaBoostClassifier,
    BaseEnsemble,
    GradientBoostingClassifier,
    LightGBMBooster,
    RandomForestClassifier,
    XGBoostBooster,
)
from .container import TreeContainer
from .parsers import (
    LightGBMTreeParser,
    SKLearnTreeParser,
    SKLearnTreeParserClassifier,
    XGBoostTreeParser,
)
from .tree import Tree

TreeParser = (
    SKLearnTreeParserClassifier
    | SKLearnTreeParser
    | LightGBMTreeParser
    | XGBoostTreeParser
)


def create_parser(base: BaseEnsemble, encoder: FeatureEncoder) -> TreeParser:
    if isinstance(base, RandomForestClassifier):
        return SKLearnTreeParserClassifier(
            encoder=encoder,
            use_hard_voting=False,
        )
    if isinstance(base, AdaBoostClassifier):
        return SKLearnTreeParserClassifier(
            encoder=encoder,
            use_hard_voting=True,
        )
    if isinstance(base, GradientBoostingClassifier):
        return SKLearnTreeParser(encoder=encoder)
    if isinstance(base, LightGBMBooster):
        return LightGBMTreeParser(encoder=encoder)
    if isinstance(base, XGBoostBooster):
        return XGBoostTreeParser(encoder=encoder)
    msg = f"Unsupported base estimator: {type(base).__name__}"
    raise TypeError(msg)


__all__ = [
    "LightGBMTreeParser",
    "SKLearnTreeParser",
    "SKLearnTreeParserClassifier",
    "Tree",
    "TreeContainer",
    "TreeParser",
    "XGBoostTreeParser",
    "create_parser",
]
