from ..feature import FeatureEncoder
from ..typing import (
    AdaBoostClassifier,
    BaseEnsemble,
    Booster,
    GradientBoostingClassifier,
    LGBMClassifier,
    RandomForestClassifier,
)
from .container import TreeContainer
from .parsers import (
    TreeParserCL,
    LightGBMTreeParser,
    TreeParserRG,
    XGBoostTreeParser,
)
from .tree import Tree

TreeParser = (
    TreeParserCL | TreeParserRG | LightGBMTreeParser | XGBoostTreeParser
)


def create_parser(base: BaseEnsemble, encoder: FeatureEncoder) -> TreeParser:
    if isinstance(base, RandomForestClassifier):
        return TreeParserCL(encoder=encoder, use_hard_voting=False)
    if isinstance(base, AdaBoostClassifier):
        return TreeParserCL(encoder=encoder, use_hard_voting=True)
    if isinstance(base, GradientBoostingClassifier):
        return TreeParserRG(encoder=encoder)
    if isinstance(base, LGBMClassifier):
        return LightGBMTreeParser(encoder=encoder)
    if isinstance(base, Booster):
        return XGBoostTreeParser(encoder=encoder)
    msg = f"Unsupported base estimator: {type(base).__name__}"
    raise TypeError(msg)


__all__ = [
    "Tree",
    "TreeContainer",
    "TreeParser",
    "TreeParserCL",
    "LightGBMTreeParser",
    "TreeParserRG",
    "XGBoostTreeParser",
    "create_parser",
]
