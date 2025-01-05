from abc import abstractmethod
from typing import Protocol, TypeVar

from ..feature import FeatureEncoder
from ..tree import Tree
from ..tree.parsers.lgbm import LightGBMTreeParser
from ..tree.parsers.skl import SKLearnTreeParser
from ..tree.parsers.xgb import XGBoostTreeParser
from ..typing import (
    AdaBoostClassifier,
    BaseEnsemble,
    GradientBoostingClassifier,
    LightGBMBooster,
    RandomForestClassifier,
    XGBoostBooster,
)
from .binders.callback import BinderCallback
from .binders.cl import SKLearnBinderClassifier
from .binders.gb import GradientBoostingBinder
from .binders.lgbm import LightGBMBinder
from .binders.xgb import XGBoostBinder

Binder = (
    SKLearnBinderClassifier
    | GradientBoostingBinder
    | LightGBMBinder
    | XGBoostBinder
)
B = TypeVar("B", bound=Binder)

Parser = SKLearnTreeParser | LightGBMTreeParser | XGBoostTreeParser
P = TypeVar("P", bound=Parser)


class GenericBuilder(Protocol[B, P]):
    binder: B
    parser: P

    def __init__(self, binder: B, parser: P) -> None:
        self.binder = binder
        self.parser = parser

    @abstractmethod
    def parse_trees(self) -> tuple[Tree, ...]:
        raise NotImplementedError


class SKLearnBuilder(
    GenericBuilder[SKLearnBinderClassifier, SKLearnTreeParser]
):
    def parse_trees(self) -> tuple[Tree, ...]:
        return tuple(self.parser.parse(tree) for tree in self.binder.base_trees)


class GradientBoostingBuilder(
    GenericBuilder[GradientBoostingBinder, SKLearnTreeParser]
):
    def parse_trees(self) -> tuple[Tree, ...]:
        return tuple(self.parser.parse(tree) for tree in self.binder.base_trees)


class LightGBMBuilder(GenericBuilder[LightGBMBinder, LightGBMTreeParser]):
    def parse_trees(self) -> tuple[Tree, ...]:
        return tuple(self.parser.parse(tree) for tree in self.binder.base_trees)


class XGBoostBuilder(GenericBuilder[XGBoostBinder, XGBoostTreeParser]):
    def parse_trees(self) -> tuple[Tree, ...]:
        return tuple(self.parser.parse(tree) for tree in self.binder.base_trees)


Builder = (
    SKLearnBuilder | GradientBoostingBuilder | LightGBMBuilder | XGBoostBuilder
)


def create_builder(
    base: BaseEnsemble,
    encoder: FeatureEncoder,
    callback: BinderCallback,
) -> Builder:
    if isinstance(base, RandomForestClassifier):
        binder = SKLearnBinderClassifier(
            base,
            callback=callback,
            use_hard_voting=False,
        )
        parser = SKLearnTreeParser(encoder=encoder, use_hard_voting=False)
        return SKLearnBuilder(binder=binder, parser=parser)
    if isinstance(base, AdaBoostClassifier):
        binder = SKLearnBinderClassifier(
            base,
            callback=callback,
            use_hard_voting=True,
        )
        parser = SKLearnTreeParser(encoder=encoder, use_hard_voting=True)
        return SKLearnBuilder(binder=binder, parser=parser)
    if isinstance(base, GradientBoostingClassifier):
        binder = GradientBoostingBinder(base, callback=callback)
        parser = SKLearnTreeParser(encoder=encoder)
        return GradientBoostingBuilder(binder=binder, parser=parser)
    if isinstance(base, LightGBMBooster):
        binder = LightGBMBinder(base, callback=callback)
        parser = LightGBMTreeParser(encoder=encoder)
        return LightGBMBuilder(binder=binder, parser=parser)
    if isinstance(base, XGBoostBooster):
        binder = XGBoostBinder(base, callback=callback)
        parser = XGBoostTreeParser(encoder=encoder)
        return XGBoostBuilder(binder=binder, parser=parser)
    msg = f"Unsupported base ensemble: {type(base).__name__}"
    raise TypeError(msg)
