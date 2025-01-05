from abc import abstractmethod
from typing import Protocol, TypeVar

from ..feature import FeatureEncoder
from ..tree import Tree
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
from .parsers.lgbm import LightGBMParser
from .parsers.skl import SKLearnParser
from .parsers.xgb import XGBoostParser

Binder = (
    SKLearnBinderClassifier
    | GradientBoostingBinder
    | LightGBMBinder
    | XGBoostBinder
)
B = TypeVar("B", bound=Binder)

Parser = SKLearnParser | LightGBMParser | XGBoostParser
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


class SKLearnBuilder(GenericBuilder[SKLearnBinderClassifier, SKLearnParser]):
    def parse_trees(self) -> tuple[Tree, ...]:
        return tuple(self.parser.parse(tree) for tree in self.binder.base_trees)


class GradientBoostingBuilder(
    GenericBuilder[GradientBoostingBinder, SKLearnParser]
):
    def parse_trees(self) -> tuple[Tree, ...]:
        return tuple(self.parser.parse(tree) for tree in self.binder.base_trees)


class LightGBMBuilder(GenericBuilder[LightGBMBinder, LightGBMParser]):
    def parse_trees(self) -> tuple[Tree, ...]:
        return tuple(self.parser.parse(tree) for tree in self.binder.base_trees)


class XGBoostBuilder(GenericBuilder[XGBoostBinder, XGBoostParser]):
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
        parser = SKLearnParser(encoder=encoder, use_hard_voting=False)
        return SKLearnBuilder(binder=binder, parser=parser)
    if isinstance(base, AdaBoostClassifier):
        binder = SKLearnBinderClassifier(
            base,
            callback=callback,
            use_hard_voting=True,
        )
        parser = SKLearnParser(encoder=encoder, use_hard_voting=True)
        return SKLearnBuilder(binder=binder, parser=parser)
    if isinstance(base, GradientBoostingClassifier):
        binder = GradientBoostingBinder(base, callback=callback)
        parser = SKLearnParser(encoder=encoder)
        return GradientBoostingBuilder(binder=binder, parser=parser)
    if isinstance(base, LightGBMBooster):
        binder = LightGBMBinder(base, callback=callback)
        parser = LightGBMParser(encoder=encoder)
        return LightGBMBuilder(binder=binder, parser=parser)
    if isinstance(base, XGBoostBooster):
        binder = XGBoostBinder(base, callback=callback)
        parser = XGBoostParser(encoder=encoder)
        return XGBoostBuilder(binder=binder, parser=parser)
    msg = f"Unsupported base ensemble: {type(base).__name__}"
    raise TypeError(msg)
