from ...typing import (
    AdaBoostClassifier,
    BaseEnsemble,
    Booster,
    GradientBoostingClassifier,
    LGBMClassifier,
    RandomForestClassifier,
)
from ..binder import BinderCallback
from .cl import SKLearnBinderClassifier
from .gb import GradientBoostingBinder
from .lgbm import LightGBMBinder
from .xgb import XGBoostBinder

Binder = (
    SKLearnBinderClassifier
    | GradientBoostingBinder
    | LightGBMBinder
    | XGBoostBinder
)


def create_binder(
    base: BaseEnsemble,
    *,
    callback: BinderCallback,
) -> Binder:
    if isinstance(base, RandomForestClassifier):
        return SKLearnBinderClassifier(
            base,
            callback=callback,
            use_hard_voting=False,
        )
    if isinstance(base, AdaBoostClassifier):
        return SKLearnBinderClassifier(
            base,
            callback=callback,
            use_hard_voting=True,
        )
    if isinstance(base, GradientBoostingClassifier):
        return GradientBoostingBinder(base, callback=callback)
    if isinstance(base, LGBMClassifier):
        return LightGBMBinder(base, callback=callback)
    if isinstance(base, Booster):
        return XGBoostBinder(base, callback=callback)
    msg = f"Unsupported base estimator: {type(base).__name__}"
    raise TypeError(msg)


__all__ = [
    "Binder",
    "create_binder",
]
