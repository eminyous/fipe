from ...typing import (
    AdaBoostClassifier,
    BaseEnsemble,
    Booster,
    GradientBoostingClassifier,
    LGBMClassifier,
    RandomForestClassifier,
)
from ..binder import BinderCallback
from .ab import AdaBoostBinder
from .gb import GradientBoostingBinder
from .lgbm import LightGBMBinder
from .rf import RandomForestBinder
from .xgb import XGBoostBinder

Binder = (
    AdaBoostBinder
    | GradientBoostingBinder
    | LightGBMBinder
    | RandomForestBinder
    | XGBoostBinder
)


def create_binder(
    base: BaseEnsemble,
    *,
    callback: BinderCallback,
) -> Binder:
    if isinstance(base, RandomForestClassifier):
        return RandomForestBinder(base, callback=callback)
    if isinstance(base, AdaBoostClassifier):
        return AdaBoostBinder(base, callback=callback)
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
