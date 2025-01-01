from ...typing import (
    AdaBoostClassifier,
    BaseEnsemble,
    Booster,
    GradientBoostingClassifier,
    LGBMClassifier,
    RandomForestClassifier,
)
from ..generic import Callback
from .ab import EnsembleAB
from .gb import EnsembleGB
from .lgbm import EnsembleLGBM
from .rf import EnsembleRF
from .xgb import EnsembleXGB

Ens = EnsembleAB | EnsembleGB | EnsembleLGBM | EnsembleRF | EnsembleXGB


def create_ensemble(base: BaseEnsemble, *, callback: Callback) -> Ens:
    if isinstance(base, RandomForestClassifier):
        return EnsembleRF(base, callback=callback)
    if isinstance(base, AdaBoostClassifier):
        return EnsembleAB(base, callback=callback)
    if isinstance(base, GradientBoostingClassifier):
        return EnsembleGB(base, callback=callback)
    if isinstance(base, LGBMClassifier):
        return EnsembleLGBM(base, callback=callback)
    if isinstance(base, Booster):
        return EnsembleXGB(base, callback=callback)
    msg = f"Unsupported base estimator: {type(base).__name__}"
    raise TypeError(msg)


__all__ = [
    "Ens",
    "create_ensemble",
]
