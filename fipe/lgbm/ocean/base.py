import gurobipy as gp
import numpy as np
from numpy.typing import ArrayLike

from ...feature import FeatureContainer, FeatureEncoder, FeatureVars
from ...mip import MIP
from ..ensemble import Ensemble, EnsembleContainer
from ..flow import FlowVars
from ..parsers import LevelParser

NUM_CLASSES_BINARY = 2


class BaseOCEAN(MIP, EnsembleContainer, FeatureContainer, LevelParser):
    _feature_vars: FeatureVars
    _flow_vars: dict[int, FlowVars]

    def __init__(
        self,
        encoder: FeatureEncoder,
        ensemble: Ensemble,
        weights: ArrayLike,
        **kwargs,
    ) -> None:
        MIP.__init__(
            self,
            name=kwargs.get("name", "OCEAN"),
            env=kwargs.get("env"),
        )
        EnsembleContainer.__init__(self, ensemble=ensemble, weights=weights)
        FeatureContainer.__init__(self, encoder=encoder)
        LevelParser.__init__(self, **kwargs)
        self.parse_levels(ensembles=[ensemble], encoder=encoder)

        self._feature_vars = FeatureVars(name="feature_vars")
        self._flow_vars = {}

    def build(self) -> None:
        self._build_features()
        self._build_flows()

    def _build_features(self) -> None:
        for feature in self.binary:
            self._feature_vars.add_binary(feature=feature)
        for feature in self.continuous:
            self._feature_vars.add_continuous(
                feature=feature,
                levels=self.levels[feature],
            )
        for feature in self.categorical:
            self._feature_vars.add_categorical(
                feature=feature,
                categories=self.categories[feature],
            )
        self._feature_vars.build(mip=self)

    def _build_flows(self) -> None:
        for t, tree in enumerate(self.ensemble):
            flow_vars = FlowVars(tree=tree, name=f"tree_{t}")
            flow_vars.build(mip=self)
            flow_vars.add_feature_constrs(
                mip=self,
                feature_vars=self._feature_vars,
            )
            self._flow_vars[t] = flow_vars

    def function(self, class_: int) -> gp.LinExpr:
        return self.weighted_function(class_=class_, weights=self._weights)

    def weighted_function(self, class_: int, weights: ArrayLike) -> gp.LinExpr:
        if self.n_classes == NUM_CLASSES_BINARY:
            return self._function_binary(class_=class_, weights=weights)
        return self._function_multi(class_=class_, weights=weights)

    def _function_binary(self, class_: int, weights: ArrayLike) -> gp.LinExpr:
        weights = np.asarray(weights)
        sign = 2 * class_ - 1
        return gp.quicksum(
            sign * weights[t] * self._flow_vars[t].value
            for t in range(self.n_estimators)
        )

    def _function_multi(self, class_: int, weights: ArrayLike) -> gp.LinExpr:
        weights = np.asarray(weights)
        return gp.quicksum(
            weights[t] * self._flow_vars[t * self.n_classes + class_].value
            for t in range(self.n_estimators)
        )
